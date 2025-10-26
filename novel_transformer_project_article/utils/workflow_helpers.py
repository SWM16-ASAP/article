import os
import pprint
import re
import requests
from typing import TYPE_CHECKING, Optional
from langchain_aws import ChatBedrock
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage
from ..state import BookState
from .logging_config import get_logger
from .exceptions import CostLimitExceededError
from typing import Any

 # 혹시 모를 circular import 방지를 위한 type hinting -> ide 확인용
if TYPE_CHECKING:
    from botocore.client import BaseClient

MODEL_PRICING = {
    "anthropic.claude-3-sonnet-20240229-v1:0": {
        "input": 0.000003, "output": 0.000015
    },
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "input": 0.00000025, "output": 0.00000125
    },
    "us.meta.llama4-maverick-17b-instruct-v1:0": {
        "input": 0.00000024, "output": 0.00000097
    },
    "us.deepseek.r1-v1:0": {
        "input": 0.00000135, "output": 0.0000054
    },
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {
        "input": 0.000003, "output": 0.000015
    },
    "amazon.nova-canvas-v1:0": { # 얘는 사진당 비용이라서 토큰별 비용이 아니다.  
        "input": 0.0, "output": 0.04
    },
    "us.meta.llama4-scout-17b-instruct-v1:0": {
        "input": 0.00000017, "output": 0.00000066
    },
    "us.meta.llama3-2-90b-instruct-v1:0": {
        "input": 0.00000072, "output": 0.00000072
    },
    "us.ai21.jamba-1-5-large-v1:0": {
        "input": 0.000002, "output": 0.000008
    },
    "us.mistral.pixtral-large-2502-v1:0": {
        "input": 0.000002, "output": 0.000006
    },
    "us.amazon.nova-lite-v1:0": {
        "input": 0.00000006, "output": 0.000000015
    }, 
    "openai/gpt-5-image-mini": { # 사진 아무리 비싸도 0.013 비슷하게 아래로 나온다.
        "input": 0, "output": 0.013
    }
    # Add other models here
}

logger = get_logger(__name__)

# JSON Parsing ------------------------------------------------------------

def clean_json_markdown(text: str | AIMessage) -> str:
    """
    마크다운 코드 블록에서 JSON 추출

    LLM이 ```json ... ``` 또는 ``` ... ``` 형태로 응답할 때
    마크다운 코드 블록을 제거하고 순수 JSON 문자열만 반환합니다.

    Args:
        text: 마크다운 코드 블록을 포함할 수 있는 텍스트 또는 AIMessage 객체

    Returns:
        순수 JSON 문자열 (마크다운 제거됨)

    Examples:
        >>> clean_json_markdown('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
        >>> clean_json_markdown('```\\n{"key": "value"}\\n```')
        '{"key": "value"}'
        >>> clean_json_markdown('{"key": "value"}')
        '{"key": "value"}'
        >>> from langchain_core.messages import AIMessage
        >>> clean_json_markdown(AIMessage(content='```json\\n{"key": "value"}\\n```'))
        '{"key": "value"}'
    """
    # AIMessage인 경우 content 추출
    if isinstance(text, AIMessage):
        text = text.content

    # ```json ... ``` 또는 ``` ... ``` 패턴 제거
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# Token ------------------------------------------------------------


def update_usage_metrics(state: BookState, model_id: str, input_tokens: int, output_tokens: int) -> None:
    """Updates the usage metrics for a specific model in the state."""
    if model_id not in state["usage_metrics"]:
        state["usage_metrics"][model_id] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0} 

    state["usage_metrics"][model_id]["input_tokens"] += input_tokens
    state["usage_metrics"][model_id]["output_tokens"] += output_tokens

    try:
        cost = MODEL_PRICING[model_id]["input"] * input_tokens + MODEL_PRICING[model_id]["output"] * output_tokens
    except KeyError:
        logger.warning(f"Model {model_id} not in MODEL_PRICING, skipping cost update")
        cost = 0.0
    state["usage_metrics"][model_id]["cost"] += cost
    state["total_cost"] += cost

class BedrockTokenTrackingWrapper(Runnable):
    """
    ChatBedrock 객체를 감싸서 토큰 사용량을 추적하는 래퍼 클래스.
    OutputFixingParser와 함께 사용하기 위해 만들어졌습니다.
    """
    def __init__(self, llm: ChatBedrock, state: BookState = None, auto_clean_json: bool = False):
        self.llm = llm
        self.state = state
        self.model_name = ""
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.auto_clean_json = auto_clean_json

    def invoke(self, input: Any, config=None, **kwargs) -> AIMessage:
        """LLM을 호출하고 토큰 사용량을 추적합니다."""
        self.clear_last_usage()
        response = self.llm.invoke(input, config=config, **kwargs)

        # 토큰 사용량 추출
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            self.last_input_tokens = response.usage_metadata.get("input_tokens", 0)
            self.last_output_tokens = response.usage_metadata.get("output_tokens", 0)

            # model_name은 response_metadata에 있음
            if hasattr(response, 'response_metadata') and response.response_metadata:
                self.model_name = response.response_metadata.get("model_name", "")

            if not self.model_name or self.last_input_tokens == 0 or self.last_output_tokens == 0:
                logger.warning(f"토큰 정보 일부 누락 - model: {self.model_name}, input: {self.last_input_tokens}, output: {self.last_output_tokens}")
                send_failure_notification(self.state, f"토큰 정보 일부 누락 - model: {self.model_name}, input: {self.last_input_tokens}, output: {self.last_output_tokens}")
        else:
            logger.error(f"usage_metadata 없음: {response}")
            send_failure_notification(self.state, f"usage_metadata 없음: {response}")

        # state가 있으면 토큰 사용량을 업데이트
        if self.state:
            update_usage_metrics(self.state, self.model_name, self.last_input_tokens, self.last_output_tokens)
            logger.debug(f"토큰 사용량 업데이트: {self.model_name} - 입력: {self.last_input_tokens}, 출력: {self.last_output_tokens}")

        # auto_clean_json이 True면 자동으로 clean_json_markdown 적용
        if self.auto_clean_json:
            cleaned_content = clean_json_markdown(response)
            # AIMessage 객체를 새로 만들어서 반환 (metadata는 유지)
            response = AIMessage(
                content=cleaned_content,
                response_metadata=response.response_metadata if hasattr(response, 'response_metadata') else {},
                usage_metadata=response.usage_metadata if hasattr(response, 'usage_metadata') else {}
            )

        return response

    def clear_last_usage(self):
        """마지막 사용량을 초기화합니다."""
        self.model_name = ""
        self.last_input_tokens = 0
        self.last_output_tokens = 0


# Bedrock ------------------------------------------------------------

def setup_bedrock(
    config: dict = None,
    client: Optional['BaseClient'] = None,
    guardrail_identifier: str = None,
    guardrail_version: str = None,
):
    """
    AWS Bedrock 클라이언트를 설정합니다.

    Args:
        model_id: 사용할 모델의 ID (config보다 우선순위 낮음)
        temperature: 모델의 생성 온도 (config보다 우선순위 낮음, 기본값 0.7)
        top_p: Top-p 샘플링 값 (기본값 0.9)
        client: (선택 사항) 미리 설정된 boto3 Bedrock-runtime 클라이언트
        guardrail_identifier: (선택 사항) Guardrail ID
        guardrail_version: (선택 사항) Guardrail 버전
        max_tokens: (선택 사항) 최대 토큰 수 (config보다 우선순위 낮음)
        config: Langfuse config dict (최우선 순위)
            {"model": str, "temperature": float, "max_tokens": int}

    Returns:
        ChatBedrock: 설정된 ChatBedrock 클라이언트.
    """
    # client가 None이면 자동으로 생성
    if client is None:
        import boto3
        region = os.environ.get("AWS_REGION", "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region)

    model_id = config.get("model")
    temperature = config.get("temperature", 0.7)
    top_p = config.get("top_p", 0.9)
    max_tokens = config.get("max_tokens")

    # temperature 기본값 설정
    if temperature is None:
        temperature = 0.7

    model_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    
    # Guardrail 설정이 있으면 guardrails 파라미터 사용
    guardrails_config = None
    if guardrail_identifier and guardrail_version:
        guardrails_config = {
            "guardrailIdentifier": guardrail_identifier,
            "guardrailVersion": guardrail_version,
            "trace": True
        }
        logger.info(f"Guardrail 설정 적용: {guardrail_identifier} v{guardrail_version}")
    
        return ChatBedrock(
            client=client,
            model_id=model_id,
            model_kwargs=model_kwargs,
            guardrails=guardrails_config,  # 올바른 Guardrail 설정 방법
        )
    else:
        return ChatBedrock(
            client=client,
            model_id=model_id,
            model_kwargs=model_kwargs,
        )


def send_failure_notification(state: BookState, error_message: str):
    """
    워크플로우 실패 알림을 백엔드 API로 전송합니다.
    FAIL_API_URL와 LINGLEVEL_API_KEY 환경 변수를 사용합니다.
    """

    title = state.get("title", "제목 없음")
    content_type = state.get("content_type", "알 수 없음")
    language = state.get("target_language_code")
    full_text = state.get("full_text")
    tag = state.get("tags")
        
    discord_message = f"""🚨 **워크플로우 실패 알림** 🚨
        
    **제목**: {title}
    **콘텐츠 타입**: {content_type}
    **언어**: {language}
    **전문**: {full_text}
    **태그**:{tag}
    **오류 메시지**: {str(error_message)[:500]}...

    운영 환경에서 워크플로우 실행 중 오류가 발생했습니다."""
        
    send_discord_webhook(discord_message)


def send_discord_webhook(message: str, webhook_url: str = None):
    """
    디스코드 웹훅으로 메시지를 전송합니다.
    DISCORD_WEBHOOK_URL 환경 변수를 사용합니다.
    """
    if not webhook_url:
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    
    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URL 환경 변수가 설정되지 않았습니다. 디스코드 웹훅을 건너뜁니다.")
        return
    
    payload = {
        "content": message,
        "username": "Ling-Level Cost Monitor"
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=15)
        response.raise_for_status()
        logger.info(f"디스코드 웹훅 전송 성공: {message[:50]}...")
    except requests.exceptions.RequestException as e:
        logger.error(f"디스코드 웹훅 전송 실패: {e}")

