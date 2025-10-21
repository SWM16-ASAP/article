import json
import random
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException

from ..state import BookState
from ..utils.logging_config import get_logger
from ..utils.workflow_helpers import setup_bedrock, update_usage_metrics
from ..utils.langfuse_client import (
    get_langfuse_client,
    is_langfuse_enabled,
    get_prompt_label,
    convert_langfuse_to_langchain,
    get_model_config_from_prompt
)
from ..prompts.generate_cover_image_prompt import get_cover_image_prompt
import boto3

logger = get_logger(__name__)

# 이미지 프롬프트 생성을 위한 Pydantic 모델
class ImagePrompt(BaseModel):
    prompt: str = Field(description="Generated positive image prompt for cover generation")

def generate_cover_image(state: BookState) -> BookState:
    """
    Generates a cover image based on the content type (literature or article).
    
    This node performs a two-step process:
    1. Use a text model to generate a high-quality positive prompt.
    2. Use the Amazon Titan Image Generator with the generated positive prompt
       and a predefined negative prompt to create the actual image.
    """
    logger.info("=== 커버 이미지 생성 시작 ===")

    # --- Step 1: Generate the positive image prompt ---

    # Use a powerful text model for prompt generation
    model_id = "us.deepseek.r1-v1:0"
    llm = None

    # Langfuse에서 프롬프트 가져오기 시도
    content_type = state["content_type"]
    if is_langfuse_enabled():
        try:
            client = get_langfuse_client()
            label = get_prompt_label()
            prompt_name = f"generate-cover-image-prompt-{content_type}"
            langfuse_prompt = client.get_prompt(prompt_name, label=label)

            model_config = get_model_config_from_prompt(langfuse_prompt)
            llm = setup_bedrock(config=model_config)

            logger.info(f"Loaded '{prompt_name}' from Langfuse (label: {label}, version: {langfuse_prompt.version})")
            prompt_template = convert_langfuse_to_langchain(langfuse_prompt)
        except Exception as e:
            logger.warning(f"Failed to load prompt from Langfuse, falling back to local: {e}")
            prompt_template = get_cover_image_prompt(content_type)
    else:
        prompt_template = get_cover_image_prompt(content_type)

    if llm is None:
        config = {
            "model": model_id
        }
        llm = setup_bedrock(config=config)

    # Negative prompt text for the image generation model
    negative_prompt = """text, words, letters, watermark, signature, ugly, deformed, blurry, low quality, pixelated, noisy"""

    prompt_generation_chain = prompt_template | llm

    input_data = ""
    if state["content_type"] == "literature":
        # Combine all chapter summaries for the book summary
        full_summary = "\n\n".join(
            [ch.get("summary", "") for ch in state.get("chapter_metadata", []) if ch.get("summary")]
        )
        if not full_summary:
            logger.warning("문학 요약이 없어 커버 이미지 프롬프트 생성을 건너뜁니다.")
            return state
        input_data= full_summary
        logger.info("문학 요약을 기반으로 이미지 프롬프트 생성 중...")

    elif state["content_type"] == "article":
        # Combine all chapter contents for the article content
        input_data = state["chapters"][0]
        logger.info("기사 내용을 기반으로 이미지 프롬프트 생성 중...")
    else:
        logger.warning(f"지원되지 않는 콘텐츠 유형({state['content_type']})으로 커버 이미지 생성을 건너뜁니다.")
        return state

    # Generate the positive prompt string with retries
    max_tries = 3
    attempts = 0
    positive_prompt = ""

    # 파서 설정: 기본 파서와 자동 복구 파서
    base_parser = PydanticOutputParser(pydantic_object=ImagePrompt)
    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=llm,
        max_retries=1  # 복구 시도는 1회만
    )

    while attempts < max_tries:
        attempts += 1
        logger.info(f"긍정 프롬프트 생성 시도 #{attempts}")

        try:
            # 2. format_instructions를 프롬프트에 추가
            prompt_input = {
                "input": input_data,
                "format_instructions": base_parser.get_format_instructions()
            }

            # 3. LLM 체인 호출
            prompt_generation_chain = prompt_template | llm
            raw_response = prompt_generation_chain.invoke(prompt_input)

            # 5. Pydantic 파서로 응답 파싱 (OutputFixingParser 적용)
            try:
                # 1차 시도: 기본 파서
                response = base_parser.parse(raw_response.content)
            except OutputParserException as parse_error:
                logger.info(f"이미지 프롬프트 파싱 실패, OutputFixingParser로 복구 시도: {str(parse_error)[:50]}...")
                # 2차 시도: OutputFixingParser로 자동 복구
                response = fixing_parser.parse(raw_response.content)
                logger.info("OutputFixingParser를 통한 이미지 프롬프트 파싱 복구 성공")

            # 6. 파싱된 결과 사용
            positive_prompt = response.prompt
            logger.info(f"성공적으로 생성된 긍정 프롬프트: {positive_prompt}")
            break  # 성공했으므로 재시도 루프 탈출

        except OutputParserException as e:
            logger.warning(f"이미지 프롬프트 파싱 실패 (시도 {attempts}/{max_tries}): {e}")
            if attempts >= max_tries:
                logger.error("최대 재시도 횟수 초과. 커버 이미지 생성을 건너뜁니다.")
                return state
            continue

        except Exception as e:
            logger.error(f"이미지 프롬프트 생성 중 오류 (시도 {attempts}/{max_tries}): {e}")
            if attempts >= max_tries:
                logger.error("최대 재시도 횟수 초과. 커버 이미지 생성을 건너뜁니다.")
                return state
            continue

    # --- Step 2: Generate the image using Amazon Titan ---

    logger.info("Titan Image Generator로 커버 이미지 생성 중...")
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    image_model_id = "amazon.nova-canvas-v1:0"

    seed = random.randint(0, 858993460)
    image_num = 1

    # Use the more general Bedrock class for image generation to control the body
    native_request = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": positive_prompt,
            "negativeText": negative_prompt
        },
        "imageGenerationConfig": {
            "seed": seed,
            "quality": "standard",
            "height": 1024,
            "width" : 1024,
            "numberOfImages": image_num,
            "cfgScale": 8.0,
        }
    }

    request = json.dumps(native_request)

    response = client.invoke_model(modelId=image_model_id, body=request)

    model_response = json.loads(response.get("body").read())

    base64_image = model_response.get("images", [None])[0]

    if not base64_image:
        logger.error("이미지 생성에 실패했거나 응답에서 이미지를 찾을 수 없습니다.")
        return state

    # 토큰 사용량 추적 (이미지 생성은 보통 1개 단위로 계산)
    input_tokens = 0  # 텍스트 입력은 별도 계산하지 않음
    output_tokens = 1 # 생성된 이미지 수
    update_usage_metrics(state, image_model_id, input_tokens, output_tokens)

    # The response is a base64 encoded string, which we store directly
    # Note: The key in the state dictionary is "cover_image_url" but we store the base64 data.
    # This can be handled downstream (e.g., by saving to S3 and replacing with a URL).
    state["cover_image_url"] = base64_image

    logger.info("=== 커버 이미지 생성 완료 ===")

    return state
