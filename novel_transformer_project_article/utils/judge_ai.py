import importlib
import os
from functools import lru_cache
from typing import Dict, Any, Tuple

import boto3
from botocore.config import Config
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

from ..state import BookState
from ..utils.logging_config import get_logger
from ..utils.workflow_helpers import setup_bedrock, BedrockTokenTrackingWrapper, clean_json_markdown
from .langfuse_client import (
    get_langfuse_client,
    is_langfuse_enabled,
    get_prompt_label,
    convert_langfuse_to_langchain,
    get_model_config_from_prompt
)

logger = get_logger(__name__)

# Judge AI 응답을 위한 Pydantic 모델

class CriteriaScores(BaseModel):
    vocabulary: int = Field(ge=0, le=25, description="Score for vocabulary usage (0-25)")
    grammar: int = Field(ge=0, le=25, description="Score for grammar and sentence structure (0-25)")
    content_preservation: int = Field(ge=0, le=25, description="Score for preserving the original content's meaning (0-25)")
    context_continuity: int = Field(ge=0, le=25, description="Score for maintaining context continuity (0-25)")

class JudgeResult(BaseModel):
    overall_score: int = Field(ge=0, le=100, description="Overall quality score (0-100)")
    is_acceptable: bool = Field(description="Whether the text meets quality standards")
    criteria_scores: CriteriaScores = Field(description="Individual criteria scores (vocabulary, grammar, content_preservation, context_continuity)")
    feedback: str = Field(description="Detailed feedback on the text quality")

@lru_cache(maxsize=10)
def get_judge_prompt_for_level(level: str):
    """
    레벨별 Judge AI 프롬프트를 동적으로 로드합니다.

    Returns:
        tuple: (ChatPromptTemplate, model_config: dict or None)
        model_config = {"model": str, "temperature": float, "max_tokens": int}
    """
    # Langfuse에서 프롬프트 가져오기 시도
    if is_langfuse_enabled():
        try:
            client = get_langfuse_client()
            label = get_prompt_label()
            prompt_name = f"{level.lower()}-judge"

            # Langfuse에서 프롬프트 가져오기
            langfuse_prompt = client.get_prompt(prompt_name, label=label)
            model_config = get_model_config_from_prompt(langfuse_prompt)
            logger.info(f"Loaded judge prompt '{prompt_name}' from Langfuse (label: {label}, version: {langfuse_prompt.version}, model: {model_config.get('model')})")

            # LangChain 형식으로 변환
            prompt_template = convert_langfuse_to_langchain(langfuse_prompt)
            return prompt_template, model_config
        except Exception as e:
            logger.warning(f"Failed to load judge prompt from Langfuse, falling back to local: {e}")

    # 로컬 프롬프트로 폴백
    try:
        prompt_module = importlib.import_module(f"novel_transformer_project_article.prompts.judging.{level.lower()}_judge_prompt")
        prompt_template = prompt_module.get_judge_prompt()
        return prompt_template, None
    except ImportError as e:
        logger.error(f"Could not import judge prompt module for level '{level}': {e}")
        raise ValueError(f"Judge prompt module for level '{level}' not found.")

def judge_text_for_level(
    level: str, 
    original_text: str, 
    leveled_text: str, 
    context: Dict[str, str],
    state: BookState
) -> Tuple[bool, Dict[str, Any]]:
    """
    레벨링된 텍스트의 품질을 평가합니다.
    
    Args:
        level: CEFR 레벨 (A0, A1, A2, B1, B2, C1, C2)
        original_text: 원본 텍스트
        leveled_text: 레벨링된 텍스트
        context: 컨텍스트 정보 (cumulative_context, current_chapter_summary, previous_chunk_context)
        state: BookState 객체 (토큰 사용량 업데이트용)
    
    Returns:
        Tuple of (is_acceptable, judge_result)
        - is_acceptable: 품질이 수용 가능한지 (80점 이상)
        - judge_result: 상세 평가 결과 (Dict)
        
    Note:
        토큰 사용량은 함수 내부에서 직접 state에 업데이트됩니다.
    """
    max_retries = 3
    # total_input_tokens = 0
    # total_output_tokens = 0
    judge_result = None
    
    try:
        # Judge 프롬프트 로드
        judge_prompt, model_config = get_judge_prompt_for_level(level)

        # Bedrock LLM 설정
        timeout_config = Config(connect_timeout=120, read_timeout=600)
        region = os.environ.get("AWS_REGION", "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region, config=timeout_config)

        # Langfuse config가 있으면 사용, 없으면 기존 하드코딩 값 사용
        if model_config and model_config.get("model"):
            logger.info(f"Using Langfuse config: model={model_config.get('model')}, temperature={model_config.get('temperature', 0.4)}")
            llm = setup_bedrock(config=model_config, client=client)
        else:
            model_id = "us.amazon.nova-lite-v1:0"
            config = {
                "model": model_id,
                "temperature": 0.4
            }
            llm = setup_bedrock(config=config, client=client)

        llm = BedrockTokenTrackingWrapper(llm, state)

        # 파서 설정: 기본 파서와 자동 복구 파서
        base_parser = PydanticOutputParser(pydantic_object=JudgeResult)
        fixing_parser = OutputFixingParser.from_llm(
            parser=base_parser,
            llm=llm,
            max_retries=1  # 복구 시도는 1회만
        )
        
        # 재시도 로직
        for attempt in range(max_retries):
            try:
                # 2. format_instructions를 프롬프트에 추가
                prompt_variables = {
                    "original_text": original_text,
                    "leveled_text": leveled_text,
                    "cumulative_context": context.get("cumulative_context", "No previous chapters."),
                    "current_chapter_summary": context.get("current_chapter_summary", "No chapter summary available."),
                    "previous_chunk_context": context.get("previous_chunk_context", "This is the first chunk."),
                    "format_instructions": base_parser.get_format_instructions()
                }
                
                # 3. LLM 체인 호출
                llm_chain = judge_prompt | llm | clean_json_markdown
                response_text = llm_chain.invoke(prompt_variables)

                # # 4. 토큰 사용량 추출
                # input_tokens = 0
                # output_tokens = 0
                # if hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
                #     input_tokens = raw_response.usage_metadata.get('input_tokens', 0)
                #     output_tokens = raw_response.usage_metadata.get('output_tokens', 0)

                # total_input_tokens += input_tokens
                # total_output_tokens += output_tokens

                # 5. Pydantic 파서로 응답 파싱 (OutputFixingParser 적용)
                try:
                    # 1차 시도: 기본 파서
                    response = base_parser.parse(response_text)
                except OutputParserException as parse_error:
                    logger.info(f"Judge AI 파싱 실패, OutputFixingParser로 복구 시도: {str(parse_error)[:50]}...")
                    # 2차 시도: OutputFixingParser로 자동 복구
                    response = fixing_parser.parse(response_text)
                    logger.info("OutputFixingParser를 통한 파싱 복구 성공")
                
                # 6. 파싱된 결과를 딕셔너리로 변환
                judge_result = {
                    "overall_score": response.overall_score,
                    "is_acceptable": response.is_acceptable,
                    "criteria_scores": response.criteria_scores,
                    "feedback": response.feedback
                }
                
                logger.debug(f"Judge AI 평가 완료: {response.overall_score}점 (시도 {attempt+1}회)")
                break  # 성공했으므로 재시도 루프 탈출
                
            except OutputParserException as e:
                logger.warning(f"Judge AI 파싱 실패 (시도 {attempt+1}/{max_retries}회): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed for judge evaluation of {level}")
                    # 최종 시도 실패 시 기본값 설정
                    judge_result = {
                        "overall_score": 0,
                        "is_acceptable": False,
                        "criteria_scores": {
                            "vocabulary": 0,
                            "grammar": 0,
                            "content_preservation": 0,
                            "context_continuity": 0
                        },
                        "feedback": f"Judge evaluation failed after {max_retries} attempts. Last parsing error: {str(e)}"
                    }
                    break
                continue
                    
            except Exception as e:
                logger.error(f"Error during judge evaluation for {level} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # 모든 재시도 실패 시 기본값 설정
                    judge_result = {
                        "overall_score": 0,
                        "is_acceptable": False,
                        "criteria_scores": {
                            "vocabulary": 0,
                            "grammar": 0,
                            "content_preservation": 0,
                            "context_continuity": 0
                        },
                        "feedback": f"Judge evaluation failed after {max_retries} attempts. Last error: {str(e)}"
                    }
                    break
        
        if judge_result is None:
            # 예상치 못한 상황에 대한 안전장치
            judge_result = {
                "overall_score": 0,
                "is_acceptable": False,
                "criteria_scores": {
                    "vocabulary": 0,
                    "grammar": 0,
                    "content_preservation": 0,
                    "context_continuity": 0
                },
                "feedback": "Judge evaluation failed - no result generated"
            }
        
        is_acceptable = judge_result.get("is_acceptable", False)
        overall_score = judge_result.get("overall_score", 0)
        
        # 토큰 사용량을 state에 업데이트
        # update_usage_metrics(state, model_id, total_input_tokens, total_output_tokens)
        
        return is_acceptable, judge_result
        
    except Exception as e:
        logger.error(f"Critical error during judge evaluation for {level}: {e}")
        
        # 에러 발생시 기본 실패 결과 반환
        error_result = {
            "overall_score": 0,
            "is_acceptable": False,
            "criteria_scores": {
                "vocabulary": 0,
                "grammar": 0,
                "content_preservation": 0,
                "context_continuity": 0
            },
            "feedback": f"Judge evaluation failed due to critical error: {str(e)}"
        }
        
        # 토큰 사용량을 state에 업데이트 (에러 발생 시에도)
        # update_usage_metrics(state, model_id, total_input_tokens, total_output_tokens)
        
        return False, error_result