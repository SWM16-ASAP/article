import importlib
from functools import lru_cache

import boto3
from botocore.config import Config
from lingua import Language, LanguageDetectorBuilder
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

from ..utils.logging_config import get_logger
from ..utils.workflow_helpers import setup_bedrock, BedrockTokenTrackingWrapper
from ..utils.rag_tools import find_alternative_words
from ..utils.judge_ai import judge_text_for_level
from ..utils.langfuse_client import (
    get_langfuse_client,
    is_langfuse_enabled,
    get_prompt_label,
    convert_langfuse_to_langchain,
    get_model_config_from_prompt
)
from ..state import BookState

logger = get_logger(__name__)

# Define a client with extended timeouts for use in this node
_extended_timeout_config = Config(connect_timeout=120, read_timeout=600)
_boto_client = boto3.client("bedrock-runtime", config=_extended_timeout_config)

# 레벨별 temperature 설정
LEVEL_TEMPERATURES = {
    'A0': 0.3,  # 낮은 레벨 - 더 일관되고 안정적인 출력
    'A1': 0.3,
    'A2': 0.6,  # 중간 레벨 - 적당한 창의성
    'B1': 0.6,
    'B2': 0.6,
    'C1': 0.8,  # 높은 레벨 - 더 창의적이고 다양한 표현
    'C2': 0.8
}

# Initialize lingua detector once at module level for efficiency
_detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.KOREAN, Language.SPANISH, Language.FRENCH, Language.GERMAN, Language.ITALIAN, Language.PORTUGUESE, Language.RUSSIAN, Language.CHINESE, Language.JAPANESE).build()

def is_english_text(text: str) -> bool:
    """Checks if the given text is in English using lingua."""
    try:
        if not text or len(text.strip()) < 3:
            return False
        detected_language = _detector.detect_language_of(text)
        return detected_language == Language.ENGLISH
    except Exception as e:
        logger.warning(f"Language detection failed for text: {text[:50]}... Error: {e}")
        return False

@lru_cache(maxsize=128)
def get_prompt_for_level(level: str, use_rag: bool, is_feedback_generated: bool):
    """
    Dynamically loads and builds the prompt for the given CEFR level.

    Returns:
        tuple: (ChatPromptTemplate, model_config: dict or None)
        model_config = {"model": str, "temperature": float, "max_tokens": int}
    """
    # Langfuse에서 프롬프트 가져오기 시도
    if is_langfuse_enabled():
        try:
            client = get_langfuse_client()
            label = get_prompt_label()

            # 프롬프트 이름 구성
            prompt_name = f"{level.lower()}-leveling"
            if use_rag:
                prompt_name += "-rag"
            if is_feedback_generated:
                prompt_name += "-feedback"
            else:
                prompt_name += "-base"

            # Langfuse에서 프롬프트 가져오기
            langfuse_prompt = client.get_prompt(prompt_name, label=label)
            model_config = get_model_config_from_prompt(langfuse_prompt)
            logger.info(f"Loaded prompt '{prompt_name}' from Langfuse (label: {label}, version: {langfuse_prompt.version}, model: {model_config.get('model')})")

            # LangChain 형식으로 변환
            prompt_template = convert_langfuse_to_langchain(langfuse_prompt)
            return prompt_template, model_config
        except Exception as e:
            logger.warning(f"Failed to load prompt from Langfuse, falling back to local: {e}")

    # 로컬 프롬프트로 폴백
    try:
        prompt_module = importlib.import_module(f"novel_transformer_project.prompts.leveling.{level.lower()}_prompt")
        prompt_template = prompt_module.get_prompt(use_rag=use_rag, is_feedback_generated=is_feedback_generated)
        return prompt_template, None
    except ImportError as e:
        logger.error(f"Could not import prompt module for level '{level}': {e}")
        raise ValueError(f"Prompt module for level '{level}' not found.")

def _generate_with_rag(model_id: str, level: str, current_chunk: str, context: dict, is_feedback_generated: bool, state: BookState) -> tuple:
    """Generates text using the RAG agent for a specific level."""
    prompt, model_config = get_prompt_for_level(level, use_rag=True, is_feedback_generated=is_feedback_generated)

    # Langfuse config가 있으면 사용, 없으면 기존 하드코딩 값 사용
    if model_config and model_config.get("model"):
        logger.info(f"Using Langfuse config: model={model_config.get('model')}, temperature={model_config.get('temperature', 0.6)}")
        llm = setup_bedrock(config=model_config, client=_boto_client)
    else:
        temperature = LEVEL_TEMPERATURES.get(level.upper(), 0.6)
        config = {
            "model": model_id,
            "temperature": temperature,
            "max_tokens": 2000
        }
        llm = setup_bedrock(config=config, client=_boto_client)

    llm = BedrockTokenTrackingWrapper(llm, state)
    tools = [find_alternative_words]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True, max_iterations=3)
    
    response = agent_executor.invoke({"text": current_chunk, **context})
    
    # RAG agent의 경우 토큰 추출이 복잡하므로 임시로 0으로 설정
    # TODO: 실제 토큰 계산을 위해 callback handler 구현 필요
    generated_text = str(response.get("output", "")).strip()
    
    return generated_text

def _generate_without_rag(model_id: str, level: str, current_chunk: str, context: dict, is_feedback_generated: bool, state: BookState) -> tuple:
    """Generates text using a direct LLM call for a specific level."""
    prompt, model_config = get_prompt_for_level(level, use_rag=False, is_feedback_generated=is_feedback_generated)

    # Langfuse config가 있으면 사용, 없으면 기존 하드코딩 값 사용
    if model_config and model_config.get("model"):
        logger.info(f"Using Langfuse config: model={model_config.get('model')}, temperature={model_config.get('temperature', 0.5)}")
        llm = setup_bedrock(config=model_config, client=_boto_client)
    else:
        temperature = LEVEL_TEMPERATURES.get(level.upper(), 0.5)
        config = {
            "model": model_id,
            "temperature": temperature,
            "max_tokens": 2000
        }
        llm = setup_bedrock(config=config, client=_boto_client)

    llm = BedrockTokenTrackingWrapper(llm, state)

    chain = prompt | llm
    
    response = chain.invoke({"text": current_chunk, **context})
    
    generated_text = response.content.strip()
    
    return generated_text

def generate_text_for_level(level: str, use_rag: bool, current_chunk: str, context: dict, state) -> tuple:
    """
    Generates text for a specific CEFR level. This function is designed to be thread-safe.
    It takes all necessary data as arguments and returns only the generated text.
    """
    response_text = ""
    feedback_text = ""
    is_feedback_generated = False
    leveled_text_to_feedback = ""
    max_retries = 5
    model_id = "us.meta.llama4-scout-17b-instruct-v1:0"
    
    # 영어 텍스트를 통과한 결과들을 저장할 리스트
    valid_attempts = []
    
    for i in range(max_retries):
        try:
            prompt_context = context.copy()
            if is_feedback_generated:
                prompt_context['feedback'] = feedback_text
                prompt_context['leveled_text'] = leveled_text_to_feedback

            if use_rag:
                generated_text = _generate_with_rag(model_id, level, current_chunk, prompt_context, is_feedback_generated, state)
            else:
                generated_text = _generate_without_rag(model_id, level, current_chunk, prompt_context, is_feedback_generated, state)
            
            
            if not is_english_text(generated_text):
                logger.warning(f"Attempt {i+1}: Generated text for {level} is not in English. Retrying...")
                continue
            

            logger.debug(f"Attempt {i+1}: Meta commentary cleanup completed for {level} level")

            is_acceptable, judge_result = judge_text_for_level(
                level=level,
                original_text=current_chunk,
                leveled_text=generated_text,
                context=context,
                state=state
            )

            if is_acceptable:
                logger.info(f"Attempt {i+1}: {level} 레벨 생성 결과 평가 점수 {judge_result.get('overall_score', 0)}점")
                response_text = generated_text
                break
            else:
                score = judge_result.get('overall_score', 0)
                
                # 평가 점수가 0인 경우 특별 로깅
                if score == 0:
                    logger.warning(f"Attempt {i+1}: {level} 레벨 생성 결과 평가 점수 0점 - 정리된 텍스트: {generated_text}")
                else:
                    logger.warning(f"Attempt {i+1}: {level} 레벨 생성 결과 품질 불량 (점수: {score})")

                # 영어 텍스트를 통과했지만 퀄리티가 낮은 결과를 저장
                attempt_result = {
                    'text': generated_text,
                    'score': judge_result.get('overall_score', 0),
                    'is_acceptable': is_acceptable,
                    'judge_result': judge_result,
                    'attempt_number': i + 1
                }
                valid_attempts.append(attempt_result)

                feedback_text = judge_result.get("feedback", "No feedback provided.")
                leveled_text_to_feedback = generated_text
                is_feedback_generated = True

        except Exception as e:
            logger.error(f"Attempt {i+1}: Error during {level} generation: {e}. Retrying...")

    # 모든 시도가 끝난 후 처리
    if not response_text:  # 성공한 결과가 없는 경우
        if valid_attempts:
            # 영어 텍스트를 통과한 것들 중 점수가 가장 높은 것 선택
            best_attempt = max(valid_attempts, key=lambda x: x['score'])
            response_text = best_attempt['text']
            logger.warning(f"No acceptable text for {level} after {max_retries} attempts. "
                         f"Using best attempt (score {best_attempt['score']})")
        else:
            # 영어 텍스트를 통과한 것이 하나도 없는 경우
            response_text = f"[ERROR: Failed to generate English text for {level} after {max_retries} attempts.]"
            logger.error(f"Failed to generate any valid English text for {level} after {max_retries} attempts")

    # update_usage_metrics(state, model_id, response_input_tokens, response_output_tokens)

    return response_text