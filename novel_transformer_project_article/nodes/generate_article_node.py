import json
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException
from lingua import Language, LanguageDetectorBuilder

from ..state import BookState
from ..utils.logging_config import get_logger
from ..utils.workflow_helpers import setup_bedrock, BedrockTokenTrackingWrapper
from ..utils.langfuse_client import (
    get_langfuse_client,
    is_langfuse_enabled,
    get_prompt_label,
    convert_langfuse_to_langchain,
    get_model_config_from_prompt
)
from ..prompts.generate_article_prompt import get_article_generation_prompt
from ..prompts.expand_article_prompt import get_article_expansion_prompt

logger = get_logger(__name__)

# Initialize lingua detector once at module level for efficiency
_detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.KOREAN,
    Language.SPANISH,
    Language.FRENCH,
    Language.GERMAN,
    Language.ITALIAN,
    Language.PORTUGUESE,
    Language.RUSSIAN,
    Language.CHINESE,
    Language.JAPANESE
).build()

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

# 상수 정의
MIN_ARTICLE_LENGTH = 1000
MAX_ARTICLE_LENGTH = 1500
ARTICLE_MODEL_ID = "us.meta.llama4-maverick-17b-instruct-v1:0"

# 기사 생성 결과를 위한 Pydantic 모델
class ArticleGeneration(BaseModel):
    title: str = Field(max_length=60, description="headline for the article")
    content: str = Field(max_length=1500, description="The main content of the article, under 1500 characters")
    # min_length 제거: 파싱 후 수동으로 길이 검증하여 OutputFixingParser로의 즉시 전달 방지


def _expand_short_article(
    short_content: str,
    title: str,
    topic_content: str,
    state: BookState,
    target_length: int = MIN_ARTICLE_LENGTH
) -> str:
    """짧은 기사 내용을 확장하여 최소 길이 요구사항을 충족시킵니다."""
    logger.info(f"기사 내용이 너무 짧습니다 ({len(short_content)}자). 확장 시도 중...")

    llm = None

    # Langfuse에서 프롬프트 가져오기 시도
    if is_langfuse_enabled():
        try:
            client = get_langfuse_client()
            label = get_prompt_label()
            langfuse_prompt = client.get_prompt("expand-article", label=label)

            model_config = get_model_config_from_prompt(langfuse_prompt)
            llm = setup_bedrock(config=model_config)

            logger.info(f"Loaded 'expand-article' from Langfuse (label: {label}, version: {langfuse_prompt.version})")
            prompt = convert_langfuse_to_langchain(langfuse_prompt)
        except Exception as e:
            logger.warning(f"Failed to load prompt from Langfuse, falling back to local: {e}")
            prompt = get_article_expansion_prompt()
    else:
        prompt = get_article_expansion_prompt()

    if llm is None:
        config = {
            "model": ARTICLE_MODEL_ID,
            "temperature": 0.4
        }
        llm = setup_bedrock(config=config)

    llm = BedrockTokenTrackingWrapper(llm, state)

    base_parser = PydanticOutputParser(pydantic_object=ArticleGeneration)
    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=llm,
        max_retries=1
    )

    chain = prompt | llm

    raw_response = chain.invoke({
        "current_length": len(short_content),
        "target_length": target_length,
        "max_length": MAX_ARTICLE_LENGTH,
        "title": title,
        "topic_content": topic_content,
        "short_content": short_content,
        "format_instructions": base_parser.get_format_instructions()
    })

    try:
        # 1차 시도: 기본 파서
        response = base_parser.parse(raw_response.content)
    except OutputParserException as parse_error:
        logger.info(f"확장된 기사 파싱 실패, OutputFixingParser로 복구 시도: {str(parse_error)[:50]}...")
        # 2차 시도: OutputFixingParser로 자동 복구
        response = fixing_parser.parse(raw_response.content)
        logger.info("확장된 기사 OutputFixingParser를 통한 파싱 복구 성공")

    expanded_content = response.content
    logger.info(f"기사 확장 완료: {len(short_content)}자 -> {len(expanded_content)}자")
    return expanded_content


def generate_article(state: BookState) -> BookState:
    """뉴스 기사를 생성합니다."""
    try:
            logger.info("=== 뉴스 기사 생성 시작 ===")
            model_id = "us.meta.llama3-2-90b-instruct-v1:0"
            llm = None

            state["original_text_level"] = "C1"

            # tags를 확인하여 FunFact 카테고리인지 판별
            tags = state.get("tags", [])
            is_lifestyle = "FunFact" in tags
            logger.info(f"카테고리: {tags}, 라이프스타일 모드: {is_lifestyle}")

            # Langfuse에서 프롬프트 가져오기 시도
            if is_langfuse_enabled():
                try:
                    client = get_langfuse_client()
                    label = get_prompt_label()

                    # 카테고리에 따라 프롬프트 이름 결정
                    prompt_name = "generate-article-funfact" if is_lifestyle else "generate-article"

                    langfuse_prompt = client.get_prompt(prompt_name, label=label)

                    model_config = get_model_config_from_prompt(langfuse_prompt)
                    llm = setup_bedrock(config=model_config)

                    logger.info(f"Loaded '{prompt_name}' from Langfuse (label: {label}, version: {langfuse_prompt.version})")
                    prompt_template = convert_langfuse_to_langchain(langfuse_prompt)
                except Exception as e:
                    logger.warning(f"Failed to load prompt from Langfuse, falling back to local: {e}")
                    prompt_template = get_article_generation_prompt(is_lifestyle=is_lifestyle)
            else:
                prompt_template = get_article_generation_prompt(is_lifestyle=is_lifestyle)

            if llm is None:
                config = {
                    "model": model_id,
                    "temperature": 0.4
                }
                llm = setup_bedrock(config=config)

            llm = BedrockTokenTrackingWrapper(llm, state)

            max_retries = 3

            # 파서 설정: 기본 파서와 자동 복구 파서
            base_parser = PydanticOutputParser(pydantic_object=ArticleGeneration)
            fixing_parser = OutputFixingParser.from_llm(
                parser=base_parser,
                llm=llm,
                max_retries=1  # 복구 시도는 1회만
            )

            # 2. format_instructions를 프롬프트에 추가
            prompt_input = {
                "topic_content": state["chapters"][0],
                "format_instructions": base_parser.get_format_instructions()
            }

            logger.info(f"Processing {len(state['chapters'])} chapter(s) with total length: {len(state['chapters'][0])} characters")
            logger.info(f"Source preview: {state['chapters'][0][:100]}...")

            for attempt in range(max_retries):
                try:
                    # 3. LLM 체인 호출
                    llm_chain = prompt_template | llm
                    raw_response = llm_chain.invoke(prompt_input)
                    
                    # 5. Pydantic 파서로 응답 파싱 (OutputFixingParser 적용)
                    try:
                        # 1차 시도: 기본 파서
                        response = base_parser.parse(raw_response.content)
                    except OutputParserException as parse_error:
                        logger.info(f"기사 생성 파싱 실패, OutputFixingParser로 복구 시도: {str(parse_error)[:50]}...")
                        # 2차 시도: OutputFixingParser로 자동 복구
                        response = fixing_parser.parse(raw_response.content)
                        logger.info("OutputFixingParser를 통한 파싱 복구 성공")

                    # 5.5 lingua로 title, content 영어인지 검증
                    if not is_english_text(response.title):
                        logger.warning(f"Attempt {attempt+1}: Generated title is not in English: {response.title[:50]}... Retrying...")
                        continue

                    if not is_english_text(response.content):
                        logger.warning(f"Attempt {attempt+1}: Generated content is not in English. Retrying...")
                        continue

                    logger.info(f"Attempt {attempt+1}: Title and content language validation passed")

                    # 6. 수동 길이 검증 및 확장 로직
                    article_content = response.content.strip()
                    if len(article_content) < MIN_ARTICLE_LENGTH:
                        logger.warning(f"기사 내용이 최소 길이({MIN_ARTICLE_LENGTH}자)보다 짧습니다 ({len(article_content)}자). 확장 시도 중...")
                        article_content = _expand_short_article(
                            short_content=article_content,
                            title=response.title,
                            topic_content=state["chapters"][0],
                            state=state,
                            target_length=MIN_ARTICLE_LENGTH
                        )
                        
                        # 확장 후에도 여전히 짧으면
                        if len(article_content) < MIN_ARTICLE_LENGTH:
                            # 마지막 시도가 아니면 재시도
                            if attempt < max_retries - 1:
                                logger.warning(f"확장 후에도 최소 길이 미달 ({len(article_content)}자). 재시도합니다 (시도 {attempt+1}/{max_retries})")
                                continue
                            # 마지막 시도였으면 경고만 하고 그대로 사용
                            else:
                                logger.warning(f"기사 생성 경고: {max_retries}회 재시도 및 확장 후에도 최소 길이 미달 ({len(article_content)}자 < {MIN_ARTICLE_LENGTH}자), 그대로 진행합니다")
                    
                    # 7. 파싱 및 검증된 결과를 state에 적용
                    state["title"] = response.title
                    state["chapters"][0] = article_content
                    
                    logger.debug(f"기사 생성 완료: {response.title} (시도 {attempt+1}회, 최종 길이: {len(article_content)}자)")
                    break  # 성공했으므로 재시도 루프 탈출
                    
                except OutputParserException as e:
                    logger.warning(f"기사 생성 파싱 실패 (시도 {attempt+1}/{max_retries}회): {e}")
                    if attempt == max_retries - 1:
                        error_msg = f"기사 생성 최종 실패: {max_retries}회 재시도 후에도 파싱 실패"
                        logger.error(error_msg)
                        raise ValueError(error_msg) from e
                    continue
                    
                except Exception as e:
                    logger.error(f"기사 생성 중 오류 (시도 {attempt+1}/{max_retries}회): {e}")
                    if attempt == max_retries - 1:
                        error_msg = f"기사 생성 최종 실패: {max_retries}회 재시도 후에도 생성 실패"
                        logger.error(error_msg)
                        raise ValueError(error_msg) from e
                    continue

            # update_usage_metrics(state, model_id, total_input_tokens, total_output_tokens)
            logger.info("=== 뉴스 기사 생성 완료 ===")
            return state
    finally:
        pass
