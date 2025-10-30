import json
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
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
logger = get_logger(__name__)

# Initialize lingua detector once at module level for efficiency
_detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.KOREAN,
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
    title: str = Field(description="headline for the article")
    content: str = Field(description="The main content of the article, under 1500 characters")
    # min_length 제거: 파싱 후 수동으로 길이 검증하여 OutputFixingParser로의 즉시 전달 방지

# 길이 조정용 Pydantic 모델 (content만 반환)
class TextContent(BaseModel):
    content: str = Field(description="The adjusted text content")

def validate_length(article: ArticleGeneration, state: BookState, max_attempts: int = 3) -> ArticleGeneration:
    """
    기사의 제목과 내용의 길이를 검증 및 확장, 요약을 합니다.
    조건을 만족하지 못할 경우 최대 max_attempts번까지 재시도합니다.

    검증 종류
    1. title이 long
    2. content가 short
    3. content가 long

    타이틀 기준
    - 요구 : 60자 이내
    - 허용 범위 : 80자 이내

    내용 기준
    - 요구 : 1000 - 1500자
    - 허용 : 900 - 1600자

    Args:
        article: 검증할 기사 객체
        state: 워크플로우 상태
        max_attempts: 최대 재시도 횟수 (기본값: 3)

    Returns:
        길이가 조정된 기사 객체 (실패 시에도 반환)
    """
    # LLM 설정 (공통)
    config = {
        "model": ARTICLE_MODEL_ID,
        "temperature": 0.3,
        "max_tokens": 2000
    }
    llm = setup_bedrock(config=config)
    llm = BedrockTokenTrackingWrapper(llm, state, auto_clean_json=True)

    # 파서 설정
    base_parser = PydanticOutputParser(pydantic_object=TextContent)
    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=llm,
        max_retries=1
    )

    # 공통 프롬프트
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a professional editor. Condense the following text to approximately {target_length} characters while preserving the core message.
        You MUST return title and content in ENGLISH.

        {format_instructions}"""),
        ("human", "Text ({current_length} characters):\n{text}\n\nCondense to approximately {target_length} characters. \n\n Don't condense text too short!!")
    ]).partial(format_instructions=base_parser.get_format_instructions())

    expand_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a professional writer. Expand the following text to approximately {target_length} characters by adding more details, context, and information.
        You MUST return title and content in ENGLISH.

        Use the background knowledge below to add relevant details:

        Background Knowledge:
        {background_knowledge}

        {format_instructions}"""),
        ("human", "Text ({current_length} characters):\n{text}\n\nExpand to approximately {target_length} characters.")
    ]).partial(format_instructions=base_parser.get_format_instructions())

    def adjust_text(text: str, target_length: int, is_expand: bool, is_title: bool = False) -> str:
        """텍스트를 요약 또는 확장합니다."""
        action = "확장" if is_expand else "요약"
        text_type = "제목" if is_title else "내용"

        # 프롬프트 선택
        prompt = expand_prompt if is_expand else summarize_prompt
        chain = prompt | llm

        # 파라미터 설정
        invoke_params = {
            "current_length": len(text),
            "target_length": target_length,
            "text": text
        }

        # 확장이면서 제목이 아닌 경우에만 배경 지식 추가
        if is_expand and not is_title:
            invoke_params["background_knowledge"] = state.get("full_text", "")

        # LLM 호출
        raw_response = chain.invoke(invoke_params)

        # 파싱
        try:
            response = base_parser.parse(raw_response.content)
        except OutputParserException as e:
            logger.info(f"{text_type} {action} 파싱 실패, OutputFixingParser로 복구 시도: {str(e)[:50]}...")
            response = fixing_parser.parse(raw_response.content)
            logger.info("OutputFixingParser를 통한 파싱 복구 성공")

        return response.content.strip()

    # 0. 제목, 내용 영어인지 검증
    if not is_english_text(article.title):
        error_msg = f"제목이 영어가 아닙니다: {article.title[:50]}..."
        logger.warning(error_msg)
        raise ValueError(error_msg)

    if not is_english_text(article.content):
        error_msg = "내용이 영어가 아닙니다"
        logger.warning(error_msg)
        raise ValueError(error_msg)

    # 1. 제목 길이 검증 및 조정 (재시도 로직)
    for attempt in range(1, max_attempts + 1):
        if len(article.title) <= 80:
            break  # 조건 만족

        original_length = len(article.title)
        logger.info(f"제목 길이 조정 시도 {attempt}/{max_attempts}: {original_length}자 -> 60자 목표")

        article.title = adjust_text(article.title, 60, is_expand=False, is_title=True)
        logger.info(f"제목 요약 완료: {original_length}자 -> {len(article.title)}자")

        # 제목 조정 후 영어인지 검증
        if not is_english_text(article.title):
            error_msg = f"제목 길이 조정 후 영어가 아님 (시도 {attempt}/{max_attempts}): {article.title[:50]}..."
            logger.warning(error_msg)
            if attempt == max_attempts:
                raise ValueError(error_msg)
            continue

        # 마지막 시도에서도 조건 불만족 시 경고만 출력
        if attempt == max_attempts and len(article.title) > 80:
            logger.warning(f"제목 길이 조정 실패: {max_attempts}번 시도 후에도 80자 초과 ({len(article.title)}자), 그대로 진행합니다")

    # 2. 내용 길이 검증 및 조정 (재시도 로직)
    for attempt in range(1, max_attempts + 1):
        content_length = len(article.content)

        # 조건 만족 확인
        if 900 <= content_length <= 1600:
            logger.info(f"내용 길이 조건 만족: {content_length}자")
            break

        # 너무 짧은 경우 확장
        if content_length < 900:
            logger.info(f"내용 확장 시도 {attempt}/{max_attempts}: {content_length}자 -> 1200자 목표")
            article.content = adjust_text(article.content, 1200, is_expand=True, is_title=False)
            logger.info(f"내용 확장 완료: {content_length}자 -> {len(article.content)}자")

            # 내용 조정 후 영어인지 검증
            if not is_english_text(article.content):
                error_msg = f"내용 확장 후 영어가 아님 (시도 {attempt}/{max_attempts})"
                logger.warning(error_msg)
                if attempt == max_attempts:
                    raise ValueError(error_msg)
                continue

        # 너무 긴 경우 요약
        elif content_length > 1600:
            logger.info(f"내용 요약 시도 {attempt}/{max_attempts}: {content_length}자 -> 1200자 목표")
            article.content = adjust_text(article.content, 1200, is_expand=False, is_title=False)
            logger.info(f"내용 요약 완료: {content_length}자 -> {len(article.content)}자")

            # 내용 조정 후 영어인지 검증
            if not is_english_text(article.content):
                error_msg = f"내용 요약 후 영어가 아님 (시도 {attempt}/{max_attempts})"
                logger.warning(error_msg)
                if attempt == max_attempts:
                    raise ValueError(error_msg)
                continue

        # 마지막 시도에서도 조건 불만족 시 경고만 출력
        if attempt == max_attempts:
            final_length = len(article.content)
            if final_length < 900 or final_length > 1600:
                logger.warning(f"내용 길이 조정 실패: {max_attempts}번 시도 후에도 범위 벗어남 ({final_length}자, 허용: 900-1600자), 그대로 진행합니다")

    return article



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
                    "temperature": 0.4,
                    "max_tokens": 2000
                }
                llm = setup_bedrock(config=config)

            llm = BedrockTokenTrackingWrapper(llm, state, auto_clean_json=True)

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
                "topic_content": state["full_text"],
                "format_instructions": base_parser.get_format_instructions()
            }

            logger.info(f"Processing {len(state['full_text'])} chapter(s) with total length: {len(state['full_text'])} characters")
            logger.info(f"Source preview: {state['full_text'][:100]}...")

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

                    # 5.5 title, content 빈 문자열 및 영어인지 검증
                    title_stripped = response.title.strip()
                    content_stripped = response.content.strip()

                    # 빈 문자열 검사
                    if not title_stripped or title_stripped.lower().startswith("example"):
                        error_msg = f"Attempt {attempt+1}: Generated title is empty or starts with 'example': {title_stripped[:50]}..."
                        logger.warning(error_msg)
                        if attempt == max_retries - 1:
                            raise ValueError(error_msg)
                        continue

                    if not content_stripped:
                        error_msg = f"Attempt {attempt+1}: Generated content is empty"
                        logger.warning(error_msg)
                        if attempt == max_retries - 1:
                            raise ValueError(error_msg)
                        continue

                    # 영어 검증
                    if not is_english_text(title_stripped):
                        error_msg = f"Attempt {attempt+1}: Generated title is not in English: {title_stripped[:50]}..."
                        logger.warning(error_msg)
                        if attempt == max_retries - 1:
                            raise ValueError(error_msg)
                        continue

                    if not is_english_text(content_stripped):
                        error_msg = f"Attempt {attempt+1}: Generated content is not in English"
                        logger.warning(error_msg)
                        if attempt == max_retries - 1:
                            raise ValueError(error_msg)
                        continue

                    logger.info(f"Attempt {attempt+1}: Title and content validation passed (non-empty and English)")

                    # 6. 수동 길이 검증 및 확장 로직
                    response = validate_length(response, state)
                    
                    # 7. 파싱 및 검증된 결과를 state에 적용
                    state["title"] = response.title
                    state["chapters"] = [response.content]
                    
                    logger.info(f"기사 생성 완료: {response.title} (시도 {attempt+1}회, 최종 길이: {len(response.content)}자)")
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

            if state["target_language_code"] is not None and state["tags"][0] != "Culture" :
                state["target_language_code"] = None

            logger.info("=== 뉴스 기사 생성 완료 ===")
            return state
    finally:
        pass
