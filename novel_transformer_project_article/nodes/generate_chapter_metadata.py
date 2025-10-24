from ..state import BookState
from ..utils.logging_config import get_logger
from ..utils.workflow_helpers import (
    setup_bedrock,
    BedrockTokenTrackingWrapper,
    clean_json_markdown,
)
from ..utils.langfuse_client import (
    get_langfuse_client,
    is_langfuse_enabled,
    get_prompt_label,
    convert_langfuse_to_langchain,
    get_model_config_from_prompt
)
from ..prompts.generate_chapter_metadata_prompt import get_metadata_prompt
import json
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException


# 제목과 요약을 모두 받아야 할 때 사용하는 모델
class ChapterMetadata(BaseModel):
    title: str = Field(
        description="A concise, engaging chapter title (max 50 characters)"
    )
    summary: str = Field(
        description="A brief summary of the chapter's key events and developments (max 200 characters)"
    )




logger = get_logger(__name__)


def generate_chapter_metadata(state: BookState) -> BookState:
    """각 챕터의 메타데이터(제목, 요약)를 생성합니다."""
    try:
        logger.info("=== 챕터 메타데이터 생성 시작 ===")
        model_id = "us.meta.llama4-maverick-17b-instruct-v1:0"
        llm = None

        for i, metadata_item in enumerate(state["chapter_metadata"]):
            logger.debug(f"챕터 {i+1} 메타데이터 생성 중...")

            chapter_content = state["chapters"][i]

            # Langfuse에서 프롬프트 가져오기 시도
            if is_langfuse_enabled():
                try:
                    client = get_langfuse_client()
                    label = get_prompt_label()
                    langfuse_prompt = client.get_prompt("generate-chapter-metadata", label=label)

                    model_config = get_model_config_from_prompt(langfuse_prompt)
                    llm = setup_bedrock(config=model_config)

                    logger.info(f"Loaded 'generate-chapter-metadata' from Langfuse (label: {label}, version: {langfuse_prompt.version})")
                    prompt_template = convert_langfuse_to_langchain(langfuse_prompt)
                except Exception as e:
                    logger.warning(f"Failed to load prompt from Langfuse, falling back to local: {e}")
                    prompt_template = get_metadata_prompt()
            else:
                prompt_template = get_metadata_prompt()

            if llm is None:
                config = {
                    "model": model_id
                }
                llm = setup_bedrock(config=config)

            llm = BedrockTokenTrackingWrapper(llm, state)

            # 최대 3번 재시도
            max_retries = 3
            # total_input_tokens = 0
            # total_output_tokens = 0

            # 파서 설정: 기본 파서와 자동 복구 파서
            base_parser = PydanticOutputParser(pydantic_object=ChapterMetadata)
            
            fixing_parser = OutputFixingParser.from_llm(
                parser=base_parser, llm=llm, max_retries=1  # 복구 시도는 1회만
            )

            for attempt in range(max_retries):
                try:
                    # 2. ⭐️ 생성된 파서로부터 format_instructions를 직접 가져와서 input에 추가합니다.
                    prompt_input = {
                        "chapter_text": chapter_content,
                        "format_instructions": base_parser.get_format_instructions(),
                    }
                    
                    # 3. 이제 모든 변수가 채워진 상태로 체인을 호출합니다.
                    llm_chain = prompt_template | llm | clean_json_markdown
                    response_text = llm_chain.invoke(prompt_input)

                    # 4. 토큰 정보를 추출합니다.
                    # input_tokens = 0
                    # output_tokens = 0
                    # if hasattr(raw_response, 'usage_metadata') and raw_response.usage_metadata:
                    # input_tokens = raw_response.usage_metadata.get('input_tokens', 0)
                    # output_tokens = raw_response.usage_metadata.get('output_tokens', 0)

                    # total_input_tokens += input_tokens
                    # total_output_tokens += output_tokens

                    # 5. LLM의 텍스트 내용을 파싱합니다 (OutputFixingParser 적용)
                    try:
                        # 1차 시도: 기본 파서
                        response = base_parser.parse(response_text)
                    except OutputParserException as parse_error:
                        logger.info(
                            f"챕터 {i+1} 파싱 실패, OutputFixingParser로 복구 시도: {str(parse_error)[:50]}..."
                        )
                        # 2차 시도: OutputFixingParser로 자동 복구
                        response = fixing_parser.parse(response_text)
                        logger.info(
                            f"챕터 {i+1} OutputFixingParser를 통한 파싱 복구 성공"
                        )

                    # 6. 파싱된 결과를 사용합니다.
                    metadata_item["title"] = response.title
                    metadata_item["summary"] = response.summary
                    
                    if (
                        state["content_type"] == "custom"
                        and state["title"] is None
                    ):
                        state["title"] = metadata_item["title"]
                    
                    logger.debug(
                        f"챕터 {i+1} 메타데이터 완료: {metadata_item.get('title', 'N/A')} (시도 {attempt+1}회)"
                    )
                    break  # 성공했으므로 재시도 루프 탈출

                except OutputParserException as e:
                    logger.warning(
                        f"챕터 {i+1} 파싱 실패 (시도 {attempt+1}/{max_retries}회): {e}"
                    )
                    if attempt == max_retries - 1:
                        logger.error(f"챕터 {i+1} 메타데이터 생성 최종 실패")
                except Exception as e:
                    logger.error(
                        f"챕터 {i+1} 메타데이터 생성 중 오류 (시도 {attempt+1}/{max_retries}회): {e}"
                    )
                    if attempt == max_retries - 1:
                        logger.error(
                            f"챕터 {i+1} 메타데이터 생성 실패: 모든 재시도 실패"
                        )
                        # 기본값은 이미 설정됨
            
            # 누적된 토큰 사용량 업데이트
            # update_usage_metrics(state, model_id, total_input_tokens, total_output_tokens)
            # logger.debug(f"챕터 {i+1} 총 토큰 사용량: {total_input_tokens} input, {total_output_tokens} output")

        logger.info(
            f"챕터 메타데이터 생성 완료: 총 {len(state['chapter_metadata'])}개 챕터"
        )

        return state

    except Exception as e:
        error_msg = f"챕터 메타데이터 생성 노드에서 복구 불가능한 오류 발생: {str(e)}"
        logger.error(error_msg)
        raise
    finally:
        pass
