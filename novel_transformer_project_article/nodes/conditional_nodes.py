from ..state import BookState
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def should_continue_chunk(state: BookState) -> str:
    """현재 챕터에서 더 처리할 청크가 있는지 확인합니다."""
    logger.debug("=== should_continue_chunk 시작 ===")
    if state["current_chunk_index"] < len(state["chapter_chunks"][state["current_chapter_index"]]):
        return "continue_chunk"
    else:
        return "next_chapter"


def should_continue_chapter(state: BookState) -> str:
    """더 처리할 챕터가 있는지 확인합니다."""
    logger.debug("=== should_continue_chapter 시작 ===")
    if state["current_chapter_index"] < len(state["chapters"]):
        return "continue_chapter"
    else:
        return "finish" 

def decide_content_path(state: BookState) -> str:
    """콘텐츠 경로를 결정합니다."""
    logger.debug("=== decide_content_path 시작 ===")
    if state["content_type"] == "literature":
        return "literature"
    elif state["content_type"] == "article":
        return "article"
    elif state["content_type"] == "custom":
        return "custom"
    else:
        raise ValueError(f"Invalid content_type: {state['content_type']}")