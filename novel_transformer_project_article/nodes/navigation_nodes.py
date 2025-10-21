from ..state import BookState
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def move_to_next_chunk(state: BookState) -> BookState:
    """다음 청크로 이동하거나 다음 챕터로 이동합니다."""
    logger.debug("=== move_to_next_chunk 시작 ===")
    state["current_chunk_index"] += 1
    return state


def move_to_next_chapter(state: BookState) -> BookState:
    """다음 챕터로 이동합니다."""
    logger.debug("=== move_to_next_chapter 시작 ===")
    state["current_chapter_index"] += 1
    state["current_chunk_index"] = 0
    return state 