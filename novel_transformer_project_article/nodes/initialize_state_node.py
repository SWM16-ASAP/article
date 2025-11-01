from ..state import BookState
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def initialize_state(state: BookState) -> BookState:
    try:
        logger.info("=== 주제 선정 및 기사 수집 시작 ===")

        # 1. 초기화 (Article은 단일 챕터만 가짐)
        if not state.get("chapter_metadata"):
            from ..state import ChapterMetadata
            initial_metadata = [ChapterMetadata(
                chapterNum=0,  # 항상 0번 챕터
                title=None,
                summary=""
            )]
            state["chapter_metadata"] = initial_metadata
            logger.info("chapter_metadata 초기화 완료")

        # 2. CEFR 레벨별 결과 구조 초기화 (Article은 항상 1개 챕터)
        if not state.get("leveled_results"):
            cefr_levels = ["A0", "A1", "A2", "B1", "B2", "C1", "C2"]
            leveled_results = []
            for level in cefr_levels:
                leveled_results.append({
                    "textLevel": level,
                    "chapters": [{
                        "chapterNum": 0,  # 항상 0번 챕터
                        "chunks": []
                    }]
                })
            state["leveled_results"] = leveled_results
            logger.info("leveled_results 초기화 완료")

        # full_text가 이미 있으면 skip
        if state.get("full_text") and state["full_text"].strip():
            logger.info("full_text가 이미 존재합니다. 주제 선정 단계를 건너뜁니다.")
            return state

    except Exception as e:
        logger.error(f"initialize_state error: {e}")
        raise

    return state