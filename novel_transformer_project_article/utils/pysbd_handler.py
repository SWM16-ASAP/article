"""
pySBD 세그멘터 싱글톤 핸들러

전체 프로젝트에서 pySBD 세그멘터를 한 번만 로드하고 재사용합니다.
SpaCy보다 더 정확한 문장 분리를 제공합니다.
"""
from typing import Any
from .logging_config import get_logger

logger = get_logger(__name__)

# 전역 변수: pySBD 세그멘터 인스턴스
_segmenter_instance: Any = None


def get_pysbd_segmenter(language: str = "en") -> Any:
    """
    pySBD 세그멘터를 싱글톤으로 로드

    Args:
        language: 세그멘터 언어 코드 (기본값: "en")

    Returns:
        pySBD Segmenter 인스턴스
    """
    global _segmenter_instance

    if _segmenter_instance is None:
        import pysbd
        logger.info(f"pySBD 세그멘터 (언어: {language}) 로딩 중...")
        _segmenter_instance = pysbd.Segmenter(language=language, clean=False)
        logger.info(f"pySBD 세그멘터 로드 완료")

    return _segmenter_instance


def reset_pysbd_segmenter() -> None:
    """
    pySBD 세그멘터 인스턴스를 초기화 (테스트용)
    """
    global _segmenter_instance
    _segmenter_instance = None
    logger.info("pySBD 세그멘터 인스턴스 초기화됨")