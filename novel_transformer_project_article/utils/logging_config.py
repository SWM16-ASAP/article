"""
Centralized Logging Configuration for Text Leveling System

이 모듈은 프로젝트 전체에서 사용할 표준화된 로깅 설정을 제공합니다.
모든 파일에서 이 모듈을 import하여 일관된 로깅을 사용하세요.

사용법:
    from novel_transformer_project_article.utils.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("메시지")
"""

import logging
import os
from typing import Optional


# 전역 로깅 설정 상태
_logging_configured = False


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    프로젝트 전체의 로깅을 설정합니다.
    
    Args:
        log_level: 로그 레벨 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
                  환경변수 LOG_LEVEL이 있으면 우선 사용
    """
    global _logging_configured
    
    if _logging_configured:
        return
    
    # 환경변수에서 로그 레벨 확인
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # 로그 레벨 검증 log_level이 없으면 INFO로 결정
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    # 루트 로거 설정
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ],
        force=True  # 기존 설정 덮어쓰기
    )
    
    # 외부 라이브러리 로그 레벨 조정 (노이즈 제거)
    noisy_loggers = [
        'langchain_aws.llms.bedrock',
        'boto3',
        'botocore',
        'urllib3',
        's3transfer',
        'langchain',
        'openai',
        'nemoguardrails'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 로거를 반환합니다.
    
    Args:
        name: 로거 이름 (보통 __name__ 사용)
        
    Returns:
        설정된 로거 객체
        
    Example:
        logger = get_logger(__name__)
        logger.info("작업 시작")
        logger.debug("세부 정보")
        logger.error("오류 발생")
    """
    # 로깅이 설정되지 않았다면 자동 설정
    if not _logging_configured:
        setup_logging()
    
    return logging.getLogger(name) 