"""
Langfuse 클라이언트 유틸리티

프롬프트 버전 관리를 위한 Langfuse 클라이언트를 제공합니다.
"""

import os
from functools import lru_cache
from typing import Optional
from langfuse import Langfuse
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from .logging_config import get_logger

logger = get_logger(__name__)

# 로컬 개발 환경인 경우에만 .env 파일 로드
# ENVIRONMENT가 설정되지 않은 경우(=로컬)만 .env 파일 로드
# dev, production 등 Pulumi 스택 배포 환경에서는 환경변수 직접 사용
environment = os.getenv("ENVIRONMENT")
if environment is None:
    load_dotenv()
    logger.info("Local environment detected (no ENVIRONMENT set): loading .env file")
else:
    logger.info(f"Deployment environment detected (ENVIRONMENT={environment}): using environment variables")

# 싱글톤 인스턴스
_langfuse_client: Optional[Langfuse] = None


def get_langfuse_client() -> Optional[Langfuse]:
    """
    Langfuse 클라이언트를 싱글톤으로 반환합니다.
    
    배포 환경(ECS)에서는 환경변수를 직접 사용하고,
    개발 환경에서는 .env 파일에서 로드합니다.

    Returns:
        Langfuse: Langfuse 클라이언트 인스턴스 (실패 시 None)
    """
    global _langfuse_client

    if _langfuse_client is None:
        # 환경 변수에서 API 키 로드
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

        if not public_key or not secret_key:
            logger.warning("Langfuse credentials not found. Prompt versioning will be disabled.")
            return None

        try:
            _langfuse_client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            logger.info(f"Langfuse client initialized successfully: {host}")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            return None

    return _langfuse_client


def get_prompt_label() -> str:
    """
    환경에 따라 적절한 프롬프트 레이블을 반환합니다.

    Returns:
        str: "production", "staging", 또는 "development"
    """
    env = os.getenv("ENVIRONMENT", "development")

    if env == "production":
        return "production"
    elif env == "staging":
        return "staging"
    else:
        # 개발 환경에서도 production 레이블 사용 (안정적인 버전)
        return "production"


def is_langfuse_enabled() -> bool:
    """
    Langfuse 프롬프트 버저닝이 활성화되어 있는지 확인합니다.

    Returns:
        bool: Langfuse 사용 가능 여부
    """
    use_langfuse = os.getenv("USE_LANGFUSE", "true").lower() == "true"
    client = get_langfuse_client()
    return use_langfuse and client is not None


def convert_langfuse_to_langchain(langfuse_prompt) -> ChatPromptTemplate:
    """
    Langfuse의 chat 포맷 프롬프트를 LangChain ChatPromptTemplate로 변환합니다.

    Args:
        langfuse_prompt: Langfuse에서 가져온 프롬프트 객체
            prompt.prompt = [
                {"type": "message", "role": "system", "content": "..."},
                {"type": "message", "role": "user", "content": "..."}
            ]

    Returns:
        ChatPromptTemplate: LangChain에서 사용 가능한 프롬프트 템플릿
    """
    # Langfuse 프롬프트의 chat 메시지 배열 추출
    messages = langfuse_prompt.prompt

    # LangChain 메시지 튜플 형식으로 변환
    langchain_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        # role을 LangChain 형식으로 매핑
        if role == "system":
            langchain_messages.append(("system", content))
        elif role == "user":
            langchain_messages.append(("human", content))
        elif role == "assistant":
            langchain_messages.append(("assistant", content))
        else:
            logger.warning(f"Unknown message role '{role}', treating as 'human'")
            langchain_messages.append(("human", content))

    # ChatPromptTemplate 생성
    return ChatPromptTemplate.from_messages(langchain_messages)


def get_model_config_from_prompt(langfuse_prompt) -> dict:
    """
    Langfuse 프롬프트의 config에서 모델 설정을 추출합니다.

    Args:
        langfuse_prompt: Langfuse에서 가져온 프롬프트 객체

    Returns:
        dict: {
            "model": str,
            "temperature": float,
            "max_tokens": int (optional)
        }
    """
    config = langfuse_prompt.config or {}

    return {
        "model": config.get("model"),
        "temperature": config.get("temperature", 0.7),
        "max_tokens": config.get("max_tokens")
    }