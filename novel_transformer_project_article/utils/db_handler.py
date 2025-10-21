import boto3
import json
import os
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from ..state import BookState
from .logging_config import get_logger

logger = get_logger(__name__)


def setup_mongo_client():
    """MongoDB 클라이언트를 설정합니다."""
    # aws secret manager에서 들고오는 것으로 여기서 가져오는 uri는 실제 uri가 아닌 secret name 값이다.
    secret_name = os.getenv("MONGO_URI_SECRET_NAME")
    region_name = os.getenv("AWS_REGION")

    if not secret_name:
        logger.error("환경 변수 'MONGO_URI_SECRET_NAME'가 설정되지 않았습니다.")
        return None
    
    if not region_name:
        logger.error("환경 변수 'AWS_REGION'가 설정되지 않았습니다.")
        return None

    # boto3 Session 생성 (수정됨)
    session = boto3.Session()
    client = session.client(
        service_name = "secretsmanager",
        region_name = region_name
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except Exception as e:
        logger.error(f"AWS Secret Manager 접근 실패: {e}")
        return None

    if "SecretString" in get_secret_value_response:
        secret = get_secret_value_response["SecretString"]
    else:
        # SecretBinary는 base64 디코딩이 필요하지만, 여기서는 SecretString을 가정
        logger.error("Secrets Manager 시크릿이 SecretString 형식이 아닙니다.")
        return None
    
    # 이런 형식으로 들어온다. {"MONGO_URI": "mongodb+srv://user:pass@cluster.mongodb.net/"} -> json 파싱
    try:
        secret_dict = json.loads(secret)
        mongo_uri = secret_dict.get("MONGO_URI")
        if not mongo_uri:
            logger.error(f"Secrets Manager 시크릿 '{secret_name}'에 'MONGO_URI' 키가 없습니다.")
            return None
    except json.JSONDecodeError as e:
        # 단순 문자열로 저장된 경우를 처리
        logger.warning(f"JSON 파싱 실패: {e}. 시크릿이 단순 문자열로 저장되어 있을 수 있습니다.")
        # 시크릿 값을 직접 URI로 사용 시도
        mongo_uri = secret.strip()
        if not mongo_uri.startswith("mongodb"):
            logger.error("유효한 MongoDB URI가 아닙니다.")
            return None

    if not mongo_uri:
        logger.error("MongoDB URI가 설정되지 않았습니다.")
        return None

    try:
        mongo_client = MongoClient(mongo_uri)
        # mongoDB 연결 확인
        mongo_client.admin.command("ping")
        logger.info("MongoDB 연결 성공")
        return mongo_client

    except ConnectionFailure as e:
        logger.error(f"MongoDB Atlas연결 실패: {e}")
        return None

    except Exception as e:
        logger.error(f"MongoDB 연결 실패: {e}")
        return None


