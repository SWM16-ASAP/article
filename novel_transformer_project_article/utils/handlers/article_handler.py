from typing import Dict, Any, Optional
import boto3
import requests
import json
import base64
import os

from ..logging_config import get_logger
from ...state import BookState
from ..workflow_helpers import send_discord_webhook

logger = get_logger(__name__)

class ArticleHandler:
    """
    Article 전용 핸들러.
    Article은 자동 생성되므로 S3에서 입력을 읽지 않고, 출력만 처리합니다.
    """

    def __init__(self, aws_region: str = "us-east-1"):
        """
        핸들러를 초기화합니다.

        Args:
            aws_region: 사용할 AWS 리전.
        """
        self.s3_client = boto3.client('s3', region_name=aws_region)

    def create_output_structure(self, state: BookState) -> Dict[str, Any]:
        """
        Article 출력 구조를 생성합니다.
        """
        # 1. leveled_results 변환
        leveled_results = state.get("leveled_results", [])
        converted_leveled_results = []
        for result in leveled_results:
            converted_chapters = []
            for chapter in result.get("chapters", []):
                converted_chunks = []
                for chunk in chapter.get("chunks", []):
                    chunk_text = chunk.get("chunkText", "")

                    converted_chunks.append({
                        "chunkNum": chunk.get("chunkNum", 0) + 1,
                        "isImage": chunk.get("isImage", False),
                        "chunkText": chunk_text,
                        "description": chunk.get("description", "")
                    })
                converted_chapters.append({
                    "chapterNum": chapter.get("chapterNum", 0) + 1,
                    "chunks": converted_chunks
                })
            converted_leveled_results.append({
                "textLevel": result.get("textLevel", ""),
                "chapters": converted_chapters
            })

        # 2. language 설정
        language_code = state.get("target_language_code")
        # None이면 그대로 None, 값이 있으면 배열로 감싸기
        target_language_code = None if language_code is None else [language_code]

        # 2. 출력 데이터 구조 생성
        output_data = {
            "id": state.get("id"),
            "content_type": state.get("content_type"),
            "title": state.get("title"),
            "author": state.get("author"),
            "target_language_code": target_language_code,
            "cover_image_url": state.get("cover_image_url"),
            "tags": state.get("tags"),
            "original_text_level": state.get("original_text_level"),
            "leveled_results": converted_leveled_results
        }

        return output_data

    def _save_json_to_s3(self, data: dict, bucket: str, key: str):
        """JSON 데이터를 S3에 저장합니다."""
        try:
            json_content = json.dumps(data, ensure_ascii=False, indent=2)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json_content.encode('utf-8'),
                ContentType='application/json'
            )
            logger.info(f"Successfully saved JSON to s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"Failed to save JSON to S3: {e}", exc_info=True)
            raise

    def _save_cover_image_to_s3(self, final_state: BookState, bucket: str, output_folder_key: str, content_type: str) -> Optional[str]:
        """커버 이미지를 S3에 저장합니다."""
        base64_image_data = final_state.get("cover_image_url")
        if not base64_image_data:
            error_msg = "No cover image data found in the state."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Data URL 형식(data:image/jpeg;base64,)인 경우 prefix 제거
            if base64_image_data.startswith('data:'):
                # 쉼표 이후의 실제 base64 데이터만 추출
                base64_image_data = base64_image_data.split(',', 1)[1]

            # 공백 및 줄바꿈 제거
            base64_image_data = base64_image_data.strip().replace('\n', '').replace('\r', '')

            image_bytes = base64.b64decode(base64_image_data)
            image_key = f"{content_type}/{output_folder_key}/images/cover.jpg"

            # S3에 업로드
            self.s3_client.put_object(
                Bucket=bucket,
                Key=image_key,
                Body=image_bytes,
                ContentType='image/jpeg'
            )
            s3_url = f"s3://{bucket}/{image_key}"
            logger.info(f"Successfully saved cover image to {s3_url}")
            return s3_url
        except (base64.binascii.Error, TypeError) as e:
            logger.error(f"Invalid base64 data: {e}", exc_info=True)
            raise ValueError(f"Invalid base64 data: {e}") from e
        except Exception as e:
            logger.error(f"Failed to save cover image to S3: {e}", exc_info=True)
            raise

    def save_output(self, final_state: BookState, output_bucket: str) -> str:
        """
        Article 생성 결과를 S3에 저장합니다.

        Args:
            final_state: 워크플로우 최종 상태
            output_bucket: 출력할 S3 버킷

        Returns:
            str: 백엔드 DB에서 생성된 Article ID
        """
        logger.info("Starting Article output process")

        # 1. JSON 출력 구조 생성
        output_data = self.create_output_structure(final_state)
        output_folder_key = final_state["id"]
        content_type = final_state["content_type"]
        output_json_key = f"{content_type}/{output_folder_key}/metadata.json"

        # 2. 커버 이미지 S3 저장
        s3_image_url = self._save_cover_image_to_s3(final_state, output_bucket, output_folder_key, content_type)
        output_data["cover_image_url"] = s3_image_url

        # 3. JSON S3 저장
        self._save_json_to_s3(output_data, output_bucket, output_json_key)

        backend_url = os.getenv("ADD_NEW_ARTICLE_URL")
        api_key = os.getenv("SWAPPER_API_KEY")

        if not backend_url:
            error_msg = "ADD_NEW_ARTICLE_URL 환경 변수가 설정되지 않았습니다."
            logger.error(error_msg)
            raise ValueError(error_msg)

        #headers
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": api_key
        }

        # Request body
        body = {
            "id": output_folder_key
        }

        try:
            response = requests.post(
                backend_url,
                headers=headers,
                json=body,  # body가 아니라 json 파라미터 사용
                timeout=15
            )

            response.raise_for_status()

            # DB ID 추출
            db_id = response.text

            # 성공 메시지
            tags = final_state.get("tags", [])
            category = tags[0] if tags else "Unknown"
            language = final_state.get("target_language_code", "Unknown")
            title = final_state.get("title", "Unknown")

            discord_message = f"""✅ **기사 생성 완료** ✅

            **ID**: `{output_folder_key}`
            **DB id**: {db_id}
            **제목**: {title}
            **카테고리**: {category}
            **언어**: {language}

            **S3 저장 경로**: `s3://{output_bucket}/{content_type}/{output_folder_key}/`
            **백엔드 알림**: 성공

            기사가 성공적으로 생성되어 백엔드에 등록되었습니다."""

            send_discord_webhook(discord_message)
            logger.info("Article output process completed successfully")

            return db_id
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to notify backend: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Article output process failed in save_output: {e}", exc_info=True)
            raise