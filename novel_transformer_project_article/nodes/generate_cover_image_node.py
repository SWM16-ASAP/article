import os
import re
import json
from openai import OpenAI

from ..state import BookState
from ..utils.logging_config import get_logger
from ..utils.workflow_helpers import update_usage_metrics
from ..utils.langfuse_client import (
    get_langfuse_client,
    is_langfuse_enabled,
    get_prompt_label,
)

logger = get_logger(__name__)

def get_cover_image_prompt(article_text: str) -> str:
    """
    Gets the cover image prompt from Langfuse if enabled, otherwise returns hardcoded prompt.

    Args:
        article_text: The article text to include in the prompt

    Returns:
        The formatted image generation prompt
    """
    # Try to get prompt from Langfuse
    if is_langfuse_enabled():
        try:
            client = get_langfuse_client()
            label = get_prompt_label()
            prompt_name = "generate-cover-image-prompt-article"

            langfuse_prompt = client.get_prompt(prompt_name, label=label)
            logger.info(f"Loaded prompt '{prompt_name}' from Langfuse (label: {label}, version: {langfuse_prompt.version})")

            prompt_text = langfuse_prompt.prompt[0]["content"]
            # Langfuse 프롬프트에서 템플릿 가져오기 (compile을 사용하여 변수 삽입)
            compiled_messages = prompt_text.format(article_text=article_text)

            logger.info(f"Compiled prompt from Langfuse: {compiled_messages[:200]}...")
            logger.info(f"Article text length: {len(article_text)}, first 100 chars: {article_text[:100]}")
            return compiled_messages
        except Exception as e:
            logger.warning(f"Failed to load prompt from Langfuse, falling back to hardcoded prompt: {e}")

    # Fallback to hardcoded prompt
    return f"""Make me a article cover image
The article is "{article_text}"

You must keep this condition:
1. The image should be 1:1 ratio
2. The image style should be same with the image I gave you
3. I want the cover image to be easily recognizable at a glance."""

def generate_cover_image(state: BookState) -> BookState:
    """
    Generates a cover image using OpenRouter GPT-5-image-mini model.
    """
    logger.info("=== 커버 이미지 생성 시작 ===")

    # Get OpenRouter API key from environment
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        error_msg = "OPENROUTER_API_KEY 환경 변수가 설정되지 않았습니다."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Initialize OpenRouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    image_model_id = "openai/gpt-5-image-mini"

    article_text = state.get("chapters")[0].strip()

    # 예시 이미지 URL (환경 변수 또는 하드코딩)
    example_image_url = os.getenv("EXAMPLE_IMAGE_URL", "https://i.ibb.co/C5FJSVsd/Gemini-Generated-Image-tl1mcktl1mcktl1m.png")
    logger.info(f"예시 이미지 URL: {example_image_url}")

    # Get image generation prompt (from Langfuse or hardcoded fallback)
    image_prompt = get_cover_image_prompt(article_text)

    logger.info("OpenRouter GPT-5-image-mini로 커버 이미지 생성 중...")

    try:
        # Create image generation request using chat.completions with example image URL
        completion = client.chat.completions.create(
            model=image_model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": image_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": example_image_url
                            }
                        }
                    ]
                }
            ]
        )

        # 이미지는 message.images 필드에 있음
        message = completion.choices[0].message

        # images 필드 확인
        if not hasattr(message, 'images') or not message.images:
            error_msg = "응답에 images 필드가 없거나 비어있습니다."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"생성된 이미지 개수: {len(message.images)}")

        # 첫 번째 이미지 추출 (dict 형식)
        first_image = message.images[0]
        base64_image = first_image['image_url']['url']

        # 토큰 사용량 추적
        input_tokens = completion.usage.prompt_tokens if completion.usage else 0
        output_tokens = completion.usage.completion_tokens if completion.usage else 1
        update_usage_metrics(state, image_model_id, input_tokens, output_tokens)

        # Store the base64 image data
        state["cover_image_url"] = base64_image
        logger.info("=== 커버 이미지 생성 완료 ===")

    except Exception as e:
        logger.error(f"OpenRouter를 통한 이미지 생성 중 오류 발생: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise

    return state
