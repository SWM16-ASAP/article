"""
Langfuse 프롬프트 업로드 스크립트 - select_topic_with_llm

fetch_topic_and_articles_node.py의 _select_topic_with_llm 함수에서 사용하는
프롬프트를 Langfuse에 업로드합니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langfuse import Langfuse
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Langfuse 클라이언트 초기화
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
)


def upload_select_topic_prompt():
    """주제 선정 프롬프트를 Langfuse에 업로드"""
    print("\n=== Uploading Select Topic Prompt ===\n")

    prompt_name = "select-topic-with-llm"

    # fetch_topic_and_articles_node.py의 _select_topic_with_llm 프롬프트
    messages = [
        {
            "role": "system",
            "content": """
            당신은 오늘의 기사를 선정을 전문적으로 하는 편집장입니다.

            당신의 응답은 아래와 같은 규칙을 지켜야 합니다!!
            - 아래의 선정 기준에 따라 3개의 기사를 선정합니다.
            - 3개의 기사는 선정 기준에 더 일치하는 순서대로 제공합니다.
            - 응답 포맷: {format_instructions}

            당신은 독자들에게 하루에 한번 밖에 글을 제공할 수 밖에 없기에 중대하고 적절한 기사를 선정해야 합니다.
            당신의 기준은 아래와 같습니다.

            선정 기준:
            1. **최근 기사들과 중복되지 않는 주제** : 다음 주제는 제외
            -  최근 올린 뉴스 주제: {recent_article_headlines}

            2. **메이저한 주제**: 하루에 하나 올리기에 충분히 중요한 주제
            - 뉴스에서 주요 뉴스로 다룰 법한 주제
            - 영향력이 국내에 한정되지 않고 국제적으로도 관심을 가질 만한 주제
            - 사회적 영향력이 큰 사건/정책/기술 혁신
            - 장기적 관심사 (단발성 사건 지양)

            3. **적절성**: 다음 주제는 반드시 제외
            - 가십/연예인 사생활/스캔들
            - 정치적 논란/당파적 이슈
            - 개인 폭로/루머성 기사
            - 선정적이거나 자극적인 내용

            4. **교육적 가치**: 독자에게 유익한 정보 제공
            - 새로운 지식/인사이트 제공
            - 트렌드의 배경과 맥락 이해
            - 실생활에 도움이 되는 정보
        """
        },
        {
            "role": "user",
            "content": """ 다음은 오늘의 뉴스 헤드라인입니다:
        {headlines}

        위 헤드라인들을 분석하여 오늘의 가장 중요하고 영향력이 큰 기사 주제를 **3개** 선정하세요.
        선정된 기사들의 제목과 url을 함께 반환해주세요
        가장 메이저하고 중요한 순서대로 정렬해주세요 (1번이 가장 중요).

        **Critical**: 선정된 기사에 대해서는 주제의 제목과 url을 변경하거나 번역하지 말고 그대로 반환하세요!!!

        아래와 같은 양식으로 반환해주세요.
        {format_instructions}
        """
        }
    ]

    config = {
        "model": "us.meta.llama4-scout-17b-instruct-v1:0",  # fetch_topic_and_articles_node.py:182
        "temperature": 0.3,  # fetch_topic_and_articles_node.py:183
        "type": "topic_selection",
        "variables": ["headlines", "recent_article_headlines", "format_instructions"]  # fetch_topic_and_articles_node.py:245
    }

    try:
        print(f"Processing {prompt_name}...")

        langfuse.create_prompt(
            name=prompt_name,
            type="chat",
            prompt=messages,
            config=config,
            labels=["production"],
            tags=["article", "topic_selection", "news"]
        )

        print(f"  ✅ {prompt_name} uploaded successfully")
        print(f"\n📋 Prompt Details:")
        print(f"  - Name: {prompt_name}")
        print(f"  - Model: {config['model']}")
        print(f"  - Temperature: {config['temperature']}")
        print(f"  - Variables: {', '.join(config['variables'])}")
        print(f"  - Type: {config['type']}")

    except Exception as e:
        print(f"  ❌ Error uploading {prompt_name}: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\n✅ Select Topic Prompt uploaded!\n")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Langfuse Select Topic Prompt Upload Script")
    print("=" * 60)

    # API 키 확인
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        print("❌ Error: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env file")
        sys.exit(1)

    print(f"\n✅ Connected to Langfuse: {os.getenv('LANGFUSE_HOST', 'https://us.cloud.langfuse.com')}")

    # 프롬프트 업로드
    upload_select_topic_prompt()

    print("=" * 60)
    print("✅ Upload completed successfully!")
    print("=" * 60)
    print(f"\n🔗 View prompt at: {os.getenv('LANGFUSE_HOST', 'https://us.cloud.langfuse.com')}/prompts/select-topic-with-llm")

    # Langfuse flush to ensure all data is sent
    langfuse.flush()


if __name__ == "__main__":
    main()
