"""
Langfuse 프롬프트 마이그레이션 스크립트

기존 Python 파일에 정의된 프롬프트들을 Langfuse로 업로드합니다.
각 레벨(A0-C2)별로 base, rag, feedback 버전의 프롬프트를 생성합니다.
"""

import os
import sys
import importlib
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

LEVELS = ["a0", "a1", "a2", "b1", "b2", "c1", "c2"]

def extract_chat_messages(prompt_template):
    """
    LangChain ChatPromptTemplate에서 Chat 포맷의 메시지 배열 추출

    Returns:
        list: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    messages = []

    for msg in prompt_template.messages:
        # 메시지 타입을 클래스 이름으로 판단
        msg_class_name = msg.__class__.__name__
        content = msg.prompt.template

        if "System" in msg_class_name:
            messages.append({"role": "system", "content": content})
        elif "Human" in msg_class_name:
            messages.append({"role": "user", "content": content})

    return messages

def upload_leveling_prompts():
    """레벨링 프롬프트 업로드 (base, rag, feedback 버전)"""
    print("\n=== Uploading Leveling Prompts ===\n")

    for level in LEVELS:
        print(f"Processing {level.upper()} level...")

        try:
            # 프롬프트 모듈 import
            module_path = f"novel_transformer_project.prompts.leveling.{level}_prompt"
            prompt_module = importlib.import_module(module_path)

            # 1. Base 버전
            print(f"  Uploading {level}-leveling-base...")
            base_prompt = prompt_module.get_prompt(use_rag=False, is_feedback_generated=False)
            messages = extract_chat_messages(base_prompt)

            langfuse.create_prompt(
                name=f"{level}-leveling-base",
                type="chat",
                prompt=messages,
                config={
                    "model": "us.meta.llama4-scout-17b-instruct-v1:0",
                    "temperature": 0.3 if level in ["a0", "a1"] else (0.6 if level in ["a2", "b1", "b2"] else 0.8),
                    "type": "base",
                    "level": level.upper(),
                    "variables": ["text", "cumulative_context", "current_chapter_summary", "previous_chapter_summary"]
                },
                labels=["production"],
                tags=[f"level:{level}", "type:base", "leveling"]
            )
            print(f"  ✅ {level}-leveling-base uploaded")

            # 2. RAG 버전
            print(f"  Uploading {level}-leveling-rag...")
            rag_prompt = prompt_module.get_prompt(use_rag=True, is_feedback_generated=False)
            messages = extract_chat_messages(rag_prompt)

            langfuse.create_prompt(
                name=f"{level}-leveling-rag",
                type="chat",
                prompt=messages,
                config={
                    "model": "us.meta.llama4-scout-17b-instruct-v1:0",
                    "temperature": 0.3 if level in ["a0", "a1"] else (0.6 if level in ["a2", "b1", "b2"] else 0.8),
                    "type": "rag",
                    "level": level.upper(),
                    "variables": ["text", "cumulative_context", "current_chapter_summary", "previous_chapter_summary", "agent_scratchpad"]
                },
                labels=["staging"],
                tags=[f"level:{level}", "type:rag", "leveling"]
            )
            print(f"  ✅ {level}-leveling-rag uploaded")

            # 3. Feedback 버전
            print(f"  Uploading {level}-leveling-feedback...")
            feedback_prompt = prompt_module.get_prompt(use_rag=False, is_feedback_generated=True)
            messages = extract_chat_messages(feedback_prompt)

            langfuse.create_prompt(
                name=f"{level}-leveling-feedback",
                type="chat",
                prompt=messages,
                config={
                    "model": "us.meta.llama4-scout-17b-instruct-v1:0",
                    "temperature": 0.3 if level in ["a0", "a1"] else (0.6 if level in ["a2", "b1", "b2"] else 0.8),
                    "type": "feedback",
                    "level": level.upper(),
                    "variables": ["text", "cumulative_context", "current_chapter_summary", "previous_chapter_summary", "feedback", "leveled_text"]
                },
                labels=["production"],
                tags=[f"level:{level}", "type:feedback", "leveling"]
            )
            print(f"  ✅ {level}-leveling-feedback uploaded")

        except Exception as e:
            print(f"  ❌ Error uploading {level} leveling prompts: {e}")
            continue

    print("\n✅ All leveling prompts uploaded!\n")

def upload_judge_prompts():
    """Judge 프롬프트 업로드"""
    print("\n=== Uploading Judge Prompts ===\n")

    for level in LEVELS:
        print(f"Processing {level.upper()} judge prompt...")

        try:
            # Judge 프롬프트 모듈 import
            module_path = f"novel_transformer_project.prompts.judging.{level}_judge_prompt"
            judge_module = importlib.import_module(module_path)

            judge_prompt = judge_module.get_judge_prompt()
            messages = extract_chat_messages(judge_prompt)

            langfuse.create_prompt(
                name=f"{level}-judge",
                type="chat",
                prompt=messages,
                config={
                    "model": "us.amazon.nova-lite-v1:0",
                    "temperature": 0.4,
                    "type": "judge",
                    "level": level.upper(),
                    "variables": ["original_text", "leveled_text", "cumulative_context", "current_chapter_summary", "previous_chapter_summary", "format_instructions"]
                },
                labels=["production"],
                tags=[f"level:{level}", "type:judge", "evaluation"]
            )
            print(f"  ✅ {level}-judge uploaded")

        except Exception as e:
            print(f"  ❌ Error uploading {level} judge prompt: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n✅ All judge prompts uploaded!\n")

def upload_utility_prompts():
    """유틸리티 프롬프트 업로드 (article, metadata, translation, validation, cover_image)"""
    print("\n=== Uploading Utility Prompts ===\n")

    utility_prompts = [
        {
            "name": "evaluate-text-prompt",
            "hardcoded": True,
            "prompt": [
                {"role": "system", "content": """You are an expert linguistic analyst specializing in the Common European Framework of Reference for Languages (CEFR). Your task is to analyze the provided English text and determine its CEFR level (from A0 to C2).

Return your analysis in the following JSON format:

{format_instructions}

---

Analyze the text holistically, considering:
- Vocabulary complexity and range
- Grammatical structures used
- Sentence complexity
- Text cohesion and coherence
- Overall readability

Provide your assessment based on the CEFR level descriptors:
- A0: Pre-A1 level, very basic understanding
- A1: Beginner
- A2: Elementary
- B1: Intermediate
- B2: Upper Intermediate
- C1: Advanced
- C2: Mastery/Native-like"""},
                {"role": "user", "content": "Please analyze this English text and determine its CEFR level:\n\n{text}"}
            ],
            "config": {
                "model": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                "temperature": 0.7,
                "type": "evaluation",
                "variables": ["text", "format_instructions"]
            },
            "description": "Evaluates text CEFR level"
        },
        {
            "name": "generate-article",
            "module": "novel_transformer_project.prompts.generate_article_prompt",
            "function": "get_article_generation_prompt",
            "config": {
                "model": "us.meta.llama3-2-90b-instruct-v1:0",  # generate_article_node.py:23
                "temperature": 0.4,  # generate_article_node.py:24
                "type": "article_generation",
                "variables": ["topic_content", "format_instructions"]  # generate_article_node.py:44-46
            },
            "description": "Generates viral news article from chapter summary"
        },
        {
            "name": "generate-chapter-metadata",
            "module": "novel_transformer_project.prompts.generate_chapter_metadata_prompt",
            "function": "get_metadata_prompt",
            "function_args": {},  # 이제 인자 없이 호출
            "config": {
                "model": "us.meta.llama4-maverick-17b-instruct-v1:0",  # generate_chapter_metadata.py:42
                "temperature": 0.7,  # default from setup_bedrock
                "type": "metadata_generation",
                "variables": ["chapter_text", "format_instructions"]  # generate_chapter_metadata.py:85-87 (업데이트됨)
            },
            "description": "Generates both title and summary for chapter"
        },
        {
            "name": "translate-to-english",
            "module": "novel_transformer_project.prompts.translate_to_english_prompt",
            "function": "get_translation_prompt",
            "config": {
                "model": "us.meta.llama4-scout-17b-instruct-v1:0",  # validate_custom_content_node.py:25
                "temperature": 0.7,  # default from setup_bedrock
                "max_tokens": 2000,  # validate_custom_content_node.py:105
                "type": "translation",
                "variables": ["text_to_translate", "format_instructions"]  # validate_custom_content_node.py:117
            },
            "description": "Translates text to English"
        },
        {
            "name": "summarize-text",
            "module": "novel_transformer_project.prompts.validate_content_prompt",
            "function": "get_text_summarization_prompt",
            "config": {
                "model": "us.meta.llama4-scout-17b-instruct-v1:0",  # validate_custom_content_node.py:26
                "temperature": 0.7,  # default from setup_bedrock
                "max_tokens": 2000,  # validate_custom_content_node.py:145
                "type": "summarization",
                "variables": ["text_to_summarize", "min_characters", "max_characters", "format_instructions"]  # validate_custom_content_node.py:156
            },
            "description": "Summarizes text to specific length"
        },
        {
            "name": "generate-cover-image-prompt-literature",
            "module": "novel_transformer_project.prompts.generate_cover_image_prompt",
            "function": "get_cover_image_prompt",
            "function_args": {"content_type": "literature"},  # Will create two versions
            "config": {
                "model": "us.deepseek.r1-v1:0",  # generate_cover_image_node.py:34
                "temperature": 0.7,  # default from setup_bedrock
                "type": "image_prompt_generation",
                "content_type": "literature",
                "variables": ["input", "format_instructions"]  # generate_cover_image_node.py:81-84
            },
            "description": "Generates cover image prompt for literature"
        },
        {
            "name": "generate-cover-image-prompt-article",
            "module": "novel_transformer_project.prompts.generate_cover_image_prompt",
            "function": "get_cover_image_prompt",
            "function_args": {"content_type": "article"},
            "config": {
                "model": "us.deepseek.r1-v1:0",  # generate_cover_image_node.py:34
                "temperature": 0.7,  # default from setup_bedrock
                "type": "image_prompt_generation",
                "content_type": "article",
                "variables": ["input", "format_instructions"]  # generate_cover_image_node.py:81-84
            },
            "description": "Generates cover image prompt for article"
        }
    ]

    for prompt_info in utility_prompts:
        prompt_name = prompt_info['name']
        print(f"Processing {prompt_name}...")

        try:
            # Hardcoded 프롬프트인 경우
            if prompt_info.get("hardcoded"):
                messages = prompt_info["prompt"]
            else:
                # 프롬프트 모듈 import
                prompt_module = importlib.import_module(prompt_info["module"])
                get_prompt_func = getattr(prompt_module, prompt_info["function"])

                # 함수 인자가 있으면 전달
                if "function_args" in prompt_info:
                    result = get_prompt_func(**prompt_info["function_args"])
                else:
                    result = get_prompt_func()

                # get_cover_image_prompt는 tuple (prompt, negative_prompt)를 반환
                if isinstance(result, tuple):
                    prompt_template = result[0]
                else:
                    prompt_template = result

                # Chat 메시지 추출
                messages = extract_chat_messages(prompt_template)

            # Langfuse에 업로드
            langfuse.create_prompt(
                name=prompt_name,
                type="chat",
                prompt=messages,
                config=prompt_info["config"],
                labels=["production"],
                tags=["utility", prompt_info["config"]["type"]]
            )
            print(f"  ✅ {prompt_name} uploaded")

        except Exception as e:
            print(f"  ❌ Error uploading {prompt_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n✅ All utility prompts uploaded!\n")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Langfuse Prompt Migration Script (Chat Format)")
    print("=" * 60)

    # API 키 확인
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        print("❌ Error: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env file")
        sys.exit(1)

    print(f"\n✅ Connected to Langfuse: {os.getenv('LANGFUSE_HOST', 'https://us.cloud.langfuse.com')}")

    # 프롬프트 업로드
    upload_leveling_prompts()
    upload_judge_prompts()
    upload_utility_prompts()

    print("=" * 60)
    print("✅ Migration completed successfully!")
    print("=" * 60)
    print("\n📊 Summary:")
    print(f"  - Leveling prompts: {len(LEVELS) * 3} (base + rag + feedback)")
    print(f"  - Judge prompts: {len(LEVELS)}")
    print(f"  - Utility prompts: 8 (evaluate-text, article, metadata, translation, summarization, cover_image x2)")
    print(f"  - Total prompts: {len(LEVELS) * 4 + 8}")
    print(f"\n🔗 View prompts at: {os.getenv('LANGFUSE_HOST', 'https://us.cloud.langfuse.com')}")
    
    # Langfuse flush to ensure all data is sent
    langfuse.flush()

if __name__ == "__main__":
    main()