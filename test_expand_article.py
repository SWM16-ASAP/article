"""
모델별 기사 확장(expand_article) 성능 테스트

이 스크립트는 다양한 LLM 모델을 사용하여 짧은 기사를 확장하는 성능을 테스트합니다.
각 모델의 확장 결과, 토큰 사용량, 비용, 그리고 최종 길이를 비교합니다.
"""

import json
import os
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException

from novel_transformer_project_article.state import BookState
from novel_transformer_project_article.utils.logging_config import get_logger
from novel_transformer_project_article.utils.workflow_helpers import (
    setup_bedrock,
    BedrockTokenTrackingWrapper,
    clean_json_markdown
)
from novel_transformer_project_article.prompts.expand_article_prompt import get_article_expansion_prompt

logger = get_logger(__name__)

# 기사 생성 결과를 위한 Pydantic 모델 (generate_article_node.py와 동일)
class ArticleGeneration(BaseModel):
    title: str = Field(max_length=70, description="headline for the article")
    content: str = Field(max_length=1500, description="The main content of the article, under 1500 characters")


# 테스트할 모델 목록
TEST_MODELS = [
    {
        "model_id": "us.meta.llama4-maverick-17b-instruct-v1:0",
        "temperature": 0.4,
        "description": "Llama 4 Maverick 17B (기본 모델)"
    },
    {
        "model_id": "us.meta.llama4-scout-17b-instruct-v1:0",
        "temperature": 0.4,
        "description": "Llama 4 Scout 17B"
    },
    {
        "model_id": "us.meta.llama3-2-90b-instruct-v1:0",
        "temperature": 0.4,
        "description": "Llama 3.2 90B"
    },
    {
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "temperature": 0.4,
        "description": "Claude 3 Haiku"
    },
    {
        "model_id": "us.amazon.nova-lite-v1:0",
        "temperature": 0.4,
        "description": "Amazon Nova Lite"
    },
]

# 목(Mock) 테스트 데이터
MOCK_TEST_CASES = [
    {
        "title": "Black Hole Merger Confirms Einstein's Predictions",
        "short_content": "A recent observation of a black hole merger has provided evidence of how black holes work. The detection was made by LIGO and revealed insights into space-time properties. The merger formed a black hole with the mass of 63 suns.",
        "topic_content": """Black Hole Merger Confirms Einstein and Hawking's Predictions

A recent observation of a black hole merger has provided the clearest evidence yet of how black holes work, confirming decades-old predictions by Albert Einstein, Stephen Hawking, and Roy Kerr. The detection, made by the Laser Interferometer Gravitational-Wave Observatory (LIGO), revealed insights into the properties of black holes and the fundamental nature of space-time.

The findings, published in Physical Review Letters, demonstrate that astrophysical black holes are the black holes predicted from Einstein's theory of general relativity. The merger formed a black hole with the mass of 63 suns and spinning at 100 revolutions per second, with the newly merged black hole being larger than the sum of its parts.

This confirms Hawking's theorem, which states that the size of a black hole's event horizon can only ever grow. The observation also hints at connections to the second law of thermodynamics, which could lead to advances in quantum gravity. The detection is a significant breakthrough for science, providing a new understanding of black holes and their behavior.

With future detectors expected to become 10 times more sensitive, scientists are hopeful that they will continue to learn more about the nature of these objects.""",
        "target_length": 1000,
        "max_length": 1500
    },
    {
        "title": "New Climate Study Reveals Alarming Trends",
        "short_content": "Scientists have discovered new patterns in global temperature data. The study shows faster warming than previously thought. Immediate action is needed to address climate change.",
        "topic_content": """Climate Change Accelerating Faster Than Expected

A comprehensive new study published in Nature Climate Change has revealed alarming trends in global warming patterns. Scientists from leading research institutions analyzed decades of temperature data and found that the planet is warming at a rate 30% faster than previous models predicted.

The study examined temperature records from over 10,000 monitoring stations worldwide, satellite data, and ocean temperature measurements. Researchers found that Arctic regions are warming at nearly twice the global average rate, leading to accelerated ice sheet melting and rising sea levels.

Dr. Sarah Martinez, lead author of the study, stated that the findings highlight the urgent need for immediate climate action. The research team also identified feedback loops in the climate system that could accelerate warming further if left unchecked.

The study recommends rapid reduction in greenhouse gas emissions and increased investment in renewable energy technologies to mitigate the worst impacts of climate change.""",
        "target_length": 1000,
        "max_length": 1500
    },
    {
        "title": "AI Breakthrough in Medical Diagnosis",
        "short_content": "Researchers developed an AI system that can detect diseases early. The system uses machine learning to analyze medical images. It shows promising results in clinical trials.",
        "topic_content": """Revolutionary AI System Transforms Early Disease Detection

Scientists at Stanford University have developed a groundbreaking artificial intelligence system that can detect multiple diseases at their earliest stages with unprecedented accuracy. The AI, named MediScan, uses advanced deep learning algorithms to analyze medical imaging data including X-rays, CT scans, and MRI images.

In clinical trials involving over 50,000 patients, MediScan demonstrated a 95% accuracy rate in detecting early-stage cancers, cardiovascular diseases, and neurological conditions. This represents a significant improvement over traditional diagnostic methods, which often miss subtle early warning signs.

The system was trained on millions of anonymized medical images and continuously learns from new cases. Dr. James Chen, the project lead, explained that the AI can identify patterns that are imperceptible to the human eye, potentially saving countless lives through earlier intervention.

Several major hospitals have already begun integrating MediScan into their diagnostic workflows, with plans for wider deployment next year. The technology could revolutionize preventive medicine and significantly reduce healthcare costs by catching diseases before they progress to advanced stages.""",
        "target_length": 1000,
        "max_length": 1500
    }
]


def create_mock_state(test_case: Dict) -> BookState:
    """테스트용 목(Mock) BookState를 생성합니다."""
    return {
        "id": "test-001",
        "content_type": "Article",
        "title": test_case["title"],
        "author": None,
        "tags": ["Science"],
        "target_language_code": "EN",
        "original_text_level": "C1",
        "chapters": [test_case["topic_content"]],
        "chapter_chunks": [],
        "current_chapter_index": 0,
        "current_chunk_index": 0,
        "chapter_metadata": [],
        "leveled_results": [],
        "origin_url": "https://example.com/test",
        "usage_metrics": {},
        "cover_image_url": None,
        "full_text": test_case["topic_content"],
        "total_cost": 0.0
    }


def expand_short_article_with_model(
    model_config: Dict,
    short_content: str,
    title: str,
    topic_content: str,
    target_length: int,
    max_length: int
) -> Dict:
    """
    특정 모델을 사용하여 짧은 기사를 확장하고 결과를 반환합니다.

    Returns:
        Dict: {
            "success": bool,
            "expanded_content": str,
            "original_length": int,
            "expanded_length": int,
            "input_tokens": int,
            "output_tokens": int,
            "cost": float,
            "error": str (실패 시)
        }
    """
    state = create_mock_state({
        "title": title,
        "short_content": short_content,
        "topic_content": topic_content,
        "target_length": target_length,
        "max_length": max_length
    })

    try:
        llm = setup_bedrock(config={
            "model": model_config["model_id"],
            "temperature": model_config["temperature"]
        })

        llm = BedrockTokenTrackingWrapper(llm, state)

        base_parser = PydanticOutputParser(pydantic_object=ArticleGeneration)
        fixing_parser = OutputFixingParser.from_llm(
            parser=base_parser,
            llm=llm,
            max_retries=1
        )

        prompt = get_article_expansion_prompt()
        chain = prompt | llm | clean_json_markdown

        response_text = chain.invoke({
            "current_length": len(short_content),
            "target_length": target_length,
            "max_length": max_length,
            "title": title,
            "topic_content": topic_content,
            "short_content": short_content,
            "format_instructions": base_parser.get_format_instructions()
        })

        # 파싱 시도
        try:
            response = base_parser.parse(response_text)
            parser_used = "base_parser"
        except OutputParserException as parse_error:
            logger.info(f"기본 파서 실패, OutputFixingParser 사용: {str(parse_error)[:50]}...")
            response = fixing_parser.parse(response_text)
            parser_used = "fixing_parser"

        expanded_content = response.content

        # 토큰 사용량 가져오기
        input_tokens = llm.last_input_tokens
        output_tokens = llm.last_output_tokens
        cost = state["total_cost"]

        return {
            "success": True,
            "expanded_content": expanded_content,
            "original_length": len(short_content),
            "expanded_length": len(expanded_content),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "parser_used": parser_used,
            "error": None
        }

    except Exception as e:
        logger.error(f"기사 확장 실패 ({model_config['model_id']}): {e}")
        return {
            "success": False,
            "expanded_content": None,
            "original_length": len(short_content),
            "expanded_length": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "parser_used": None,
            "error": str(e)
        }


def run_model_comparison_test():
    """모든 모델에 대해 확장 테스트를 실행하고 결과를 비교합니다."""
    print("=" * 80)
    print("기사 확장(expand_article) 모델 성능 비교 테스트")
    print("=" * 80)
    print()

    all_results = []

    for test_idx, test_case in enumerate(MOCK_TEST_CASES, 1):
        print(f"\n{'=' * 80}")
        print(f"테스트 케이스 {test_idx}/{len(MOCK_TEST_CASES)}: {test_case['title']}")
        print(f"원본 길이: {len(test_case['short_content'])}자")
        print(f"목표 길이: {test_case['target_length']}자")
        print(f"{'=' * 80}\n")

        test_results = {
            "test_case": test_case['title'],
            "original_length": len(test_case['short_content']),
            "target_length": test_case['target_length'],
            "model_results": []
        }

        for model in TEST_MODELS:
            print(f"\n테스트 중: {model['description']} ({model['model_id']})")
            print("-" * 80)

            result = expand_short_article_with_model(
                model_config=model,
                short_content=test_case['short_content'],
                title=test_case['title'],
                topic_content=test_case['topic_content'],
                target_length=test_case['target_length'],
                max_length=test_case['max_length']
            )

            if result['success']:
                length_diff = result['expanded_length'] - test_case['target_length']
                length_diff_pct = (length_diff / test_case['target_length']) * 100

                print(f"✅ 성공")
                print(f"   확장 결과: {result['original_length']}자 → {result['expanded_length']}자")
                print(f"   목표 대비: {length_diff:+d}자 ({length_diff_pct:+.1f}%)")
                print(f"   사용 파서: {result['parser_used']}")
                print(f"   토큰 사용: 입력 {result['input_tokens']}, 출력 {result['output_tokens']}")
                print(f"   비용: ${result['cost']:.6f}")
                print(f"   내용 미리보기: {result['expanded_content'][:100]}...")
            else:
                print(f"❌ 실패: {result['error']}")

            test_results['model_results'].append({
                "model_id": model['model_id'],
                "description": model['description'],
                **result
            })

        all_results.append(test_results)

    # 최종 요약 출력
    print("\n" + "=" * 80)
    print("전체 테스트 결과 요약")
    print("=" * 80)

    for test_result in all_results:
        print(f"\n테스트 케이스: {test_result['test_case']}")
        print(f"원본 길이: {test_result['original_length']}자, 목표 길이: {test_result['target_length']}자")
        print("-" * 80)
        print(f"{'모델':<50} {'결과':<10} {'확장 길이':<12} {'목표 대비':<15} {'비용':<12}")
        print("-" * 80)

        for model_result in test_result['model_results']:
            if model_result['success']:
                length_diff = model_result['expanded_length'] - test_result['target_length']
                length_diff_pct = (length_diff / test_result['target_length']) * 100
                status = "✅ 성공"
                length_str = f"{model_result['expanded_length']}자"
                diff_str = f"{length_diff:+d}자 ({length_diff_pct:+.1f}%)"
                cost_str = f"${model_result['cost']:.6f}"
            else:
                status = "❌ 실패"
                length_str = "N/A"
                diff_str = "N/A"
                cost_str = "N/A"

            model_name = model_result['description'][:48]
            print(f"{model_name:<50} {status:<10} {length_str:<12} {diff_str:<15} {cost_str:<12}")

    # JSON 파일로 상세 결과 저장
    output_file = "test_expand_article_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n상세 결과가 {output_file}에 저장되었습니다.")
    print("=" * 80)


if __name__ == "__main__":

    run_model_comparison_test()
