import boto3
import os
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from .state import BookState, ChapterMetadata
from .nodes.navigation_nodes import move_to_next_chunk, move_to_next_chapter
from .nodes.parallel_cefr_processing_node import create_all_cefr_versions_parallel
from .nodes.generate_article_node import generate_article
from .nodes.fetch_topic_and_articles_node import fetch_topic_and_articles
from .nodes.conditional_nodes import should_continue_chunk, should_continue_chapter
from .nodes.final_summary_log_node import log_final_summary
from .nodes.generate_cover_image_node import generate_cover_image
from .nodes.split_nodes import split_chapter_into_chunks
from .utils.handlers.article_handler import ArticleHandler
from .utils.logging_config import get_logger
# .env 파일 로드
load_dotenv()

# 로거 초기화
logger = get_logger(__name__)

def create_article_transformer_workflow():
    """아티클 변환 워크플로우를 생성합니다."""

    workflow = StateGraph(BookState)

    workflow.add_node("fetch_topic_and_articles", fetch_topic_and_articles)
    workflow.add_node("generate_article", generate_article)
    workflow.add_node("generate_cover_image", generate_cover_image)
    workflow.add_node("split_chapter_into_chunks", split_chapter_into_chunks)
    workflow.add_node("create_all_cefr_versions_parallel", create_all_cefr_versions_parallel)
    workflow.add_node("move_to_next_chunk", move_to_next_chunk)
    workflow.add_node("move_to_next_chapter", move_to_next_chapter)
    workflow.add_node("log_final_summary", log_final_summary)

    # workflow
    workflow.set_entry_point("fetch_topic_and_articles")
    workflow.add_edge("fetch_topic_and_articles", "generate_article")
    workflow.add_edge("generate_article", "split_chapter_into_chunks")
    workflow.add_edge("split_chapter_into_chunks", "create_all_cefr_versions_parallel")
    workflow.add_edge("create_all_cefr_versions_parallel", "move_to_next_chunk")

    workflow.add_conditional_edges(
        "move_to_next_chunk",
        should_continue_chunk,
        {
            "continue_chunk": "create_all_cefr_versions_parallel",
            "next_chapter": "move_to_next_chapter"
        }
    )
    
    workflow.add_conditional_edges(
        "move_to_next_chapter",
        should_continue_chapter,
        {
            "continue_chapter": "split_chapter_into_chunks",
            "finish": "log_final_summary"
        }
    )
    
    workflow.add_edge("log_final_summary", END)
    
    return workflow.compile()


def transform_article() -> Dict[str, Any]:
    """Article을 자동 생성하고 다양한 난이도로 변환하여 S3에 결과를 저장합니다."""
    logger.info("Article 자동 생성 및 레벨링 시작")

    output_s3_bucket = os.getenv('OUTPUT_S3_BUCKET')
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    category = os.getenv('ARTICLE_CATEGORY')  # Sports, Science, Business, Culture, Technology
    language = os.getenv('ARTICLE_LANGUAGE', 'ko')  # ko, ja, common

    if not output_s3_bucket:
        raise ValueError("OUTPUT_S3_BUCKET 환경변수가 필요합니다.")

    if not category:
        raise ValueError("ARTICLE_CATEGORY 환경변수가 필요합니다. (Sports, Science, Business, Culture, Technology)")

    # 카테고리 유효성 검사
    valid_categories = ["Sports", "Science", "Business", "Culture", "Technology"]
    if category not in valid_categories:
        raise ValueError(f"유효하지 않은 카테고리: {category}. 가능한 값: {', '.join(valid_categories)}")

    # 언어 유효성 검사
    valid_languages = ["ko", "ja", "common"]
    if language not in valid_languages:
        raise ValueError(f"유효하지 않은 언어: {language}. 가능한 값: {', '.join(valid_languages)}")

    logger.info(f"Output S3 Bucket: {output_s3_bucket}")
    logger.info(f"AWS Region: {aws_region}")
    logger.info(f"Category: {category}")
    logger.info(f"Language: {language}")

    # 1. 초기 상태 생성 (Article은 자동 생성되므로 full_text는 비어있음)
    # Article은 단일 챕터만 가짐
    initial_metadata = [ChapterMetadata(
        chapterNum=0,  # 항상 0번 챕터
        title=None,
        summary=""
    )]

    # CEFR 레벨별 결과 구조 초기화 (Article은 항상 1개 챕터)
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

    # ID 생성: article-20250120-sports-ko
    from datetime import datetime
    article_id = f"article-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{category.lower()}-{language}"

    initial_state: BookState = {
        "full_text": "",
        "id": article_id,
        "content_type": "article",
        "title": "",
        "author": "",
        "tags": [category],  # 카테고리를 태그로 설정
        "target_language_code": language,  # 언어 정보 추가
        "original_text_level": "",
        "chapters": [],
        "chapter_chunks": [],
        "current_chapter_index": 0,
        "current_chunk_index": 0,
        "chapter_metadata": initial_metadata,
        "leveled_results": leveled_results,
        "origin_url": "",
        "usage_metrics": {},
        "cover_image_url": None,
        "total_cost": 0.0
    }

    try:
        # 2. 워크플로우 실행
        workflow = create_article_transformer_workflow()
        logger.info("=== Article 생성 및 레벨링 워크플로우 시작 ===")
        final_state = workflow.invoke(initial_state, config={"recursion_limit": 10000})
        logger.info("=== Article 생성 및 레벨링 워크플로우 완료 ===")

        # 3. ArticleHandler로 JSON 출력 처리 및 이미지 저장
        handler = ArticleHandler(aws_region=aws_region)
        handler.save_output(final_state, output_bucket=output_s3_bucket)

    except Exception as e:
        error_msg = f"워크플로우 실행 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        raise

    # 최종 상태의 일부를 반환하여 확인 용도로 사용
    return {
        "id": final_state["id"],
        "content_type": final_state["content_type"],
        "original_level": final_state["original_text_level"],
        "results_summary": f"{len(final_state['leveled_results'])} levels generated."
    }

if __name__ == "__main__":
    output_s3_bucket = os.getenv('OUTPUT_S3_BUCKET')
    category = os.getenv('ARTICLE_CATEGORY')

    if output_s3_bucket and category:
        logger.info("=== Article 자동 생성 모드로 실행 ===")
        logger.info(f"OUTPUT_S3_BUCKET: {output_s3_bucket}")
        logger.info(f"ARTICLE_CATEGORY: {category}")
        logger.info(f"ARTICLE_LANGUAGE: {os.getenv('ARTICLE_LANGUAGE', 'ko')}")

        logger.info("Article 생성 및 레벨링을 시작합니다... 이 과정은 시간이 걸릴 수 있습니다.")

        try:
            result = transform_article()

            if result:
                logger.info("Article 생성 및 레벨링이 완료되었습니다!")
                logger.info(f"결과 요약: {result}")
            else:
                logger.error("Article 생성 중 오류가 발생했습니다.")

        except Exception as e:
            logger.error(f"예상치 못한 오류가 발생했습니다: {e}")

    else:
        logger.error("=== 환경변수 누락 ===")
        logger.error("다음 환경변수가 필요합니다:")
        logger.error("- OUTPUT_S3_BUCKET: 생성된 Article을 저장할 S3 버킷 (필수)")
        logger.error("- ARTICLE_CATEGORY: 기사 카테고리 (필수)")
        logger.error("  가능한 값: Sports, Science, Business, Culture, Technology")
        logger.error("- ARTICLE_LANGUAGE: 기사 언어 (선택, 기본값: ko)")
        logger.error("  가능한 값: ko, ja")
        logger.error("- AWS_REGION: AWS 리전 (선택, 기본값: us-east-1)")