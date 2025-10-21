from typing import List
import re
import json

from ..state import BookState
from ..utils.logging_config import get_logger
from ..utils.workflow_helpers import is_custom_content, calculate_custom_progress, send_progress_update

logger = get_logger(__name__)

def split_by_paragraph(text: str) -> List[str]:
    """문단별로 분할"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

def adjust_chunk_size(paragraphs: List[str], min_length: int) -> List[str]:
    """청크 크기를 조정"""
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # 문단 사이에 구분자 추가
        if current_chunk:
            potential_chunk = current_chunk + "\n\n" + paragraph
        else:
            potential_chunk = paragraph

        # 청크 최소 길이 체크
        if len(potential_chunk) >= min_length:
            chunks.append(potential_chunk)
            current_chunk = ""
        else:
            current_chunk = potential_chunk

    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def split_chapter_into_chunks(state: BookState) -> BookState:
    """챕터를 문단별로 분할하여 적절한 크기의 청크로 생성"""
    logger.info(f"=== 챕터 {state['current_chapter_index'] + 1} 청킹 시작 (컨텐츠 기반) ===")
    
    try:
        current_chapter = state["chapters"][state["current_chapter_index"]]

        # Article의 경우 커버 이미지 태그를 앞에 추가
        if state.get("content_type") == "article":
            # 커버 이미지 태그를 위한 JSON 데이터 생성
            cover_image_data = {
                "file": "cover.jpg",
                "description": f"{state.get('title', 'Article')}",
                "alt_text": "Article cover image"
            }
            cover_image_tag = f'[illustration: {json.dumps(cover_image_data, ensure_ascii=False)}]'
            current_chapter = f"{cover_image_tag}\n\n{current_chapter}"

        # 이미지 태그 기준으로 청킹 (문단 단위 분할 제거)
        # Preserve 태그와 illustration 태그를 기준으로 분리
        special_tag_pattern = re.compile(r'(\[illustration:.*?\]|<Preserve>.*?</Preserve>)', re.DOTALL)
        parts = special_tag_pattern.split(current_chapter)
        final_chunks : List[str] = []

        for part in parts:
            if not part or part.isspace():
                continue

            if part.strip().startswith("[illustration:"):
                # 이미지 태그는 그대로 하나의 청크로
                final_chunks.append(part.strip())
            elif part.strip().startswith("<Preserve>"):
                # Preserve 태그 내용을 추출하여 하나의 청크로 처리
                preserve_content = re.sub(r'<Preserve>(.*?)</Preserve>', r'\1', part.strip(), flags=re.DOTALL)
                final_chunks.append(f"[preserve:{preserve_content}]")
            else:
                # 일반 텍스트는 문단 구분(\n\n)을 유지하되 하나의 청크로
                # 빈 문자열이 아닌 경우에만 추가
                text_chunk = part.strip()
                if text_chunk:
                    final_chunks.append(text_chunk)
        
        state["chapter_chunks"].append(final_chunks)

        logger.info(f"{state['current_chapter_index']}번째 챕터 청크 분할 완료: 총 {len(final_chunks)}개 청크")

        # custom 콘텐츠일 때만 진행률 전송 (해당 챕터 분할 직후)
        try:
            if is_custom_content(state):
                progress = calculate_custom_progress(state, "after_split")
                send_progress_update(state, progress)
        except Exception:
            pass

        return state
    
    except Exception as e:
        error_msg = f"챕터 청킹 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        raise 