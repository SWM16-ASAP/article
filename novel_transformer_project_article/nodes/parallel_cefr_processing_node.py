import concurrent.futures
import json
from ..state import BookState
from ..utils.logging_config import get_logger
from ..utils.workflow_helpers import update_usage_metrics
from .generate_level_specific_text_node import generate_text_for_level
from ..utils.pysbd_handler import get_pysbd_segmenter

logger = get_logger(__name__)

def get_cumulative_context(state: BookState, current_chapter_idx: int) -> str:
    """현재 챕터 이전까지의 누적 요약을 동적으로 생성합니다."""
    if current_chapter_idx == 0:
        return ""
    
    previous_summaries = []
    for i in range(current_chapter_idx): # 0 부터 시작
        chapter_meta = state["chapter_metadata"][i]
        previous_summaries.append(f"Chapter {i+1}: {chapter_meta['summary']}")
    
    return "\n".join(previous_summaries) # 구분자로 문자열 연결


def get_previous_chunk_context(state: BookState) -> str:
    """바로 이전 텍스트 청크의 원문을 가져옵니다. (이미지/preserve 태그는 건너뜀)"""
    chapter_idx = state["current_chapter_index"]
    chunk_idx = state["current_chunk_index"]

    if chunk_idx == 0:
        return ""

    # 현재 청크 이전부터 역순으로 탐색
    for i in range(chunk_idx - 1, -1, -1):
        prev_chunk = state["chapter_chunks"][chapter_idx][i]
        # 이미지나 preserve 태그가 아닌 실제 텍스트 청크만 반환
        if not prev_chunk.startswith("[illustration:") and not prev_chunk.startswith("[preserve:"):
            return prev_chunk

    # 모든 이전 청크가 이미지/preserve인 경우
    return ""

def create_all_cefr_versions_parallel(state: BookState) -> BookState:
    """Generates all CEFR versions in parallel using a single generic node."""
    chapter_idx = state['current_chapter_index']
    chunk_idx = state['current_chunk_index']
    chapter_num = chapter_idx + 1
    chunk_num = chunk_idx + 1
    
    logger.info(f"=== Chapter {chapter_num}, Chunk {chunk_num}: Starting parallel CEFR level conversion ===")
    
    try:
        cefr_levels_to_process = ["A0", "A1", "A2", "B1", "B2", "C1", "C2"]
        original_level = state["original_text_level"]
        current_chunk_text = state["chapter_chunks"][chapter_idx][chunk_idx]
        use_rag = False # 여기서 rag 결정

        # illustration 태그 처리
        if current_chunk_text.startswith("[illustration:"):
            try:
                json_part = current_chunk_text.strip()[len('[illustration:'):-1]
                image_data = json.loads(json_part)

                image_chunk = {
                    "chunkNum": chunk_idx,
                    "isImage": True,
                    "chunkText": image_data.get("file", "error.jpg"),
                    "description": image_data.get("description", "")
                }

                for res in state["leveled_results"]:
                    # Ensure the chunk hasn't already been added (e.g., from a previous failed run)
                    if not any(chunk["chunkNum"] == chunk_idx for chunk in res["chapters"][chapter_idx]["chunks"]):
                        res["chapters"][chapter_idx]["chunks"].append(image_chunk)

                logger.info(f"Image chunk for '{image_chunk['chunkText']}' added to all levels.")
                return state

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format for illustration chunk: {current_chunk_text}")
                # Create an error chunk if parsing fails
                error_chunk = {
                    "chunkNum": chunk_idx,
                    "isImage": True,
                    "chunkText": "parsing_error.jpg",
                    "description": f"Error: {e}"
                }

                for res in state["leveled_results"]:
                    if not any(chunk["chunkNum"] == chunk_idx for chunk in res["chapters"][chapter_idx]["chunks"]):
                        res["chapters"][chapter_idx]["chunks"].append(error_chunk)
                return state

        # Preserve 태그 처리
        if current_chunk_text.startswith("[preserve:"):
            try:
                preserve_content = current_chunk_text.strip()[len('[preserve:'):-1]
                preserve_chunk = {
                    "chunkNum": chunk_idx,
                    "isImage": False,
                    "chunkText": preserve_content,
                    "description": None
                }
                for res in state["leveled_results"]:
                    if not any(chunk["chunkNum"] == chunk_idx for chunk in res["chapters"][chapter_idx]["chunks"]):
                        res["chapters"][chapter_idx]["chunks"].append(preserve_chunk)
                return state
            except Exception as e:
                logger.error(f"Error parsing Preserve tag: {current_chunk_text}")
                return state


        # 1. Store the original text - 문장 단위로 분리
        segmenter = get_pysbd_segmenter()
        original_sentences = segmenter.segment(current_chunk_text)

        # 후처리: 짧고 알파벳이 없는 문장을 이전 문장에 병합
        processed_original_sentences = []
        for sentence in original_sentences:
            sentence = sentence.strip()

            # 빈 문장은 스킵
            if not sentence:
                continue

            # 조건: 3자 이내 + 알파벳이 하나도 없음
            if len(sentence) <= 3 and not any(c.isalpha() for c in sentence):
                # 이전 문장이 있으면 붙이기
                if processed_original_sentences:
                    processed_original_sentences[-1] += sentence
                # 첫 문장이면 스킵 (붙일 곳이 없음)
            else:
                # 정상 문장은 그냥 추가
                processed_original_sentences.append(sentence)

        for res in state["leveled_results"]:
            if res["textLevel"] == original_level:
                # 각 문장을 별도의 청크로 추가
                for sentence in processed_original_sentences:
                    res["chapters"][chapter_idx]["chunks"].append({
                        "chunkNum": chunk_idx,
                        "isImage": False,
                        "chunkText": sentence,
                        "description": None
                    })
                break

        # 2. Prepare context and levels for parallel processing
        context = {
            "cumulative_context": get_cumulative_context(state, chapter_idx) or "No previous chapters.",
            "current_chapter_summary": state["chapter_metadata"][chapter_idx]["summary"],
            "previous_chunk_context": get_previous_chunk_context(state) or "This is the first chunk of the chapter.",
            "feedback": "",
            "leveled_text": ""
        }

        # Exclude the original level from the list of levels to generate
        levels_to_run = [level for level in cefr_levels_to_process if level != original_level]

        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(levels_to_run)) as executor:
            # Map each level to a future
            future_to_level = {executor.submit(generate_text_for_level, level, use_rag, current_chunk_text, context, state): level for level in levels_to_run}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_level):
                level = future_to_level[future]
                try:
                    generated_text = future.result()

                    # 레벨링된 텍스트를 문장 단위로 분리
                    leveled_sentences = segmenter.segment(generated_text)

                    # 후처리: 짧고 알파벳이 없는 문장을 이전 문장에 병합
                    processed_leveled_sentences = []
                    for sentence in leveled_sentences:
                        sentence = sentence.strip()

                        # 빈 문장은 스킵
                        if not sentence:
                            continue

                        # 조건: 3자 이내 + 알파벳이 하나도 없음
                        if len(sentence) <= 3 and not any(c.isalpha() for c in sentence):
                            # 이전 문장이 있으면 붙이기
                            if processed_leveled_sentences:
                                processed_leveled_sentences[-1] += sentence
                            # 첫 문장이면 스킵 (붙일 곳이 없음)
                        else:
                            # 정상 문장은 그냥 추가
                            processed_leveled_sentences.append(sentence)

                    for res in state["leveled_results"]:
                        if res["textLevel"] == level:
                            # 각 문장을 별도의 청크로 추가
                            for sentence in processed_leveled_sentences:
                                res["chapters"][chapter_idx]["chunks"].append({
                                    "chunkNum": chunk_idx,
                                    "isImage": False,
                                    "chunkText": sentence,
                                    "description": None
                                })
                            break
                    
                except Exception as e:
                    logger.error(f"Error generating CEFR {level} version: {e}")
                    # Store the error message in the state for debugging
                    for res in state["leveled_results"]:
                        if res["textLevel"] == level:
                            res["chapters"][chapter_idx]["chunks"].append({
                                "chunkNum": chunk_idx,
                                "isImage": False,
                                "chunkText": f"[ERROR: {e}]",
                                "description": None
                            })
                            break

        logger.info(f"=== Chapter {chapter_num}, Chunk {chunk_num}: All CEFR level conversions complete ===")
        return state
    
    except Exception as e:
        error_msg = f"CEFR 레벨 변환 중 오류 발생 (Chapter {chapter_num}, Chunk {chunk_num}): {str(e)}"
        logger.error(error_msg)
        raise
    finally:
        pass
