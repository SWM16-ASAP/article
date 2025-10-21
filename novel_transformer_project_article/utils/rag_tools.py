from langchain_core.tools import tool
from typing import List, Dict, Any
from .rag_handler import RAGHandler
from .logging_config import get_logger

logger = get_logger(__name__)

@tool
def find_alternative_words(words: str, target_level: str, context: str = "") -> Dict[str, List[str]]:
    """
    Finds alternative words for multiple words at a specific CEFR target level.
    Use this when multiple words in the original text need to be simplified.
    
    Args:
        words: Comma-separated string of words to find alternatives for (e.g., "word1, word2, word3")
        target_level: CEFR level (A1, A2, B1, B2, C1, C2)
        context: Surrounding text context for better matching
    
    Returns:
        Dictionary mapping original words to their best alternative (original_word: alternative_word)
    """
    # 쉼표로 구분된 문자열을 리스트로 파싱
    if not words or not words.strip():
        logger.warning("빈 단어 문자열이 입력되었습니다.")
        return {}
    
    # 쉼표로 구분하여 단어 리스트 생성
    word_list = [word.strip() for word in words.split(",") if word.strip()]
    
    if not word_list:
        logger.warning("유효한 단어가 없습니다.")
        return {}
    
    if target_level not in ["A0", "A1", "A2", "B1", "B2", "C1", "C2"]:
        logger.warning(f"유효하지 않은 CEFR 레벨: {target_level}")
        return {}
    
    logger.info(f"RAG Tool batch called: {len(word_list)} words → {target_level} level")
    
    try:
        # 싱글톤 인스턴스 사용 (연결 재사용)
        rag_handler = RAGHandler.get_instance()
        
        # 각 단어별로 대안 단어 검색 (순차 처리)
        alternatives_dict = {}
        for word in word_list:
            alternatives = rag_handler.get_alternative_words(
                word=word,
                target_level=target_level,
                context=context,
                limit=5
            )
            alternatives_dict[word] = alternatives
        
        
        logger.info(f"Batch processing completed: {len(alternatives_dict)} words processed")
        return alternatives_dict
        
    except Exception as e:
        logger.error(f"RAG Tool batch error: {e}")
        return {}
