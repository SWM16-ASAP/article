import json
from typing import List, Dict, Any, Optional
from langchain_aws import BedrockEmbeddings
from .db_handler import setup_mongo_client
from .logging_config import get_logger

logger = get_logger(__name__)

class RAGHandler:
    _instance = None
    _initialized = False
    
    # cls => class의 줄임말
    # 싱글톤으로 하나의 인스턴스를 생성하고 __new__로 하나의 인스턴스를 등록해놓는다.
    def __new__(cls, *args, **kwargs):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 database_name: str = "rag", 
                 collection_name: str = "words",
                 model_id: str = "amazon.titan-embed-text-v1",
                 aws_region: str = "us-east-1"):
        """
        RAG 핸들러 초기화 (싱글톤 - 한 번만 초기화)
        
        Args:
            database_name: MongoDB 데이터베이스 이름
            collection_name: 임베딩이 저장된 컬렉션 이름
            model_id: 사용할 Bedrock 임베딩 모델 ID
            aws_region: AWS 리전
        """
        # 이미 초기화된 경우 건너뛰기
        if self._initialized:
            return
            
        self.database_name = database_name
        self.collection_name = collection_name
        self.model_id = model_id
        self.aws_region = aws_region
        
        # MongoDB 클라이언트 설정
        self.mongo_client = setup_mongo_client()
        
        if not self.mongo_client:
            logger.error("MongoDB 연결 실패")
            raise Exception("MongoDB 연결을 설정할 수 없습니다.")
            
        self.db = self.mongo_client[database_name]
        self.collection = self.db[collection_name]
        
        # Bedrock 임베딩 모델 로드
        try:
            self.embeddings = BedrockEmbeddings(
                model_id=model_id,
                region_name=aws_region
            )
            logger.info(f"Bedrock 임베딩 모델 로드 완료: {model_id}")
        except Exception as e:
            logger.error(f"Bedrock 임베딩 모델 로드 실패: {e}")
            raise
        
        # 초기화 완료 플래그
        self._initialized = True
        logger.info("RAG 핸들러 싱글톤 초기화 완료")
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_query_embedding(self, query: str) -> List[float]:
        """쿼리 텍스트를 임베딩 벡터로 변환합니다."""
        try:
            embedding = self.embeddings.embed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"임베딩 변환 실패: {e}")
            return []

    def get_alternative_words(self, 
                            word: str, 
                            target_level: str, 
                            context: str = "", 
                            limit: int = 5) -> List[str]:
        """
        특정 단어의 대안 단어들을 찾습니다.
        
        Args:
            word: 원본 단어
            target_level: 대상 CEFR 레벨 (A1, A2, B1, B2, C1, C2)
            context: 문맥 정보
            limit: 반환할 대안 단어 수
            
        Returns:
            대안 단어들의 리스트
        """
        try:
            # MongoDB 연결 확인
            if not self.mongo_client:
                logger.error("MongoDB 연결이 없습니다. 빈 대안 단어 리스트 반환")
                return []
            
            # 컨텍스트와 함께 검색 쿼리 생성
            search_query = f"{word} {context}".strip()
            
            # 메타데이터 필터 설정 (실제 데이터 구조에 맞게 수정)
            metadata_filter = {
                "level": target_level
            }
            
            # 벡터 검색 수행
            results = self.vector_search(search_query, limit * 2, metadata_filter)  # 더 많은 결과로 시작
            
            # 단어만 추출하고 원본 단어 제외
            alternatives = []
            for result in results:
                if 'text' not in result:
                    continue
                    
                word_text = result['text']
                # "happy_adj" → "happy"
                clean_word = word_text.split('_')[0]
                
                # 원본 단어와 다르고 아직 리스트에 없으면 추가
                if (clean_word.lower() != word.lower() and 
                    clean_word.lower() not in [w.lower() for w in alternatives] and
                    len(alternatives) < limit):
                    alternatives.append(clean_word)
            
            logger.info(f"'{word}' 대안 단어 {len(alternatives)}개 찾음: {alternatives}")
            return alternatives
            
        except Exception as e:
            logger.error(f"대안 단어 검색 중 오류 발생: {e}")
            return []
    
    
    def vector_search(self, 
                     query: str, 
                     limit: int, 
                     metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        벡터 유사도 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            limit: 반환할 결과 수
            metadata_filter: 메타데이터 필터 조건
            
        Returns:
            검색 결과 리스트
        """
        try:
            # MongoDB 연결 확인
            if not self.mongo_client:
                logger.error("MongoDB 연결이 없습니다.")
                return []
            
            # 쿼리 임베딩 생성
            query_embedding = self.get_query_embedding(query)
            
            if not query_embedding:
                logger.warning("쿼리 임베딩 생성 실패")
                return []
            
            # MongoDB 벡터 검색 파이프라인 구성
            pipeline = [   
                {
                    "$vectorSearch": {
                        "index": "vector_index",  # Atlas Vector Search 인덱스 이름
                        "path": "bedrock_embedding",      # 임베딩 필드 이름
                        "queryVector": query_embedding,
                        "numCandidates": max(limit * 10, 100),  # similarity search 검색 후 후보 수
                        "limit": limit # 최종 반환 수
                    }
                }
            ]
            
            # 메타데이터 필터 추가
            if metadata_filter:
                pipeline.append({"$match": metadata_filter}) # $match : 메타데이터 필터 추가
            
            # 결과 프로젝션 (실제 데이터 구조에 맞게 수정)
            pipeline.append({
                "$project": {
                    "_id": 1,  # 1 = 포함 (include)
                    "text": "$bedrock_text_chunk",  # 실제 텍스트 필드 매핑
                    "level": 1,  # 레벨 필드 포함
                    "score": {"$meta": "vectorSearchScore"} 
                }
            })
            
            # 검색 실행
            results = list(self.collection.aggregate(pipeline))
            
            logger.info(f"벡터 검색 완료: '{query[:50]}...' → {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            # 구체적인 오류 유형에 따른 추가 로깅
            if "index not found" in str(e).lower():
                logger.error("MongoDB Vector Search 인덱스 'vector_index'가 생성되지 않았습니다.")
            elif "namespace" in str(e).lower():
                logger.error("MongoDB 데이터베이스 또는 컬렉션이 존재하지 않습니다.")
            elif "connection" in str(e).lower():
                logger.error("MongoDB 연결 오류가 발생했습니다.")
            
            return []
    
    def close(self):
        """
        MongoDB 연결을 닫습니다.
        주의: 싱글톤 패턴에서는 일반적으로 연결을 유지합니다.
        프로세스 종료 시에만 호출하세요.
        """
        logger.info("RAG 핸들러 연결 유지 중 (싱글톤)")
        # 싱글톤에서는 연결을 닫지 않음
        pass
    
    def force_close(self):
        """
        강제로 MongoDB 연결을 닫고 싱글톤을 리셋합니다.
        프로세스 종료 시에만 사용하세요.
        """
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("MongoDB 연결 강제 해제")
        
        # 싱글톤 리셋
        RAGHandler._instance = None
        RAGHandler._initialized = False
        logger.info("RAG 핸들러 싱글톤 리셋 완료") 