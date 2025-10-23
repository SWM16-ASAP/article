"""
Google News RSS로 트렌드 수집 → LLM으로 주제 선정 → Tavily로 기사 수집하는 노드
"""
import os
import requests
from re import T
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException
import feedparser
from tavily import TavilyClient
from newspaper import Article
from ..state import BookState
from ..utils.logging_config import get_logger
from ..utils.workflow_helpers import setup_bedrock, BedrockTokenTrackingWrapper, send_discord_webhook
from langchain_core.prompts import ChatPromptTemplate

logger = get_logger(__name__)

# Pydantic 모델: LLM이 선정한 주제들
class SelectedTopic(BaseModel):
    title: str = Field(description="Selected topic title")
    url: str = Field(description="Selected topic url")

class TopicSelection(BaseModel):
    topics: List[SelectedTopic] = Field(description="List of 3 topic candidates in order of relevance and interest (most relevant first)")

# Pydantic 모델: LLM이 추출한 기사 정보
class ArticleExtraction(BaseModel):
    content: str = Field(description="Article main content")


def _fetch_trending_news_with_google_rss(category: str, language: str = "ko") -> List[Dict[str, str]]:
    """Google News RSS로 최신 뉴스 헤드라인 수집"""

    # 카테고리 및 언어별로 미리 정의된 RSS URL
    RSS_URLS = {
        "Sports": {
            "ko": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtdHZHZ0pMVWlnQVAB?hl=ko&gl=KR&ceid=KR:ko",
            "ja": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtdHZHZ0pMVWlnQVAB?hl=ja&gl=JP&ceid=JP:ja"
        },
        "Science": {
            "ko": "https://scitechdaily.com/feed/",
            "ja": "https://scitechdaily.com/feed/"
        },
        "Business": {
            "ko": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtdHZHZ0pMVWlnQVAB?hl=ko&gl=KR&ceid=KR:ko",
            "ja": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtdHZHZ0pMVWlnQVAB?hl=ja&gl=JP&ceid=JP:ja"
        },
        "Technology": {
            "ko": "https://www.cnbc.com/id/19854910/device/rss/rss.html",
            "ja": "https://www.cnbc.com/id/19854910/device/rss/rss.html"
        },
        "Culture": {
            "ko": "https://www.yna.co.kr/rss/entertainment.xml",
            "ja": "https://news.yahoo.co.jp/rss/categories/entertainment.xml"
        }
    }

    # URL 가져오기
    rss_url = RSS_URLS.get(category, {}).get(language)

    if not rss_url:
        logger.error(f"지원하지 않는 카테고리 또는 언어: category={category}, language={language}")
        raise ValueError(f"지원하지 않는 카테고리 또는 언어: {category}, {language}")

    logger.info(f"Google News RSS 요청: category={category}, language={language}")
    logger.info(f"RSS URL: {rss_url}")

    try:
        # RSS 피드 파싱
        feed = feedparser.parse(rss_url)

        if not feed.entries:
            logger.warning(f"RSS 피드에서 항목을 가져오지 못했습니다: {rss_url}")
            return []

        # 헤드라인 추출 (최대 30개)
        headlines = []
        for entry in feed.entries[:30]:
            headlines.append({
                "title": entry.get("title", ""),
                "description": entry.get("summary", ""),
                "url": entry.get("link", ""),
                "published_at": entry.get("published", datetime.now().isoformat()),
                "source": entry.get("source", {}).get("title", "Google News")
            })

        logger.info(f"Google News RSS로 최신 뉴스 {len(headlines)}개 수집 완료")
        return headlines

    except Exception as e:
        logger.error(f"Google News RSS 호출 중 오류: {e}")
        raise

def _remove_duplicate_headlines(headlines: List[Dict[str, str]], state: BookState) -> List[Dict[str, str]]:
    """
    백엔드를 통해 최근 기사들을 가져와서 headlines에서 최근 사용한 주제는 제거하기
    url 기반 중복 검사
    """

    # 백엔드 API 
    swagger_api_key = os.getenv("SWAPPER_API_KEY")
    backend_url = os.getenv("READ_PAST_ARTICLE_URL")

    targetLanguageCode = state.get("target_language_code")
    if targetLanguageCode == "common":
        targetLanguageCode = "ko"

    targetLanguageCode = targetLanguageCode.upper();
    
    # 쿼리 파라미터
    params = {
        "tags" : state.get("tags")[0].strip(),
        "targetLanguageCode" : targetLanguageCode,
        "page":1,
        "limit":10
    }

    # header
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY":swagger_api_key
    }

    try:
        logger.info(f"백엔드 API 호출: {backend_url} with params={params}")

        response = requests.get(
            backend_url,
            params=params,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        response_data = response.json()
        article_data = response_data.get("data", [])

        logger.info(f"백엔드에서 {len(article_data)}개 최근 기사 가져옴")

        # 백엔드 기사의 originUrl 목록 추출
        recent_urls = set()
        for article in article_data:
            origin_url = article.get("originUrl", "")
            if origin_url:
                recent_urls.add(origin_url.strip())

        logger.info(f"백엔드에서 {len(recent_urls)}개 고유 URL 추출")

        # headlines에서 중복 URL 제거
        filtered_headlines = []
        duplicates_found = 0

        for headline in headlines:
            headline_url = headline.get("url", "").strip()

            if headline_url in recent_urls:
                logger.info(f"중복 발견 (URL): {headline.get('title', '')[:50]}")
                duplicates_found += 1
            else:
                filtered_headlines.append(headline)

        logger.info(f"중복 제거 완료: {duplicates_found}개 제거, {len(filtered_headlines)}개 남음")

        return filtered_headlines

    except requests.exceptions.Timeout:
        logger.error("백엔드 API 타임아웃 - 중복 체크 건너뜀")
        return headlines  # 에러 시 원본 반환
    except Exception as e:
        logger.error(f"_remove_duplicate_headlines error: {e} - 중복 체크 건너뜀")
        return headlines  # 에러 시 원본 반환



def _select_topic_with_llm(headlines: List[Dict[str, str]], state: BookState) -> List[str]:
    """LLM을 사용하여 헤드라인에서 가장 적합한 주제 3개 선정"""
    logger.info("LLM을 사용하여 주제 3개 선정 중...")

    # 헤드라인 텍스트 생성
    headlines_text = "\n".join([
        f"{i+1}. {h['title']} - {h['description'][:100] if h['description'] else ''}"
        for i, h in enumerate(headlines[:30])
    ])

    # LLM 설정
    config = {
        "model": "us.meta.llama4-scout-17b-instruct-v1:0",
        "temperature": 0.3
    }
    llm = setup_bedrock(config=config)
    llm = BedrockTokenTrackingWrapper(llm, state)

    # 파서 설정
    base_parser = PydanticOutputParser(pydantic_object=TopicSelection)
    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=llm,
        max_retries=3
    )

    # 프롬프트
    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate(
        template="""다음은 오늘의 뉴스 헤드라인입니다:

        {headlines}

        위 헤드라인들을 분석하여 한국과 일본 독자들에게 가장 흥미롭고 교육적인 기사 주제를 **3개** 선정하세요.
        선정된 기사들의 제목과 url을 함께 반환해주세요
        가장 메이저하고 중요한 순서대로 정렬해주세요 (1번이 가장 중요).


        **Critical**: 선정된 기사에 대해서는 주제의 제목과 url을 변경하거나 번역하지 말고 그대로 반환하세요!!!

        선정 기준:
        1. **메이저한 주제**: 하루에 하나 올리기에 충분히 중요한 주제
        - 사회적 영향력이 큰 사건/정책/기술 혁신
        - 국제적으로도 관심을 가질 만한 주제
        - 장기적 관심사 (단발성 사건 지양)

        2. **적절성**: 다음 주제는 반드시 제외
        - 가십/연예인 사생활/스캔들
        - 정치적 논란/당파적 이슈
        - 개인 폭로/루머성 기사
        - 선정적이거나 자극적인 내용

        3. **교육적 가치**: 독자에게 유익한 정보 제공
        - 새로운 지식/인사이트 제공
        - 트렌드의 배경과 맥락 이해
        - 실생활에 도움이 되는 정보

        4. **한국-일본 공통 관심사 우대**:
        - 양국 독자 모두에게 의미 있는 주제 우선
        - 글로벌 관점에서 접근 가능한 주제


        {format_instructions}
        """,
        input_variables=["headlines"],
        partial_variables={"format_instructions": base_parser.get_format_instructions()}
    )

    chain = prompt | llm
    raw_response = chain.invoke({"headlines": headlines_text})

    try:
        response = base_parser.parse(raw_response.content)
    except OutputParserException as e:
        logger.info(f"주제 선정 파싱 실패, OutputFixingParser로 복구 시도: {str(e)[:50]}...")
        response = fixing_parser.parse(raw_response.content)
        logger.info("OutputFixingParser를 통한 파싱 복구 성공")

    logger.info(f"선정된 주제 후보 {len(response.topics)}개:")
    for i, topic in enumerate(response.topics, 1):
        logger.info(f"  {i}. {topic}")

    return response.topics


def _fetch_articles_with_tavily(topic: str, max_results: int = 20, min_score: float = 0.5) -> List[Dict[str, str]]:
    """
    Tavily로 주제 관련 기사 링크 및 raw_content 수집

    Args:
        topic: 검색할 주제
        max_results: 최대 검색 결과 수 (기본값: 20)
        min_score: 최소 점수 (기본값: 0.5, 이 점수 이상인 결과만 반환)

    Returns:
        score가 min_score 이상인 기사 정보 리스트 (score 내림차순 정렬)
        각 항목: {"url": str, "raw_content": str, "score": float, "title": str}
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY 환경변수가 설정되지 않았습니다.")

    tavily = TavilyClient(api_key=tavily_key)

    logger.info(f"Tavily로 '{topic}' 관련 기사 {max_results}개 검색 중 (최소 점수: {min_score})...")

    try:
        response = tavily.search(
            query=topic,
            max_results=max_results,
            include_raw_content="markdown"
        )

        articles = response.get("results", [])

        # score >= min_score인 기사만 필터링
        filtered_articles = [
            article for article in articles
            if article.get("score", 0) >= min_score
        ]

        # score 내림차순 정렬
        sorted_articles = sorted(
            filtered_articles,
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        # URL, raw_content, score, title 추출
        article_data = [
            {
                "url": article.get("url", ""),
                "raw_content": article.get("raw_content", ""),
                "score": article.get("score", 0),
                "title": article.get("title", "")
            }
            for article in sorted_articles
            if article.get("url") and article.get("raw_content")
        ]

        logger.info(f"Tavily 검색 완료: 전체 {len(articles)}개 중 점수 {min_score} 이상 {len(article_data)}개 필터링")

        # 상위 결과 로깅
        for i, article in enumerate(article_data[:5], 1):
            logger.info(f"  {i}. [{article['score']:.2f}] {article['title'][:50]}")

        return article_data

    except Exception as e:
        logger.error(f"Tavily 호출 중 오류: {e}")
        raise


def _parse_articles_with_llm(article_data: List[Dict[str, str]], target_count: int = 3, state: BookState = None) -> str:
    """
    LLM을 사용하여 raw_content에서 제목과 본문 추출

    Args:
        article_data: Tavily에서 가져온 기사 데이터 리스트
                     각 항목: {"url": str, "raw_content": str, "score": float, "title": str}
        target_count: 목표 기사 개수 (기본값: 3)
        state: BookState (에러 알림용, 토큰 추적용)

    Returns:
        파싱된 기사들을 결합한 텍스트 (포맷: === 기사 제목 ===\n본문)

    Raises:
        ValueError: 파싱된 기사가 1개 이하인 경우
    """
    logger.info(f"LLM을 사용하여 기사 파싱 시작 (목표: {target_count}개)")

    # LLM 설정
    config = {
        "model": "us.meta.llama4-scout-17b-instruct-v1:0",
        "temperature": 0.3,
        "max_tokens": 2000
    }
    llm = setup_bedrock(config=config)
    llm = BedrockTokenTrackingWrapper(llm, state)

    # 파서 설정
    base_parser = PydanticOutputParser(pydantic_object=ArticleExtraction)
    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=llm,
        max_retries=1
    )

    # 프롬프트

    prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 본문 추출 전문가야. 
        
        사용자가 기사를 스크랩해서 마크 다운 형식으로 주면 너는 거기서 광고, url 과 같은 잡다한 정보들을 걷어내고 기사의 본문만 추출해서 사용자에게 제공을 해


        구조는 아래와 같이 줘야 해
        
        {format_instructions}"""),
        ("human", """ 기사를 단순 스크롤한 정보는 아래와 같아.

        {raw_content}

        여기서 잡다한 정보를 걸러내고 기사의 본문한 추출해서 제공을 해줘""")
    ]).partial(format_instructions=base_parser.get_format_instructions())

    chain = prompt | llm

    parsed_articles = []
    failed_count = 0

    for i, article in enumerate(article_data):
        # 목표 개수 달성하면 종료
        if len(parsed_articles) >= target_count:
            logger.info(f"목표 기사 {target_count}개 파싱 완료")
            break

        try:
            logger.info(f"[{i+1}/{len(article_data)}] LLM 파싱 시도: {article['title'][:50]}")

            # LLM 호출
            raw_response = chain.invoke({"raw_content": article["raw_content"][:15000]})  # 토큰 제한 고려

            # 파싱
            try:
                extracted = base_parser.parse(raw_response.content)
            except OutputParserException as e:
                logger.info(f"  파싱 실패, OutputFixingParser로 복구 시도: {str(e)[:50]}...")
                extracted = fixing_parser.parse(raw_response.content)
                logger.info("  OutputFixingParser를 통한 파싱 복구 성공")

            # 제목과 본문 검증
            title = article["title"].strip()
            content = extracted.content.strip() if extracted.content else ""

            if not title or not content or len(content) < 50:
                logger.warning(f"  ⚠️ 제목 또는 본문이 부족함 - 제목: '{title[:50]}', 본문 길이: {len(content)}자")
                failed_count += 1
                continue

            # 성공적으로 파싱됨
            parsed_articles.append({
                "title": title,
                "content": content,
                "url": article["url"],
                "score": article["score"]
            })
            logger.info(f"  ✅ 파싱 성공 - 제목: {title[:50]}..., 본문 길이: {len(content)}자")

        except Exception as e:
            logger.warning(f"  ❌ LLM 파싱 실패: {article['url']} - 오류: {str(e)[:100]}")
            failed_count += 1
            continue

    # 결과 확인
    parsed_count = len(parsed_articles)
    logger.info(f"파싱 완료: 성공 {parsed_count}개, 실패 {failed_count}개")

    # 1개 이하면 에러
    if parsed_count <= 1:
        error_msg = f"기사 파싱 실패: {parsed_count}개만 성공 (최소 2개 필요). 주제: {state.get('topic', 'N/A') if state else 'N/A'}"
        logger.error(error_msg)

        # 디스코드 알림
        discord_message = f"""⚠️ **기사 파싱 실패** ⚠️

        **Request ID**: {state.get('id', 'N/A') if state else 'N/A'}
        **주제**: {state.get('topic', 'N/A') if state else 'N/A'}
        **카테고리**: {state.get('tags', ['N/A'])[0] if state and state.get('tags') else 'N/A'}
        **언어**: {state.get('language', 'N/A') if state else 'N/A'}

        **파싱 결과**: {parsed_count}개 성공 (최소 2개 필요)
        **시도한 기사 개수**: {len(article_data)}개
        **실패한 개수**: {failed_count}개

        LLM으로 기사를 파싱했으나 충분한 기사를 확보하지 못했습니다."""

        send_discord_webhook(discord_message)
        raise ValueError(error_msg)

    # 2개 이상이면 진행
    logger.info(f"✅ 파싱된 기사 {parsed_count}개로 진행")

    # 기사들을 하나의 텍스트로 결합
    combined_text = ""
    for i, article in enumerate(parsed_articles, 1):
        combined_text += f"\n\n=== {article['title']} ===\n"
        combined_text += article['content']

    logger.info(f"결합된 텍스트 길이: {len(combined_text):,}자")

    return combined_text.strip()


def _parse_articles_with_newspaper(article_urls: List[str], target_count: int = 3, state: BookState = None) -> str:
    """
    newspaper3k를 사용하여 기사 URL에서 제목과 본문 파싱

    Args:
        article_urls: 파싱할 기사 URL 리스트
        target_count: 목표 기사 개수 (기본값: 3)
        state: BookState (에러 알림용)

    Returns:
        파싱된 기사들을 결합한 텍스트

    Raises:
        ValueError: 파싱된 기사가 1개 이하인 경우
    """
    logger.info(f"newspaper3k로 기사 파싱 시작 (목표: {target_count}개)")

    parsed_articles = []
    failed_urls = []

    for i, url in enumerate(article_urls):
        # 목표 개수 달성하면 종료
        if len(parsed_articles) >= target_count:
            logger.info(f"목표 기사 {target_count}개 파싱 완료")
            break

        try:
            logger.info(f"[{i+1}/{len(article_urls)}] 파싱 시도: {url}")

            # Article 객체 생성 및 다운로드
            article = Article(url)
            article.download()
            article.parse()

            # 제목과 본문 검증
            title = article.title.strip() if article.title else ""
            text = article.text.strip() if article.text else ""

            if not title or not text:
                logger.warning(f"  ⚠️ 제목 또는 본문이 비어있음 - 제목: '{title[:50]}', 본문 길이: {len(text)}")
                failed_urls.append(url)
                continue

            # 성공적으로 파싱됨
            parsed_articles.append({
                "title": title,
                "text": text,
                "url": url
            })
            logger.info(f"  ✅ 파싱 성공 - 제목: {title[:50]}..., 본문 길이: {len(text)}자")

        except Exception as e:
            logger.warning(f"  ❌ 파싱 실패: {url} - 오류: {str(e)[:100]}")
            failed_urls.append(url)
            continue

    # 결과 확인
    parsed_count = len(parsed_articles)
    logger.info(f"파싱 완료: 성공 {parsed_count}개, 실패 {len(failed_urls)}개")

    # 1개 이하면 에러
    if parsed_count <= 1:
        error_msg = f"기사 파싱 실패: {parsed_count}개만 성공 (최소 2개 필요). 주제: {state.get('topic', 'N/A') if state else 'N/A'}"
        logger.error(error_msg)

        # 디스코드 알림
        discord_message = f"""⚠️ **기사 파싱 실패** ⚠️

        **Request ID**: {state.get('id', 'N/A') if state else 'N/A'}
        **주제**: {state.get('topic', 'N/A') if state else 'N/A'}
        **카테고리**: {state.get('tags', ['N/A'])[0] if state and state.get('tags') else 'N/A'}
        **언어**: {state.get('language', 'N/A') if state else 'N/A'}

        **파싱 결과**: {parsed_count}개 성공 (최소 2개 필요)
        **시도한 URL 개수**: {len(article_urls)}개
        **실패한 URL 개수**: {len(failed_urls)}개

        newspaper3k로 기사를 파싱했으나 충분한 기사를 확보하지 못했습니다."""

        send_discord_webhook(discord_message)
        raise ValueError(error_msg)

    # 2개 이상이면 진행
    logger.info(f"✅ 파싱된 기사 {parsed_count}개로 진행")

    # 기사들을 하나의 텍스트로 결합
    combined_text = ""
    for i, article in enumerate(parsed_articles, 1):
        combined_text += f"\n\n=== 기사 {i}: {article['title']} ===\n"
        combined_text += article['text']

    logger.info(f"결합된 텍스트 길이: {len(combined_text):,}자")

    return combined_text.strip()


def fetch_topic_and_articles(state: BookState) -> BookState:
    """
    NewsAPI로 주제 선정 → Tavily로 기사 수집

    state["full_text"]가 이미 있으면 skip (기존 워크플로우 사용)
    없으면 자동으로 주제 선정 및 기사 수집
    """
    try:
        logger.info("=== 주제 선정 및 기사 수집 시작 ===")

        # 1. 초기화 (Article은 단일 챕터만 가짐)
        if not state.get("chapter_metadata"):
            from ..state import ChapterMetadata
            initial_metadata = [ChapterMetadata(
                chapterNum=0,  # 항상 0번 챕터
                title=None,
                summary=""
            )]
            state["chapter_metadata"] = initial_metadata
            logger.info("chapter_metadata 초기화 완료")

        # 2. CEFR 레벨별 결과 구조 초기화 (Article은 항상 1개 챕터)
        if not state.get("leveled_results"):
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
            state["leveled_results"] = leveled_results
            logger.info("leveled_results 초기화 완료")

        # full_text가 이미 있으면 skip
        if state.get("full_text") and state["full_text"].strip():
            logger.info("full_text가 이미 존재합니다. 주제 선정 단계를 건너뜁니다.")
            return state

        # tags에서 카테고리 추출
        tags = state.get("tags", [])
        if not tags:
            raise ValueError("tags가 비어있습니다. 카테고리를 지정해주세요.")

        category = tags[0]  # 첫 번째 태그를 카테고리로 사용

        # 언어 감지 (id 또는 별도 필드로 판단 가능)
        # 현재는 한국어를 기본으로 설정
        # TODO: state에서 언어 정보 가져오기 (예: state.get("language", "ko"))
        language = state.get("target_language_code")  # "ko" 또는 "ja"

        logger.info(f"카테고리: {category}, 언어: {language}")

        if language == "common":
            language = "ko"

        # 1. Google News RSS로 트렌드 수집
        headlines = _fetch_trending_news_with_google_rss(category=category, language=language)

        if not headlines:
            raise ValueError(f"뉴스 API에서 트렌드를 가져오지 못했습니다. (카테고리: {category}, 언어: {language})")


        # 2.5. 백엔드에 api를 쏴서 겹치는 것들 제거

        headlines = _remove_duplicate_headlines(headlines, state);
        # 2. LLM으로 주제 3개 선정 (중요도 순)
        topic_candidates = _select_topic_with_llm(headlines, state)

        # 3. 각 주제에 대해 순차적으로 시도
        selected_topic = None
        selected_topic_url = None
        article_data = None

        for i, topic in enumerate(topic_candidates, 1):
            logger.info(f"주제 후보 {i}/{len(topic_candidates)} 시도 중: {topic}")

            try:
                # Tavily로 관련 기사 데이터 수집 (URL + raw_content)
                article_data = _fetch_articles_with_tavily(topic.title, max_results=20, min_score=0.5)

                # score 0.5 이상인 기사가 5개 이상인지 확인
                if len(article_data) >= 5:
                    logger.info(f"✅ 주제 '{topic.title}': {len(article_data)}개 기사 확보 - 사용 가능")
                    selected_topic = topic
                    selected_topic_url = topic.url
                    break
                else:
                    logger.warning(f"⚠️ 주제 '{topic.title}': {len(article_data)}개 기사만 확보 (5개 미만) - 다음 주제 시도")
                    continue

            except Exception as e:
                logger.warning(f"❌ 주제 '{topic.title}' 기사 수집 실패: {str(e)[:100]} - 다음 주제 시도")
                continue

        # 모든 주제에서 실패한 경우
        if not selected_topic or not article_data:
            error_msg = f"모든 주제 후보에서 충분한 기사를 찾지 못했습니다. 시도한 주제: {', '.join([t.title for t in topic_candidates[:3]])}"
            logger.error(error_msg)

            # 디스코드 알림
            discord_message = f"""⚠️ **기사 수집 실패** ⚠️

            **Request ID**: {state.get('id', 'N/A')}
            **카테고리**: {category}
            **언어**: {language}

            **시도한 주제 후보들**:
            {chr(10).join([f"{i}. {topic.title}" for i, topic in enumerate(topic_candidates, 1)])}

            모든 주제에서 score 0.5 이상 기사 5개 이상을 확보하지 못했습니다."""

            send_discord_webhook(discord_message)
            raise ValueError(error_msg)

        logger.info(f"최종 선정된 주제: {selected_topic.title}")

        # 4. LLM으로 기사 파싱 (최소 2개, 목표 3개)
        articles_text = _parse_articles_with_llm(article_data, target_count=3, state=state)

        # 5. state에 저장
        state["full_text"] = articles_text
        state["chapters"] = [articles_text]
        state["origin_url"] = selected_topic_url

        logger.info(f"=== 주제 선정 및 기사 수집 완료 ===")
        logger.info(f"선정된 주제: {selected_topic.title}")
        logger.info(f"수집된 기사 길이: {len(articles_text):,}자")

        return state

    except Exception as e:
        logger.error(f"주제 선정 및 기사 수집 중 오류: {e}")
        raise
