import os
import json
import requests
from re import T
from typing import List, Dict
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException
import feedparser
from ..state import BookState
from ..utils.logging_config import get_logger
from ..utils.workflow_helpers import setup_bedrock, BedrockTokenTrackingWrapper
from ..utils.langfuse_client import (
    get_langfuse_client,
    is_langfuse_enabled,
    get_prompt_label,
    convert_langfuse_to_langchain,
    get_model_config_from_prompt
)
from ..prompts.select_topic_with_llm_prompt import get_select_topic_prompt
from ..prompts.outputFixingParser_prompt import get_output_fixing_prompt

# Pydantic 모델: LLM이 선정한 주제들
class TopicSelection(BaseModel):
    topics: List[str] = Field(description="List of 3 topic titles only in order of relevance and interest (most relevant first)")

# Pydantic 모델: 선정된 주제 (제목 + URL)
class SelectedTopic(BaseModel):
    title: str = Field(description="Topic title")
    url: str = Field(description="Topic URL")

logger = get_logger(__name__)

def _fetch_trending_news_with_google_rss(category: str, language: str = "KO") -> List[Dict[str, str]]:
    """Google News RSS로 최신 뉴스 헤드라인 수집"""

    # 카테고리 및 언어별로 미리 정의된 RSS URL
    RSS_URLS = {
        "Sports": {
            "KO": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtdHZHZ0pMVWlnQVAB?hl=ko&gl=KR&ceid=KR:ko",
            "JA": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtdHZHZ0pMVWlnQVAB?hl=ja&gl=JP&ceid=JP:ja"
        },
        "Science": {
            "KO": [
                "https://www.sciencenews.org/feed",
                "https://www.sciencedaily.com/rss/all.xml",
                "https://www.livescience.com/feeds.xml",
                "https://phys.org/rss-feed/"
            ],
            "JA": [
                "https://www.sciencenews.org/feed",
                "https://www.sciencedaily.com/rss/all.xml",
                "https://www.livescience.com/feeds.xml",
                "https://phys.org/rss-feed/"
            ]
        },
        "Business": {
            "KO": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtdHZHZ0pMVWlnQVAB?hl=ko&gl=KR&ceid=KR:ko",
            "JA": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtdHZHZ0pMVWlnQVAB?hl=ja&gl=JP&ceid=JP:ja"
        },
        "Technology": {
            "KO": "https://www.cnbc.com/id/19854910/device/rss/rss.html",
            "JA": "https://www.cnbc.com/id/19854910/device/rss/rss.html"
        },
        "Culture": {
            "KO": "https://www.yna.co.kr/rss/entertainment.xml",
            "JA": "https://news.yahoo.co.jp/rss/categories/entertainment.xml"
        }
    }

    # URL 가져오기
    rss_data = RSS_URLS.get(category, {}).get(language)

    if not rss_data:
        logger.error(f"지원하지 않는 카테고리 또는 언어: category={category}, language={language}")
        raise ValueError(f"지원하지 않는 카테고리 또는 언어: {category}, {language}")

    # Science 카테고리의 경우 날짜 기반 로테이션
    if category == "Science" and isinstance(rss_data, list):
        day_of_year = datetime.now().timetuple().tm_yday
        rss_index = day_of_year % len(rss_data)
        rss_url = rss_data[rss_index]
        logger.info(f"Science 카테고리 로테이션: day_of_year={day_of_year}, index={rss_index}")
    else:
        rss_url = rss_data

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

def _fetch_recent_articles(state: BookState) -> List[Dict[str, str]]:
    """
    백엔드 API를 통해 최근 발행된 기사들을 가져옵니다.

    Returns:
        List[Dict[str, str]]: 백엔드에서 가져온 최근 기사 리스트
            각 항목: {"title": str, "originUrl": str, ...}
    """
    # 백엔드 API
    swagger_api_key = os.getenv("SWAPPER_API_KEY")
    backend_url = os.getenv("READ_PAST_ARTICLE_URL")

    targetLanguageCode = state.get("target_language_code")

    # None이면 기본값 "KO" 설정
    if targetLanguageCode is None:
        targetLanguageCode = "KO"

    # 쿼리 파라미터
    params = {
        "tags": state.get("tags")[0].strip(),
        "targetLanguageCode": targetLanguageCode,
        "page": 1,
        "limit": 10
    }

    # header
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": swagger_api_key
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

        return article_data

    except requests.exceptions.Timeout:
        logger.error("_fetch_recent_articles error - 백엔드 API 타임아웃 - 빈 리스트 반환")
        return []
    except Exception as e:
        logger.error(f"_fetch_recent_articles error {e}")
        return []


def _filter_duplicate_headlines(headlines: List[Dict[str, str]], recent_articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    URL 기반으로 최근 기사와 중복되는 헤드라인을 필터링합니다.
    (명확한 중복만 제거 - 동일 URL)

    Args:
        headlines: RSS에서 가져온 헤드라인 리스트
        recent_articles: 백엔드에서 가져온 최근 기사 리스트

    Returns:
        List[Dict[str, str]]: 중복이 제거된 헤드라인 리스트
    """
    # 최근 기사들의 URL set 생성
    recent_urls = set()
    for article in recent_articles:
        url = article.get("originUrl", "").strip()
        if url:
            recent_urls.add(url)

    logger.info(f"최근 기사 URL {len(recent_urls)}개 수집")

    # URL 기반 필터링
    filtered_headlines = []
    duplicate_count = 0

    for headline in headlines:
        url = headline.get("url", "").strip()

        if url and url in recent_urls:
            logger.info(f"URL 중복 기사 제외: {headline.get('title', '')[:50]}")
            duplicate_count += 1
            continue

        filtered_headlines.append(headline)

    logger.info(f"URL 중복 필터링 완료: {duplicate_count}개 제거, {len(filtered_headlines)}개 남음")

    return filtered_headlines


def _select_topic_with_llm(headlines: List[Dict[str, str]], recent_article_headlines: List[str], state: BookState) -> List[SelectedTopic]:
    """LLM을 사용하여 헤드라인에서 가장 적합한 주제 3개 선정 (title과 url 포함)"""
    logger.info("LLM을 사용하여 주제 3개 선정 중...")

    # 헤드라인 텍스트 생성
    headlines_text = "\n".join([
        f"{i+1}. {h['title']}"
        for i, h in enumerate(headlines[:30])
    ])

    llm = None

    # Langfuse에서 프롬프트 가져오기 시도
    if is_langfuse_enabled():
        try:
            client = get_langfuse_client()
            label = get_prompt_label()
            langfuse_prompt = client.get_prompt("select-topic-with-llm", label=label)

            model_config = get_model_config_from_prompt(langfuse_prompt)
            llm = setup_bedrock(config=model_config)

            logger.info(f"Loaded 'select-topic-with-llm' from Langfuse (label: {label}, version: {langfuse_prompt.version})")
            prompt = convert_langfuse_to_langchain(langfuse_prompt)
        except Exception as e:
            logger.warning(f"Failed to load prompt from Langfuse, falling back to local: {e}")
            prompt = get_select_topic_prompt()
    else:
        prompt = get_select_topic_prompt()

    # LLM이 Langfuse에서 설정되지 않았으면 기본 설정 사용
    if llm is None:
        config = {
            "model": "us.meta.llama4-scout-17b-instruct-v1:0",
            "temperature": 0.3, 
            "max_tokens": 2000
        }
        llm = setup_bedrock(config=config)

    # OutputFixingParser가 clean_json_markdown을 자동으로 적용하도록 auto_clean_json=True 설정
    llm = BedrockTokenTrackingWrapper(llm, state, auto_clean_json=True)

    # 파서 설정
    base_parser = PydanticOutputParser(pydantic_object=TopicSelection)
    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=llm,  # clean_json_markdown을 자동으로 적용하는 LLM 사용
        prompt=get_output_fixing_prompt(),
        max_retries=3
    )

    country = "korea"
    if state.get("target_language_code") == "JA" :
        country = "japan"

    chain = prompt | llm
    raw_response = chain.invoke(
        {
            "headlines": headlines_text, 
            "recent_article_headlines": recent_article_headlines, 
            "format_instructions": base_parser.get_format_instructions(),
            "country": country,
            "category": state.get("tags")[0]
        })
    
    try:
        # 테스트로 fixingparser로 바로 파싱
        response = base_parser.parse(raw_response.content)
    except OutputParserException as e:
        logger.info(f"주제 선정 파싱 실패, OutputFixingParser로 복구 시도: {str(e)[:50]}...")
        response = fixing_parser.parse(raw_response.content)
        logger.info("OutputFixingParser를 통한 파싱 복구 성공")

    logger.info(f"선정된 주제 후보 {len(response.topics)}개:")
    for i, topic_title in enumerate(response.topics, 1):
        logger.info(f"  {i}. {topic_title}")

    # LLM이 반환한 제목을 기반으로 headlines에서 URL 찾기
    selected_topics = []
    for topic_title in response.topics:
        matched_url = None

        # headlines에서 topic_title이 포함된 항목 찾기
        for headline in headlines:
            headline_title = headline.get("title", "")
            if topic_title in headline_title or headline_title in topic_title:
                matched_url = headline.get("url", "")
                logger.info(f"주제 '{topic_title}'의 URL 매칭 완료: {matched_url}")
                break

        if not matched_url:
            logger.warning(f"주제 '{topic_title}'의 URL을 headlines에서 찾지 못함 - 건너뜀")
            continue

        selected_topics.append(SelectedTopic(title=topic_title, url=matched_url))

    return selected_topics

def select_topic(state: BookState) -> BookState:
    """
    LLM을 사용하여 헤드라인에서 가장 적합한 주제 3개 선정 (title과 url 포함)
    """
    try:
        logger.info("=== 주제 선정 시작 ===")

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

        # language는 대문자 ("KO", "JA", None)로 들어옴
        if language is None:
            language = "KO"

        # 1. Google News RSS로 트렌드 수집
        headlines = _fetch_trending_news_with_google_rss(category=category, language=language)

        if not headlines:
            raise ValueError(f"뉴스 API에서 트렌드를 가져오지 못했습니다. (카테고리: {category}, 언어: {language})")


        # 2.5. 백엔드에 api를 쏴서 겹치는 것들 제거
        recent_articles = _fetch_recent_articles(state)
        filtered_headlines = _filter_duplicate_headlines(headlines, recent_articles)

        # 최근 기사들의 제목만 추출
        recent_article_titles = [article.get("title", "").strip() for article in recent_articles if article.get("title")]
        logger.info(f"최근 기사 제목 {len(recent_article_titles)}개 추출")

        # 2. LLM으로 주제 3개 선정 (중요도 순)
        topic_candidates = _select_topic_with_llm(filtered_headlines, recent_article_titles, state)

        # dict로 변환하여 state에 저장
        state["topic_candidates"] = [
            {"title": t.title, "url": t.url}
            for t in topic_candidates
        ]

        logger.info(f"topic_candidates {len(topic_candidates)}개 저장 완료:")
        logger.info(f"{json.dumps(state['topic_candidates'], indent=2, ensure_ascii=False)}")

        return state

    except Exception as e:
        logger.error(f"select_topic error: {e}")
        raise