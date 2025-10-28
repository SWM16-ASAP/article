"""
Google News RSS로 트렌드 수집 → LLM으로 주제 선정 → Tavily로 기사 수집하는 노드
"""
import os
import json
import requests
from re import T
from typing import List, Dict
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException
import feedparser
from tavily import TavilyClient
from newspaper import Article
from ..state import BookState
from ..utils.logging_config import get_logger
from ..utils.workflow_helpers import setup_bedrock, BedrockTokenTrackingWrapper, send_discord_webhook
from ..utils.langfuse_client import (
    get_langfuse_client,
    is_langfuse_enabled,
    get_prompt_label,
    convert_langfuse_to_langchain,
    get_model_config_from_prompt
)
from ..prompts.select_topic_with_llm_prompt import get_select_topic_prompt
from ..prompts.outputFixingParser_prompt import get_output_fixing_prompt


logger = get_logger(__name__)

# Pydantic 모델: LLM이 선정한 주제들

class TopicSelection(BaseModel):
    topics: List[str] = Field(description="List of 3 topic titles only in order of relevance and interest (most relevant first)")

# Pydantic 모델: 선정된 주제 (제목 + URL)
class SelectedTopic(BaseModel):
    title: str = Field(description="Topic title")
    url: str = Field(description="Topic URL")

# Pydantic 모델: LLM이 추출한 기사 정보
class ArticleExtraction(BaseModel):
    content: str = Field(description="Article main content")

# Pydantic 모델: LLM이 생성한 주제 요약
class TopicSummary(BaseModel):
    summary: str = Field(description="200-character summary of the topic article")


def _fetch_trending_news_with_google_rss(category: str, language: str = "KO") -> List[Dict[str, str]]:
    """Google News RSS로 최신 뉴스 헤드라인 수집"""

    # 카테고리 및 언어별로 미리 정의된 RSS URL
    RSS_URLS = {
        "Sports": {
            "KO": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtdHZHZ0pMVWlnQVAB?hl=ko&gl=KR&ceid=KR:ko",
            "JA": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtdHZHZ0pMVWlnQVAB?hl=ja&gl=JP&ceid=JP:ja"
        },
        "Science": {
            "KO": "https://www.sciencenews.org/feed",
            "JA": "https://www.sciencenews.org/feed"
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

def _get_recent_article_headlines(state: BookState) -> List[str]:
    """
    백엔드를 통해 최근 기사들을 가져오기

    Returns:
        List[str]: 백엔드에서 가져온 최근 기사 제목 리스트
    """

    # 백엔드 API
    swagger_api_key = os.getenv("SWAPPER_API_KEY")
    backend_url = os.getenv("READ_PAST_ARTICLE_URL")

    targetLanguageCode = state.get("target_language_code")

    # None이면 기본값 "KO" 설정
    if targetLanguageCode is None:
        targetLanguageCode = "KO"

    # 이미 대문자로 들어오므로 upper() 불필요
    # targetLanguageCode는 "KO", "JA" 등

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

        # 백엔드 기사의 title 목록 추출
        recent_article_titles = []
        for article in article_data:
            title = article.get("title", "")
            if title:
                recent_article_titles.append(title.strip())

        logger.info(f"백엔드에서 {len(recent_article_titles)}개 기사 제목 추출")

        return recent_article_titles

    except requests.exceptions.Timeout:
        logger.error("_get_recent_article_headlines error - 백엔드 API 타임아웃 - 빈 리스트 반환 ")
        return []  # 에러 시 빈 제목 리스트
    except Exception as e:
        logger.error(f"_get_recent_article_headlines error {e} - 빈 리스트 반환")
        return []  # 에러 시 빈 제목 리스트


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

    broken_content = raw_response.content.replace('{', '{{BROKEN')
    
    try:
        # 테스트로 fixingparser로 바로 파싱
        response = base_parser.parse(broken_content)
    except OutputParserException as e:
        logger.info(f"주제 선정 파싱 실패, OutputFixingParser로 복구 시도: {str(e)[:50]}...")
        response = fixing_parser.parse(broken_content)
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


def _fetch_articles_with_tavily(topic: str, category: str = None, max_results: int = 20, min_score: float = 0.3) -> List[Dict[str, str]]:
    """
    Tavily로 주제 관련 기사 링크 및 raw_content 수집

    Args:
        topic: 검색할 주제
        category: 카테고리 (Science, Technology 등) - tavily 설정 조정용
        max_results: 최대 검색 결과 수 (기본값: 20)
        min_score: 최소 점수 (기본값: 0.3, 이 점수 이상인 결과만 반환)

    Returns:
        score가 min_score 이상인 기사 정보 리스트 (score 내림차순 정렬)
        각 항목: {"url": str, "raw_content": str, "score": float, "title": str}
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY 환경변수가 설정되지 않았습니다.")

    tavily = TavilyClient(api_key=tavily_key)

    logger.info(f"Tavily로 '{topic}' 관련 기사 {max_results}개 검색 중 (카테고리: {category}, 최소 점수: {min_score})...")

    # 검색 파라미터 기본 설정
    search_params = {
        "query": topic,
        "max_results": max_results,
        "include_raw_content": "markdown",
        "exclude_domains": ["youtube.com","x.com","instagram.com","facebook.com"]
    }

    # Science, Technology는 topic="news" 설정
    if category in ["Science", "Technology"]:
        search_params["topic"] = "news"
        logger.info(f"카테고리 {category}: topic='news' 설정")

    # Science가 아닌 경우에만 날짜 제한 추가
    if category != "Science":
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        start_date = week_ago.strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        search_params["start_date"] = start_date
        search_params["end_date"] = end_date
        logger.info(f"검색 기간: {start_date} ~ {end_date}")
    else:
        logger.info(f"카테고리 Science: 날짜 제한 없음")

    try:
        response = tavily.search(**search_params)

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

        # 중복 제거를 위한 제목 추적 set
        seen_titles = set()
        article_data = []

        for article in sorted_articles:
            # URL과 raw_content가 있는지 확인
            if not article.get("url") or not article.get("raw_content"):
                continue

            title = article.get("title", "").strip()

            # 제목에서 언론사 부분 제거 (예: "제목 - 연합뉴스" -> "제목")
            # 마지막 " - " 이후를 언론사로 간주하고 제거
            title_without_source = title.rsplit(" - ", 1)[0].strip() if " - " in title else title

            # 제목이 비어있거나 이미 본 제목이면 건너뛰기
            if not title_without_source or title_without_source in seen_titles:
                logger.info(f"중복 기사 제외: {title[:50]}")
                continue

            # 새로운 제목이면 추가
            seen_titles.add(title_without_source)
            article_data.append({
                "url": article.get("url", ""),
                "raw_content": article.get("raw_content", ""),
                "score": article.get("score", 0),
                "title": title  # 원본 제목은 그대로 저장
            })

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
    llm = BedrockTokenTrackingWrapper(llm, state, auto_clean_json=True)

    # 파서 설정
    base_parser = PydanticOutputParser(pydantic_object=ArticleExtraction)
    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=llm,
        prompt=get_output_fixing_prompt(),
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


def _parse_articles_with_diffbot(article_data: List[Dict[str, str]], target_count: int = 3, state: BookState = None) -> str:
    """
    Diffbot API를 사용하여 기사 URL에서 제목과 본문 파싱

    Args:
        article_data: Tavily에서 가져온 기사 데이터 리스트
                     각 항목: {"url": str, "raw_content": str, "score": float, "title": str}
        target_count: 목표 기사 개수 (기본값: 3)
        state: BookState (에러 알림용)

    Returns:
        파싱된 기사들을 결합한 텍스트 (포맷: === 기사 제목 ===\n본문)

    Raises:
        ValueError: DIFFBOT_API_TOKEN이 없거나 파싱된 기사가 1개 이하인 경우
    """
    logger.info(f"Diffbot API로 기사 파싱 시작 (목표: {target_count}개)")

    # Diffbot API 토큰 확인
    diffbot_token = os.getenv("DIFFBOT_API_TOKEN")
    if not diffbot_token:
        raise ValueError("DIFFBOT_API_TOKEN 환경변수가 설정되지 않았습니다.")

    parsed_articles = []
    failed_count = 0

    for i, article in enumerate(article_data):
        # 목표 개수 달성하면 종료
        if len(parsed_articles) >= target_count:
            logger.info(f"목표 기사 {target_count}개 파싱 완료")
            break

        try:
            url = article.get("url", "")
            if not url:
                logger.warning(f"  ⚠️ URL이 없는 기사 데이터 건너뜀")
                failed_count += 1
                continue

            logger.info(f"[{i+1}/{len(article_data)}] Diffbot 파싱 시도: {url}")

            # Diffbot API 호출
            api_url = "https://api.diffbot.com/v3/article"

            headers = {"accept": "application/json"}

            params = {
                'url': url,
                'token': diffbot_token
            }

            response = requests.get(api_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # objects 배열에서 첫 번째 기사 추출
            if not data.get('objects') or len(data['objects']) == 0:
                logger.warning(f"  ⚠️ Diffbot 응답에 objects가 없음")
                failed_count += 1
                continue

            diffbot_article = data['objects'][0]

            # 제목과 본문 추출
            title = diffbot_article.get('title', '').strip()
            text = diffbot_article.get('text', '').strip()

            # 제목과 본문 검증
            if not title or not text or len(text) < 50:
                logger.warning(f"  ⚠️ 제목 또는 본문이 부족함 - 제목: '{title[:50]}', 본문 길이: {len(text)}자")
                failed_count += 1
                continue

            # 성공적으로 파싱됨
            parsed_articles.append({
                "title": title,
                "content": text
            })
            logger.info(f"  ✅ 파싱 성공 - 제목: {title[:50]}..., 본문 길이: {len(text)}자")

        except requests.exceptions.Timeout:
            logger.warning(f"  ❌ Diffbot API 타임아웃: {url}")
            failed_count += 1
            continue
        except requests.exceptions.RequestException as e:
            logger.warning(f"  ❌ Diffbot API 요청 실패: {url} - 오류: {str(e)[:100]}")
            failed_count += 1
            continue
        except Exception as e:
            logger.warning(f"  ❌ 파싱 실패: {url} - 오류: {str(e)[:100]}")
            failed_count += 1
            continue

    # 결과 확인
    parsed_count = len(parsed_articles)
    logger.info(f"Diffbot 파싱 완료: 성공 {parsed_count}개, 실패 {failed_count}개")

    # 1개 이하면 에러
    if parsed_count <= 1:
        error_msg = f"기사 파싱 실패: {parsed_count}개만 성공 (최소 2개 필요). 주제: {state.get('topic', 'N/A') if state else 'N/A'}"
        logger.error(error_msg)

        # 디스코드 알림
        discord_message = f"""⚠️ **기사 파싱 실패 (Diffbot)** ⚠️

        **Request ID**: {state.get('id', 'N/A') if state else 'N/A'}
        **주제**: {state.get('topic', 'N/A') if state else 'N/A'}
        **카테고리**: {state.get('tags', ['N/A'])[0] if state and state.get('tags') else 'N/A'}
        **언어**: {state.get('language', 'N/A') if state else 'N/A'}

        **파싱 결과**: {parsed_count}개 성공 (최소 2개 필요)
        **시도한 URL 개수**: {len(article_data)}개
        **실패한 URL 개수**: {failed_count}개

        Diffbot API로 기사를 파싱했으나 충분한 기사를 확보하지 못했습니다."""

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


def _summarize_topic_with_diffbot_and_llm(topic_url: str, state: BookState = None) -> str:
    """
    Diffbot으로 주제 기사의 본문을 추출하고 LLM으로 200자 요약

    Args:
        topic_url: 요약할 주제 기사의 URL
        state: BookState (토큰 추적용)

    Returns:
        200자 이내의 요약 텍스트

    Raises:
        ValueError: Diffbot 파싱 실패 또는 요약 생성 실패
    """
    logger.info(f"주제 URL 요약 시작: {topic_url}")

    # 1. Diffbot으로 기사 본문 추출
    diffbot_token = os.getenv("DIFFBOT_API_TOKEN")
    if not diffbot_token:
        raise ValueError("DIFFBOT_API_TOKEN 환경변수가 설정되지 않았습니다.")

    try:
        logger.info("Diffbot으로 주제 기사 본문 추출 중...")
        api_url = "https://api.diffbot.com/v3/article"
        headers = {"accept": "application/json"}
        params = {'url': topic_url, 'token': diffbot_token}

        response = requests.get(api_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if not data.get('objects') or len(data['objects']) == 0:
            raise ValueError("Diffbot 응답에 objects가 없습니다.")

        diffbot_article = data['objects'][0]
        text = diffbot_article.get('text', '').strip()

        if len(text) < 400 :
            return text

        logger.info(f"Diffbot 본문 추출 완료 - 본문 길이: {len(text)}자")

    except Exception as e:
        logger.error(f"Diffbot으로 주제 기사 추출 실패: {e}")
        raise ValueError(f"Diffbot 파싱 실패: {str(e)}")

    # 2. LLM으로 200자 요약 (최대 3번 재시도)
    config = {
        "model": "us.meta.llama4-scout-17b-instruct-v1:0",
        "temperature": 0.3,
        "max_tokens": 500
    }
    llm = setup_bedrock(config=config)
    llm = BedrockTokenTrackingWrapper(llm, state, auto_clean_json=True)

    base_parser = PydanticOutputParser(pydantic_object=TopicSummary)
    fixing_parser = OutputFixingParser.from_llm(
        parser=base_parser,
        llm=llm,
        prompt=get_output_fixing_prompt(),
        max_retries=1
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """너는 뉴스 기사 요약 전문가야.

        사용자가 기사 제목과 본문을 주면, 핵심 내용을 200자 이내로 요약해서 제공해.

        요약 규칙:
        1. 반드시 200자 이내로 작성 (공백 포함)
        2. 핵심 사실과 내용만 간결하게
        3. 불필요한 수식어 제거
        4. 육하원칙 위주로 작성

        구조는 아래와 같이 줘야 해:
        {format_instructions}"""),
        ("human", """본문: {text}

        위 기사를 200자 이내로 요약해줘.""")
    ]).partial(format_instructions=base_parser.get_format_instructions())

    chain = prompt | llm

    max_retries = 3
    current_text = text[:5000] if len(text) > 5000 else text  # 초기 텍스트

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"LLM 요약 시도 {attempt}/{max_retries}... (입력 텍스트 길이: {len(current_text)}자)")

            raw_response = chain.invoke({"text": current_text})

            try:
                result = base_parser.parse(raw_response.content)
            except OutputParserException as e:
                logger.info(f"파싱 실패, OutputFixingParser로 복구 시도: {str(e)[:50]}...")
                result = fixing_parser.parse(raw_response.content)
                logger.info("OutputFixingParser를 통한 파싱 복구 성공")

            summary = result.summary.strip()
            summary_length = len(summary)

            logger.info(f"요약 생성 완료 - 길이: {summary_length}자")
            logger.info(f"요약 내용: {summary[:100]}...")

            # 200자 이내면 성공
            if summary_length <= 400:
                logger.info(f"✅ 400자 이내 요약 생성 성공 ({summary_length}자)")
                return summary

            # 400자 초과면 재시도
            else:
                logger.warning(f"요약이 400자를 초과함 ({summary_length}자) - 재시도")
                # 요약이 기존 텍스트보다 짧으면 이걸로 재시도
                if summary_length < len(current_text):
                    logger.info(f"요약된 텍스트({summary_length}자)를 입력으로 재요약 시도")
                    current_text = summary
                continue

        except Exception as e:
            logger.warning(f"LLM 요약 시도 {attempt} 실패: {str(e)[:100]}")
            if attempt == max_retries:
                raise ValueError(f"LLM 요약 생성 실패 (최대 재시도 횟수 초과): {str(e)}")
            continue

    # 모든 재시도 실패
    raise ValueError("200자 이내 요약 생성에 실패했습니다 (최대 재시도 횟수 초과)")


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

        # language는 대문자 ("KO", "JA", None)로 들어옴
        if language is None:
            language = "KO"

        # 1. Google News RSS로 트렌드 수집
        headlines = _fetch_trending_news_with_google_rss(category=category, language=language)

        if not headlines:
            raise ValueError(f"뉴스 API에서 트렌드를 가져오지 못했습니다. (카테고리: {category}, 언어: {language})")


        # 2.5. 백엔드에 api를 쏴서 겹치는 것들 제거
        recent_article_headlines = _get_recent_article_headlines(state)

        # 2. LLM으로 주제 3개 선정 (중요도 순)
        topic_candidates = _select_topic_with_llm(headlines, recent_article_headlines, state)

        logger.info(f"topic_candidates:\n{json.dumps([t.model_dump() for t in topic_candidates], indent=2, ensure_ascii=False)}")

        # 3. 각 주제에 대해 순차적으로 시도
        selected_topic = None
        selected_topic_url = None
        article_data = None

        for i, topic in enumerate(topic_candidates, 1):
            logger.info(f"주제 후보 {i}/{len(topic_candidates)} 시도 중: {topic}")

            try:
                # 주제 URL을 200자 요약으로 변환 (tavily query로 사용)
                tavily_query = _summarize_topic_with_diffbot_and_llm(topic.url, state)
                logger.info(f"Tavily 검색 쿼리 (요약): {tavily_query}")

                # Tavily로 관련 기사 데이터 수집 (URL + raw_content)
                article_data = _fetch_articles_with_tavily(
                    topic=tavily_query,
                    category=category,
                    max_results=20,
                    min_score=0.3
                )

                # score 0.5 이상인 기사가 5개 이상인지 확인
                if len(article_data) >= 2:
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

            모든 주제에서 score 0.3 이상 기사 5개 이상을 확보하지 못했습니다."""

            send_discord_webhook(discord_message)
            raise ValueError(error_msg)

        logger.info(f"최종 선정된 주제: {selected_topic.title}")

        # 4. LLM으로 기사 파싱 (최소 2개, 목표 3개)
        articles_text = _parse_articles_with_diffbot(article_data, target_count=3, state=state)

        # 4.5. 기사 텍스트를 10000자로 제한
        if len(articles_text) > 10000:
            articles_text = articles_text[:10000]
            logger.info(f"기사 텍스트를 10000자로 제한함")

        # 5. state에 저장
        state["full_text"] = articles_text
        state["origin_url"] = selected_topic_url

        logger.info(f"=== 주제 선정 및 기사 수집 완료 ===")
        logger.info(f"선정된 주제: {selected_topic.title}")
        logger.info(f"수집된 기사 길이: {len(articles_text):,}자")

        return state

    except Exception as e:
        logger.error(f"주제 선정 및 기사 수집 중 오류: {e}")
        raise
