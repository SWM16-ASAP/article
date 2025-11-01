"""
기사 수집 노드: topic_candidates를 받아서 Tavily로 관련 기사 수집
"""
import os
from typing import List, Dict
from urllib.parse import urlparse
import requests
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException
from tavily import TavilyClient
from lingua import Language, LanguageDetectorBuilder

from ..state import BookState
from ..utils.logging_config import get_logger
from ..utils.workflow_helpers import setup_bedrock, BedrockTokenTrackingWrapper, send_discord_webhook
from ..prompts.outputFixingParser_prompt import get_output_fixing_prompt

logger = get_logger(__name__)

# Lingua 언어 감지기 초기화 (한국어, 일본어, 영어만 감지)
_language_detector = LanguageDetectorBuilder.from_languages(
    Language.KOREAN, Language.JAPANESE, Language.ENGLISH
).build()

# === Internal Models (이 파일 전용) ===

class TopicSummary(BaseModel):
    """주제 요약 (Diffbot 파싱용)"""
    summary: str = Field(description="200-character summary of the topic article")


def _detect_language(text: str) -> str:
    """
    Lingua를 사용하여 텍스트의 언어 감지

    Args:
        text: 언어를 감지할 텍스트

    Returns:
        언어명 문자열 ("Korean", "Japanese", "English")
    """
    # 텍스트가 너무 짧으면 샘플 확대
    sample_text = text[:1000] if len(text) > 1000 else text

    detected_language = _language_detector.detect_language_of(sample_text)

    if detected_language == Language.KOREAN:
        return "Korean"
    elif detected_language == Language.JAPANESE:
        return "Japanese"
    elif detected_language == Language.ENGLISH:
        return "English"
    else:
        # 기본값
        return "English"


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

        logger.info(f"Diffbot 본문 추출 완료 - 본문 길이: {len(text)}자")

        if not text.strip():
            raise ValueError("Diffbot 본문이 비어있습니다.")

        if len(text) < 400 :
            logger.info(f"요약 내용: {text[:100]}...")
            return text

    except Exception as e:
        logger.error(f"Diffbot으로 주제 기사 추출 실패: {e}")
        raise ValueError(f"Diffbot 파싱 실패: {str(e)}")

    # 1.5. 언어 감지
    detected_lang = _detect_language(text)
    logger.info(f"감지된 언어: {detected_lang}")

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

    # 언어별 프롬프트 매핑
    language_instruction = {
        "Korean": "Korean (한국어)",
        "Japanese": "Japanese (日本語)",
        "English": "English"
    }

    lang_name = language_instruction.get(detected_lang, "the same language as the article")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a news article summarization expert.

        When given an article text, provide a summary of the key content within 200 characters.

        **IMPORTANT: This article is written in {language}. You MUST summarize it in {language}.**

        Summary rules:
        1. Must be within 200 characters (including spaces)
        2. **MUST summarize in {language}** (never translate to another language)
        3. Include only key facts and content concisely
        4. Remove unnecessary modifiers
        5. Focus on the 5W1H principle

        Output format must be:
        {format_instructions}"""),
        ("human", """Article text: {text}

        Summarize the above article in {language} within 200 characters.""")
    ]).partial(
        format_instructions=base_parser.get_format_instructions(),
        language=lang_name
    )

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
        "include_raw_content": "markdown"
    }

    # Science, Technology는 topic="news" 설정
    if category in ["Science", "Technology"]:
        search_params["topic"] = "news"
        logger.info(f"카테고리 {category}: topic='news' 설정")

    try:
        response = tavily.search(**search_params)

        articles = response.get("results", [])

        # 제외할 도메인 목록 (정확한 도메인만 매칭)
        exclude_domains = ["youtube.com", "www.youtube.com", "x.com", "www.x.com",
                          "twitter.com", "www.twitter.com", "instagram.com", "www.instagram.com",
                          "facebook.com", "www.facebook.com"]

        # score >= min_score이면서 제외 도메인이 아닌 기사만 필터링
        filtered_articles = []
        for article in articles:
            # score 체크
            if article.get("score", 0) < min_score:
                continue

            # URL에서 도메인 추출
            url = article.get("url", "")
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.hostname  # 순수 호스트명만 추출 (www.example.com)

                if domain:
                    domain = domain.lower()
                    # 정확한 도메인 매칭 (sciencex.com은 제외되지 않음)
                    if domain in exclude_domains:
                        logger.info(f"제외 도메인 기사 필터링: {url} (도메인: {domain})")
                        continue

            except Exception as e:
                logger.warning(f"URL 파싱 실패: {url} - {e}")
                # 파싱 실패 시 포함 (안전한 선택)
                pass

            filtered_articles.append(article)

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


def fetch_articles(state: BookState) -> BookState:
    """
    state의 topic_candidates를 받아서 Tavily로 관련 기사 수집 및 파싱
    """
    try:
        logger.info("=== 기사 수집 시작 ===")

        # full_text가 이미 있으면 skip
        if state.get("full_text") and state["full_text"].strip():
            logger.info("full_text가 이미 존재합니다. 기사 수집 단계를 건너뜁니다.")
            return state

        # state에서 topic_candidates 가져오기
        topic_candidates = state.get("topic_candidates", [])

        if not topic_candidates:
            raise ValueError("topic_candidates가 없습니다. select_topic 노드가 먼저 실행되어야 합니다.")

        logger.info(f"topic_candidates {len(topic_candidates)}개 확인")

        # tags에서 카테고리 추출
        tags = state.get("tags", [])
        if not tags:
            raise ValueError("tags가 비어있습니다. 카테고리를 지정해주세요.")

        category = tags[0]

        # 각 주제에 대해 순차적으로 시도
        selected_topic_title = None
        selected_topic_url = None
        article_data = None

        for i, topic in enumerate(topic_candidates, 1):
            topic_title = topic['title']
            topic_url = topic['url']

            logger.info(f"주제 후보 {i}/{len(topic_candidates)} 시도 중: {topic_title}")

            try:
                # 주제 URL을 200자 요약으로 변환 (tavily query로 사용)
                tavily_query = _summarize_topic_with_diffbot_and_llm(topic_url, state)
                logger.info(f"Tavily 검색 쿼리 (요약): {tavily_query}")

                tavily_min_score = 0.5
                if state.get("tags")[0] == "Science":
                    tavily_min_score = 0.3

                # Tavily로 관련 기사 데이터 수집 (URL + raw_content)
                article_data = _fetch_articles_with_tavily(
                    topic=tavily_query,
                    category=category,
                    max_results=20,
                    min_score=tavily_min_score
                )

                # score 0.5 이상인 기사가 2개 이상인지 확인
                if len(article_data) >= 2:
                    logger.info(f"✅ 주제 '{topic_title}': {len(article_data)}개 기사 확보 - 사용 가능")
                    selected_topic_title = topic_title
                    selected_topic_url = topic_url
                    break
                else:
                    logger.warning(f"⚠️ 주제 '{topic_title}': {len(article_data)}개 기사만 확보 (2개 미만) - 다음 주제 시도")
                    continue

            except Exception as e:
                logger.warning(f"❌ 주제 '{topic_title}' 기사 수집 실패: {str(e)[:100]} - 다음 주제 시도")
                continue

        # 모든 주제에서 실패한 경우
        if not selected_topic_title or not article_data:
            error_msg = f"모든 주제 후보에서 충분한 기사를 찾지 못했습니다. 시도한 주제: {', '.join([t['title'] for t in topic_candidates[:3]])}"
            logger.error(error_msg)

            # 디스코드 알림
            discord_message = f"""⚠️ **기사 수집 실패** ⚠️

            **Request ID**: {state.get('id', 'N/A')}
            **카테고리**: {category}
            **언어**: {state.get('target_language_code', 'N/A')}

            **시도한 주제 후보들**:
            {chr(10).join([f"{i}. {topic['title']}" for i, topic in enumerate(topic_candidates, 1)])}

            모든 주제에서 충분한 기사를 확보하지 못했습니다."""

            send_discord_webhook(discord_message)
            raise ValueError(error_msg)

        logger.info(f"최종 선정된 주제: {selected_topic_title}")

        # 4. Diffbot으로 기사 파싱 (최소 2개, 목표 3개)
        articles_text = _parse_articles_with_diffbot(article_data, target_count=3, state=state)

        # 4.5. 기사 텍스트를 10000자로 제한
        if len(articles_text) > 10000:
            articles_text = articles_text[:10000]
            logger.info(f"기사 텍스트를 10000자로 제한함")

        # 5. state에 저장
        state["full_text"] = articles_text
        state["origin_url"] = selected_topic_url

        logger.info(f"=== 기사 수집 완료 ===")
        logger.info(f"선정된 주제: {selected_topic_title}")
        logger.info(f"수집된 기사 길이: {len(articles_text):,}자")

        return state

    except Exception as e:
        logger.error(f"fetch_articles error: {e}")
        raise
