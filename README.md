# Article Automation System

> 언어 학습 콘텐츠를 위한 AI 기반 뉴스 기사 자동 생성 시스템

## 개요

Google News RSS, Tavily Search, LangGraph를 활용한 다국어(한국어/일본어) 뉴스 기사 자동 생성 파이프라인입니다.

**월 150개 기사 자동 생성** (일 8개)
- 5개 분야: Sports, Science, Business, Culture, Technology
- 2개 언어: 한국어, 일본어 (Science/Technology는 공통)

## 주요 기능

### 1. 자동화된 워크플로우

```
카테고리 선택 → Google News RSS 크롤링 → AI 주제 선정 →
Tavily 기사 수집 → Newspaper3k 파싱 → 기사 생성
```

### 2. 핵심 특징

- **주제 자동 선정**: Google News RSS에서 최신 헤드라인 수집 후 LLM이 최적 주제 선택
- **스마트 필터링**: 가십/정치 뉴스 자동 제외, 분야별 품질 기준 적용
- **자동 크롤링**: Tavily로 관련 기사 2-3개 수집 및 본문 파싱
- **다국어 지원**: 한국/일본 타겟 뉴스 자동 생성

## 기술 스택

- **Python 3.11+**
- **LangGraph**: 워크플로우 오케스트레이션
- **Tavily Search API**: 뉴스 검색 및 본문 추출
- **Newspaper3k**: HTML 파싱
- **MongoDB Atlas**: 데이터 저장
- **AWS S3**: 결과 파일 저장
- **GitHub Actions**: 스케줄 기반 자동 실행

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirement.txt
```

### 2. 환경변수 설정

`.env` 파일 생성:

```bash
# AWS & MongoDB
MONGO_URI_SECRET_NAME=your-secret-name
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret

# APIs
TAVILY_API_KEY=your-tavily-key
LANGSMITH_API_KEY=your-langsmith-key

# Optional
DISCORD_WEBHOOK_URL=your-webhook-url
```

### 3. 로컬 실행

```bash
export ARTICLE_CATEGORY=Sports
export ARTICLE_LANGUAGE=ko
export OUTPUT_S3_BUCKET=your-bucket

python novel_transformer_project_article/main.py
```

## GitHub Actions 자동화

매일 UTC 00:00 (KST 09:00)에 8개 기사 자동 생성:

- Business (ko, ja)
- Sports (ko, ja)
- Culture (ko, ja)
- Technology (common)
- Science (common)

### Secrets 설정 필요

Repository Settings → Secrets and variables → Actions:

```
MONGO_URI_SECRET_NAME
AWS_REGION
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
OUTPUT_S3_BUCKET
TAVILY_API_KEY
LANGSMITH_API_KEY
DISCORD_WEBHOOK_URL (선택)
```

## 프로젝트 구조

```
novel_transformer_project_article/
├── main.py                          # 메인 실행 파일
├── state.py                         # LangGraph State 정의
├── nodes/
│   ├── fetch_topic_and_articles_node.py  # 주제/기사 수집
│   ├── generate_article_node.py     # 기사 생성
│   └── ...
├── prompts/                         # LLM 프롬프트
└── utils/                           # 헬퍼 함수들

.github/
└── workflows/
    └── article-automation.yml       # GitHub Actions 워크플로우
```

## 워크플로우 상세

1. **주제 선정**: Google News RSS에서 카테고리별 헤드라인 수집
2. **AI 필터링**: LLM이 가장 적합한 주제 선택 (가십/정치 제외)
3. **기사 수집**: Tavily로 관련 기사 최대 20개 검색 (점수 0.5+)
4. **본문 파싱**: Newspaper3k로 상위 3개 기사 본문 추출
5. **콘텐츠 생성**: 수집된 기사 기반 새 콘텐츠 생성

## 비용 효율성

- **시간 절감**: 월 100시간 → 25분 (99.6% 감소)
- **API 비용**: Tavily 무료 플랜 (월 1,000 크레딧) 내 운영 가능
- **인프라**: GitHub Actions 무료 플랜 활용

## 라이선스

MIT

## 문서

자세한 개발 과정은 [docs/automation.md](docs/automation.md) 참고
