# Ling-Level Article Automation

> 언어 학습을 위한 AI 기반 뉴스 기사 자동 생성 및 CEFR 레벨링 시스템

## 개요

Google News RSS, Tavily Search, Diffbot, LangGraph를 활용한 완전 자동화 기사 생성 파이프라인입니다. 원본 기사를 수집하여 C1 레벨 영문 기사를 생성한 후, **7개의 CEFR 레벨(A0-C2)로 자동 변환**합니다.

**월 150개 기사 자동 생성** (일 8개)
- 5개 분야: Sports, Science, Business, Culture, Technology
- 2개 타겟 언어: 한국어, 일본어 (Science/Technology는 공통)
- 7개 CEFR 레벨: A0, A1, A2, B1, B2, C1, C2

## 주요 기능

### 1. 완전 자동화된 엔드-투-엔드 파이프라인

```
카테고리 선택 → Google News RSS 헤드라인 수집 → AI 주제 선정 →
Tavily 기사 검색 → Diffbot 본문 파싱 → C1 기사 생성 →
커버 이미지 생성 → 7개 CEFR 레벨 병렬 변환 → S3 저장 → 백엔드 등록
```

### 2. 핵심 특징

- **AI 주제 선정**: Google News RSS에서 30개 헤드라인 수집 후 LLM이 3개 후보 선택
- **스마트 필터링**: 가십/정치 뉴스 자동 제외, 중복 방지, 분야별 품질 기준 적용
- **고품질 크롤링**: Tavily로 최대 20개 관련 기사 검색 → Diffbot으로 상위 3개 본문 파싱
- **다국어 지원**: 한국/일본 타겟 시장 자동 구분 (Culture는 언어 무관)
- **CEFR 자동 레벨링**: 6개 레벨 병렬 처리 (A0-B2, C2), 원본은 C1 유지
- **품질 관리**: Judge AI가 각 레벨 평가 (80점 이상 필수), 피드백 기반 재생성
- **AI 커버 이미지**: GPT-5-image-mini로 1:1 비율 이미지 자동 생성
- **비용 추적**: 실시간 토큰 사용량 및 비용 계산, Discord 알림

## 기술 스택

### Core Framework
- **Python 3.12**: 메인 언어
- **LangGraph 0.2+**: 상태 기반 워크플로우 오케스트레이션
- **LangChain 0.3.27**: LLM 통합 및 프롬프트 관리

### AI Models & APIs
- **AWS Bedrock**:
  - `Llama3-2-90B`: C1 기사 생성
  - `Llama4-Scout-17B`: CEFR 레벨링 (A0-C2)
  - `Llama4-Maverick-17B`: 기사 길이 조정
  - `Amazon Nova Lite`: 품질 평가 (Judge AI)
- **OpenRouter**:
  - `GPT-5-image-mini`: 커버 이미지 생성

### Content Collection
- **Google News RSS**: 카테고리별 헤드라인 수집
- **Tavily Search API**: 관련 기사 검색 (최대 20개)
- **Diffbot API**: 웹 페이지 본문 파싱 및 요약

### Language Processing
- **pysbd 0.3.4**: 문장 분할
- **Lingua**: 언어 감지 (한국어, 영어, 일본어)

### Infrastructure
- **AWS S3**: 기사 메타데이터 및 이미지 저장
- **MongoDB Atlas**: 중복 방지를 위한 기사 이력 관리
- **GitHub Actions**: 매일 자동 실행 (UTC 23:00)

### Monitoring & Management
- **Langfuse 3.5+**: 프롬프트 버전 관리
- **LangSmith**: 워크플로우 추적
- **Discord Webhooks**: 성공/실패/비용 알림

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirement.txt
```

### 2. 환경변수 설정

`.env` 파일 생성:

```bash
# AWS Credentials
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
OUTPUT_S3_BUCKET=your-bucket-name

# Content Collection APIs
TAVILY_API_KEY=your-tavily-key
DIFFBOT_API_TOKEN=your-diffbot-token
OPENROUTER_API_KEY=your-openrouter-key

# Backend Integration
SWAPPER_API_KEY=your-backend-api-key
READ_PAST_ARTICLE_URL=https://your-backend/api/articles/recent
ADD_NEW_ARTICLE_URL=https://your-backend/api/articles
ALARM_API_URL=https://your-backend/api/alarm

# Langfuse (Prompt Management)
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
USE_LANGFUSE=true

# Monitoring (Optional)
LANGSMITH_API_KEY=your-langsmith-key
DISCORD_WEBHOOK_URL=your-discord-webhook

# Execution Config
ARTICLE_CATEGORY=Sports  # Sports, Science, Business, Culture, Technology
ARTICLE_LANGUAGE=ko      # ko, ja, or null
```

### 3. 로컬 실행

```bash
export ARTICLE_CATEGORY=Sports
export ARTICLE_LANGUAGE=ko
export OUTPUT_S3_BUCKET=your-bucket

python novel_transformer_project_article/main.py
```

## GitHub Actions 자동화

매일 UTC 23:00 (KST 익일 08:00)에 8개 기사 자동 생성:

- Business (ko, ja)
- Sports (ko, ja)
- Culture (ko, ja) - 언어 코드 null로 처리
- Technology (null) - 공통 컨텐츠
- Science (null) - 공통 컨텐츠

### 워크플로우 구조

**Job 1: `generate-articles`**
- Matrix 전략으로 8개 기사 병렬 생성 (max-parallel: 1, fail-fast: false)
- 각 작업은 독립적으로 실행되며 실패해도 다른 작업에 영향 없음
- 결과는 `article_result.json` 아티팩트로 저장 (1일 보관)

**Job 2: `notify-completion`**
- 모든 기사 생성 완료 후 실행 (runs always)
- 8개 아티팩트를 수집하여 백엔드에 일괄 알림
- 동일 articleId는 targetLanguageCodes 배열로 그룹화

### Secrets 설정 필요

Repository Settings → Secrets and variables → Actions:

```bash
# AWS (3개)
AWS_REGION
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
OUTPUT_S3_BUCKET

# APIs (5개)
TAVILY_API_KEY
DIFFBOT_API_TOKEN
OPENROUTER_API_KEY
LANGSMITH_API_KEY
DISCORD_WEBHOOK_URL

# Backend (3개)
SWAPPER_API_KEY
READ_PAST_ARTICLE_URL
ADD_NEW_ARTICLE_URL
ALARM_API_URL

# Langfuse (3개)
LANGFUSE_PUBLIC_KEY
LANGFUSE_SECRET_KEY
LANGFUSE_HOST
```

## 프로젝트 구조

```
ling-level-article-automation/
├── .github/
│   └── workflows/
│       └── article-automation.yml       # GitHub Actions 워크플로우
├── docs/
│   ├── automation.md                    # 개발 과정 문서
│   ├── automationFinish.md              # 완료 문서
│   └── relevant_news.md                 # 뉴스 관련 문서
├── novel_transformer_project_article/   # 메인 애플리케이션
│   ├── main.py                          # 진입점 및 워크플로우 오케스트레이션
│   ├── state.py                         # LangGraph 상태 정의
│   ├── nodes/                           # 워크플로우 노드 구현
│   │   ├── fetch_topic_and_articles_node.py    # 주제 선정 & 기사 수집
│   │   ├── generate_article_node.py            # C1 기사 생성
│   │   ├── generate_cover_image_node.py        # 커버 이미지 생성
│   │   ├── split_nodes.py                      # 텍스트 청킹
│   │   ├── parallel_cefr_processing_node.py    # 병렬 CEFR 변환
│   │   ├── generate_level_specific_text_node.py # 개별 레벨 텍스트 생성
│   │   ├── navigation_nodes.py                 # 워크플로우 네비게이션
│   │   ├── conditional_nodes.py                # 조건부 라우팅
│   │   └── final_summary_log_node.py           # 최종 메트릭 로깅
│   ├── prompts/                         # LLM 프롬프트 템플릿
│   │   ├── generate_article_prompt.py           # 기사 생성 프롬프트
│   │   ├── select_topic_with_llm_prompt.py      # 주제 선정 프롬프트
│   │   ├── generate_cover_image_prompt.py       # 이미지 생성 프롬프트
│   │   ├── outputFixingParser_prompt.py         # JSON 파싱 수정 프롬프트
│   │   ├── leveling/                            # CEFR 레벨별 프롬프트
│   │   │   ├── a0_prompt.py ~ c2_prompt.py     # 7개 레벨 프롬프트
│   │   └── judging/                             # 품질 평가 프롬프트
│   │       └── a0_judge_prompt.py ~ c2_judge_prompt.py  # 7개 평가 프롬프트
│   └── utils/                           # 유틸리티 모듈
│       ├── workflow_helpers.py                  # Bedrock 설정, 토큰 추적, 알림
│       ├── langfuse_client.py                   # Langfuse 프롬프트 버전 관리
│       ├── judge_ai.py                          # 품질 평가 시스템
│       ├── logging_config.py                    # 로깅 설정
│       ├── data_models.py                       # 데이터 구조 정의
│       ├── json_handler.py                      # JSON 처리 유틸
│       ├── pysbd_handler.py                     # 문장 분할
│       ├── exceptions.py                        # 커스텀 예외
│       └── handlers/
│           └── article_handler.py               # 기사 출력 & S3 저장
├── script/                              # 유틸리티 스크립트
│   ├── upload_select_topic_prompt.py           # Langfuse 프롬프트 업로드
│   └── langfuse_example.py                     # Langfuse 사용 예제
├── test/                                # 테스트 파일
├── .env                                 # 환경 변수 (로컬)
├── .gitignore                           # Git 무시 규칙
├── langgraph.json                       # LangGraph 설정
├── requirements.txt                     # Python 의존성
└── README.md                            # 프로젝트 문서
```

## 워크플로우 상세

### LangGraph 워크플로우 (8단계)

```
START
  ↓
1. fetch_topic_and_articles
   - Google News RSS에서 30개 헤드라인 수집
   - 백엔드에서 최근 기사 제목 가져와 중복 체크
   - LLM이 3개 후보 주제 선정 (가십/정치 제외)
   - 각 후보마다:
     * Diffbot으로 주제 URL 요약 (200자)
     * Tavily로 관련 기사 최대 20개 검색 (최소 점수 0.3)
     * 상위 3개 기사를 Diffbot으로 본문 파싱
   - 최소 2개 기사 확보될 때까지 후보 순회
  ↓
2. generate_article
   - 수집된 기사를 기반으로 C1 레벨 영문 기사 생성
   - 제목: 60자 목표, 80자 최대 (A1 레벨)
   - 본문: 1000-1500자 목표, 900-1600자 허용 (C1 레벨)
   - Lingua로 영어 검증, 길이 조건 미달 시 최대 3회 재조정
   - target_language_code는 Culture 제외하고 null 처리
  ↓
3. generate_cover_image
   - GPT-5-image-mini로 1:1 비율 커버 이미지 생성
   - 예제 이미지 스타일 참조
   - base64 형식으로 상태에 저장
  ↓
4. split_chapter_into_chunks
   - 커버 이미지 태그 추가: [illustration: {json_data}]
   - 특수 태그로 청킹: [illustration:...], <Preserve>...</Preserve>
   - 일러스트레이션과 보존 구간은 별도 청크로 분리
  ↓
5. create_all_cefr_versions_parallel
   - C1 원본: pysbd로 문장 분할
   - 6개 레벨(A0-B2, C2): ThreadPoolExecutor로 병렬 생성
   - 각 레벨마다:
     * 레벨별 프롬프트로 텍스트 생성
     * Judge AI가 품질 평가 (0-100점)
     * 80점 미만 시 피드백 기반으로 최대 5회 재생성
     * 생성된 텍스트를 문장 단위로 분할
   - 일러스트레이션/보존 청크는 모든 레벨에 복사
  ↓
6. move_to_next_chunk (조건부)
   - 현재 챕터에 더 많은 청크가 있으면 5번으로 복귀
   - 없으면 7번으로 진행
  ↓
7. move_to_next_chapter (조건부)
   - 더 많은 챕터가 있으면 4번으로 복귀
   - 없으면 8번으로 진행
  ↓
8. log_final_summary
   - 모델별 토큰 사용량 로깅
   - 총 비용 계산
   - 비용이 $0.04 초과 시 Discord 알림
  ↓
END
```

### 품질 관리 시스템

**Judge AI 평가 기준** (각 25점):
1. **Vocabulary Usage**: CEFR 레벨에 맞는 어휘 사용
2. **Grammar & Sentence Structure**: 적절한 문법 및 문장 구조
3. **Content Preservation**: 원본 내용 유지
4. **Context Continuity**: 문맥 연속성

**재생성 로직**:
- 1차 생성 → Judge AI 평가 → 80점 미만 시 피드백 제공 → 재생성
- 최대 5회 시도, 모두 실패 시 가장 높은 점수의 결과 사용
- 각 레벨별 동적 temperature (A0/A1: 0.3, A2-B1: 0.6, B2-C2: 0.8)

## 비용 효율성

### 시간 절약
- **수동 작업**: 월 100시간 (기사당 40분)
- **자동화 후**: 월 25분 (모니터링만)
- **효율성**: 99.6% 시간 절감

### API 비용 (월 150개 기사 기준)

| 항목 | 모델/서비스 | 단가 | 월 사용량 | 월 비용 |
|------|------------|------|----------|---------|
| 주제 선정 | Llama3-2-90B | $0.72/M 토큰 | ~700K | $0.50 |
| 기사 생성 | Llama3-2-90B | $0.72/M 토큰 | ~2.8M | $2.00 |
| 길이 조정 | Llama4-Maverick-17B | $0.66/M 토큰 | ~500K | $0.33 |
| CEFR 레벨링 | Llama4-Scout-17B | $0.17/M 토큰 | ~15M | $2.55 |
| 품질 평가 | Nova Lite | $0.06/M 토큰 | ~10M | $0.60 |
| 커버 이미지 | GPT-5-image-mini | $0.013/이미지 | 150 | $1.95 |
| 기사 검색 | Tavily Search | 월 1,000 크레딧 | 900 크레딧 | $0 (무료) |
| 본문 파싱 | Diffbot | 월 10,000 요청 | 900 요청 | $0 (무료) |

**총 월 비용**: **약 $7.93** (기사당 $0.053)

### 인프라 비용
- **GitHub Actions**: 무료 플랜 (월 2,000분 제공, 사용량 ~240분)
- **AWS S3**: 월 $0.50 미만 (메타데이터 150개 + 이미지 150개)
- **MongoDB Atlas**: 무료 Tier (기사 이력 관리)

### 총 운영 비용
**월 $8.50 미만** (인건비 제외 시)

vs 수동 작업 시 100시간 × 시급 = 대폭 절감

## 주요 특징 상세

### 1. Langfuse 프롬프트 버전 관리
- 모든 프롬프트를 Langfuse에서 중앙 관리
- 코드 배포 없이 프롬프트 실시간 업데이트 가능
- 환경별 프롬프트 분리 (production/staging/development)
- A/B 테스트 지원
- 로컬 fallback: Langfuse 장애 시 로컬 파일 사용

### 2. 실시간 비용 추적
- 모델별 토큰 사용량 자동 계산
- 실시간 비용 집계 (입력/출력 토큰 구분)
- 비용 초과 시 Discord 알림 ($0.04 임계값)
- 최종 요약 로그에 상세 비용 분석 포함

### 3. 중복 방지 시스템
- 백엔드 API를 통해 최근 30일 기사 제목 조회
- LLM 주제 선정 시 중복 제목 자동 제외
- MongoDB에 생성 이력 저장

### 4. 다단계 오류 처리
- API 호출 실패 시 자동 재시도 (최대 3-5회)
- 품질 미달 시 피드백 기반 재생성
- 최종 실패 시 Discord 알림 + GitHub Actions 실패 표시
- 부분 실패 허용 (fail-fast: false)

### 5. S3 결과 저장 구조
```
s3://bucket/article/{articleId}/
├── metadata.json          # 기사 메타데이터 (7개 레벨 모두 포함)
└── images/
    └── cover.jpg          # 커버 이미지
```

**metadata.json 구조**:
```json
{
  "id": "article-20250128-...",
  "content_type": "article",
  "title": "Article Title",
  "author": "auto-generator",
  "tags": ["tag1", "tag2"],
  "targetLanguageCode": ["KO", "JA"] or null,
  "targetCategory": "Sports",
  "originUrl": "https://news.google.com/...",
  "coverImageUrl": "s3://bucket/article/{id}/images/cover.jpg",
  "leveledResults": [
    {
      "level": "A0",
      "chapters": [
        {
          "chapterNum": 1,
          "chapterTitle": "Chapter Title",
          "chunks": [
            {
              "chunkNum": 1,
              "chapterNum": 1,
              "sentenceTexts": ["Sentence 1.", "Sentence 2."]
            }
          ]
        }
      ]
    },
    // ... A1, A2, B1, B2, C1, C2
  ]
}
```

## 제한사항 및 고려사항

1. **Tavily 크레딧 제한**: 월 1,000 크레딧 (기사당 6크레딧 → 최대 166개)
2. **Diffbot 요청 제한**: 월 10,000 요청 (현재 사용량 900)
3. **GitHub Actions 실행 시간**: 작업당 평균 12-15분 (최대 30분)
4. **LLM 안정성**: 드물게 JSON 파싱 실패 시 OutputFixingParser로 복구
5. **언어 감지**: 일부 다국어 혼합 텍스트에서 오탐 가능

## 향후 개선 계획

- [ ] 다른 언어 지원 확대 (중국어, 스페인어 등)
- [ ] 웹 UI 대시보드 (기사 현황, 비용 추적, 품질 모니터링)
- [ ] 커스텀 주제 입력 지원 (RSS 외 수동 입력)
- [ ] 이미지 다양성 향상 (DALL-E 3, Midjourney 통합)
- [ ] 음성 버전 생성 (TTS 통합)

## 라이선스

MIT

## 문서

- [개발 과정 상세](docs/automation.md)
- [완료 체크리스트](docs/automationFinish.md)
- [뉴스 소스 정보](docs/relevant_news.md)

## 문의 및 기여

이슈 및 PR은 GitHub Repository에서 환영합니다.
