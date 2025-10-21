# articleBasedOnTopic Branch

# 해결 해야 하는 문제 
현재 아티클을 5가지 분야에서 올리고 있다
- Business
- Sports
- Science
- Life
- FunFact

현재 `generate_article_prompt.py`를 보면 
```
        You Must not add any background knowledge, external information, or details not present in the sources.
        Use the key information from all sources to create a comprehensive story that connects the different pieces of information.
        **Focus on finding common themes, contrasting viewpoints, or developing storylines that emerge from the combined sources.**
        Enrich the article with relevant context while staying true to the facts presented in the source materials.
        The final article must be a cohesive synthesis of all provided sources, not just a summary of individual pieces.
```
(한국어 번역)
라는 파트가 있는데 Business, Sports, Science의 경우에는 뉴스 기사들을 보아서 **topic_content**로 주고, 사실 기반으로 만들어져야 하기에 괜찮은 퀄리티가 나온다.

문제는 Life, FunFact 기사들인데 이것들은 기사들이 아니라 **한 줄 정도의 주제**만 던져주고 생성을 한다.

그렇기에 생기는 문제가 
1. background 지식을 이용을 해야 하는데 그걸 막아놓아서 내용이 풍부성이 부족하다.
2. Life, Fun Fact는 다른 기사들 처럼 좀 정적으로 사실을 전달하기 보다는 좀 더 부드럽고 밝은 분위기로 말해야 하는데 정적인 말투로 나오는 게 문제다.

# 해결책

## 구현 방안
`generate_article_node.py`에서 state의 tags 필드를 확인하여 Life/FunFact 여부를 판별하고, `get_article_generation_prompt(is_lifestyle=True/False)`로 전달해서 프롬프트 분기 처리

## 카테고리별 프롬프트 차별화

**일반 카테고리 (Business, Sports, Science) - `is_lifestyle=False`**
- 제약: 소스 기반 사실만 사용, 외부 배경 지식 금지
- 톤: 전문적/중립적, 객관적 정보 전달

**라이프스타일 카테고리 (Life, FunFact) - `is_lifestyle=True`**
- 제약: 배경 지식 활용 허용, 주제 관련 일반 상식 확장 가능
- 톤: 친근한 대화체, 밝고 긍정적, 예시/비유 활용

## 주의사항
- ⚠️ Life/FunFact에서 배경 지식 허용 시 **Hallucination 위험** → 사실 검증 필요
- 카테고리별 샘플 생성 후 내용 풍부성, 톤 일관성, 사실 정확성 테스트

---

# 실행 계획

## 1단계: 프롬프트 수정
- [ ] `generate_article_prompt.py` 현재 구조 파악
- [ ] 일반 카테고리용 제약/톤 프롬프트 (기존 유지)
- [ ] 라이프스타일용 제약/톤 프롬프트 작성
- [ ] `get_article_generation_prompt(is_lifestyle: bool = False)` 파라미터 추가
- [ ] `is_lifestyle` 값에 따라 프롬프트 분기

## 2단계: 노드 로직 수정
- [ ] `generate_article_node.py`에서 tags 확인
- [ ] `is_lifestyle = any(tag in ["Life", "FunFact"] for tag in tags)` 로직 추가
- [ ] `get_article_generation_prompt(is_lifestyle=is_lifestyle)` 호출

## 3단계: 테스트
- [ ] Life 카테고리 샘플 생성 (3개) 및 검증
- [ ] FunFact 카테고리 샘플 생성 (3개) 및 검증
- [ ] Business/Sports/Science 정상 동작 확인

## 4단계: 정리
- [ ] 코드 리뷰 및 커밋

# langfuse
기존에 `genreate_article_node.py`에서 langfuse로 부터 generate-article 프롬프트를 가져오고 있는데 이제 업데이트를 했으니
generate-article-lifestyle 프롬프트를 가져오게 해야 한다.

1. 일단 `/Users/kim-yudam/ling-level/backup/scripts`를 참고해서 **generate-article-lifestyle** 프롬프트를 스크립트를 만들어서 langfuse에 업로드 한다.

2. `genreate_article_node.py`에서 조건에 따라 langfuse에서 다른 이름의 프롬프트 가져오게 한다.
*(`generate_level_specific_text_node.py`의 get_prompt_for_level() 함수 참고해서 작성!!)