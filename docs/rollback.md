# feat/rollback 브랜치 내용

## 해야 할 것

1. 청킹 이미지가 단위로 -> 이미지가 없으면 챕터 전문이 하나의 청크가 된다.
    * 이거 langfuse에 올라가 있는 프롬프트가 챕터 단위 레벨링 청크라서 현재 Langfuse에서 챕터 들고 오면 안된다.


2. 마지막에 문장 단위로 분리해서 문장을 청크로 넣어서 제공을 해주기 -> 원문 전처리도 포함

1. 문장 분리 라이브러리 교체
SpaCy -> pySBD로 변경

---- 

# 해야 할 일
* langfuse에서 프롬프트 한번만 가져오게 하기
* summarize 처리하기
* 대화문 안에서 문장 분리 규칙을 좀 정하기

---

## 대화문 문장 분리 규칙 (pySBD 후처리)

### 문제점
pySBD가 대화문 내부에서 문장을 잘못 분리하는 문제:
```
입력: "hello! my name is james."
현재: ["hello!", "my name is james."]  ❌
원하는 결과: ["hello! my name is james."]  ✅
```

### 후처리 규칙 (Robust)

#### 1. **인용부호 매칭 규칙** (핵심)
- 열린 인용부호가 닫히지 않으면 다음 문장과 합침
- 지원 인용부호: `"`, `'`, `"`, `"`, `'`, `'`, `「`, `」`, `『`, `』`, `«`, `»`
- **중첩 인용부호 처리**: 스택 기반으로 매칭 추적
  ```
  예: "He said 'wait here' to me." → 올바르게 매칭
  ```

#### 2. **소유격/축약형 예외 처리** (중요)
- `\w'\w` 패턴은 인용부호로 카운트하지 않음
- 예외 케이스:
  - 소유격: `Clark's car`, `James' book`
  - 축약형: `It's`, `don't`, `I'm`, `we're`, `they've`, `you'll`, `he'd`
- 규칙: apostrophe 앞뒤에 영문자가 있으면 인용부호가 아님

#### 3. **대화 태그 인식**
다음 문장이 대화 태그로 시작하면 이전 대화문과 합침:
- 직접 태그: `said`, `asked`, `replied`, `whispered`, `shouted`, `exclaimed`, `murmured`, `muttered`, `answered`, `responded`, `continued`, `added`, `noted`, `remarked`, `observed`, `commented`, `stated`, `declared`, `announced`, `mentioned`, `explained`, `began`, `interrupted`, `protested`, `insisted`, `admitted`, `confessed`, `agreed`, `disagreed`, `argued`, `suggested`
- 대명사 + 태그 패턴: `he said`, `she asked`, `I replied`, `they shouted`
  ```
  예: ["\"I don't know,\"", "she said quietly."]
  결과: ["\"I don't know,\" she said quietly."]
  ```

#### 4. **불완전한 대화 처리**
의도적으로 끊긴 대화는 다음 문장과 합침:
- **Em-dash로 끝남**: `"Wait, I—"`
- **생략부호**: `"Well..."` (인용부호 없이 끝나는 경우는 제외)
  ```
  예: ["\"Wait, I—\"", "She interrupted him."]
  결과: ["\"Wait, I—\" She interrupted him."]
  ```

#### 5. **구두점 고려 병합**
문장을 합칠 때 스페이스 처리:
- Em-dash 뒤에는 스페이스 없이: `I—She`
- 구두점으로 시작하면 스페이스 없이: `know,` + `,she` → `know,she`
- 그 외: 스페이스 추가

### 적용 위치
`parallel_cefr_processing_node.py`에서 pySBD 세그먼테이션 직후:
1. **Line 114 이후**: 원문 문장 분리
2. **Line 155 이후**: 레벨링된 텍스트 문장 분리 후

### 테스트 케이스
```python
# 1. 기본 대화문
["\"Hello!", "My name is James.\""]
→ ["\"Hello! My name is James.\""]

# 2. 소유격 (예외 처리)
["Clark's car is fancy.", "It's very expensive."]
→ ["Clark's car is fancy.", "It's very expensive."]  # 분리 유지

# 3. 대화 태그
["\"I don't know,\"", "she said quietly."]
→ ["\"I don't know,\" she said quietly."]

# 4. 중첩 인용부호
["\"He told me 'wait here' yesterday.\"", "That was strange."]
→ ["\"He told me 'wait here' yesterday.\"", "That was strange."]

# 5. 불완전한 대화
["\"Wait, I—\"", "She interrupted him."]
→ ["\"Wait, I—\" She interrupted him."]

# 6. 혼합
["\"It's John's car,\"", "he said.", "\"Really?\"", "she asked."]
→ ["\"It's John's car,\" he said.", "\"Really?\" she asked."]
```

### 구현 전략
- 함수명: `merge_dialogue_sentences(sentences: List[str]) -> List[str]`
- 정규식 패턴으로 소유격 필터링
- 스택 기반 인용부호 매칭
- Look-ahead로 다음 문장 확인 후 병합 여부 결정