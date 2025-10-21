# 요약 길이 제약 개선 방안

## 문제점
- `SummarizedText`에 `min_length=2600` 검증 설정
- 모델이 2600자보다 짧게 응답
- Pydantic 검증 실패 → `OutputFixingParser` 작동하지만 여전히 짧은 결과 생성

## 해결 방안

### 1. Pydantic 검증 제거
```python
class SummarizedText(BaseModel):
    summarized_text: str = Field(max_length=SUMMARIZE_THRESHOLD, description="The summarized text")
    # min_length 제약 제거
```
- 왜냐면 일단 파싱하고 길이 검증을 직접 할 것인데 min_length 걸어놓으면 파싱 실패해서 **outputfixingparser**로 바로 넘어가버린다.

### 2. 파싱 후 수동 길이 검증
```python
response = base_parser.parse(raw_response.content)
summarized_text = response.summarized_text

if len(summarized_text) < 2600:
    # 확장 로직 실행
    summarized_text = _expand_short_summary(...)
```

### 3. 확장 프롬프트로 재요청
- 원본 텍스트 일부 + 짧은 요약 제공
- "반드시 2600자 이상으로 확장하라" 명확히 지시
- 동일한 Pydantic 모델로 파싱
- 프롬프트 위치 -> **/Users/kim-yudam/ling-level/novel_transformer_project/prompts**

### 4. 테스트 및 마무리
- langgraph studio로 잘 되는지 테스트
- 잘 되면 prompt langfuse에 올리고 langfuse에서 가져와서 사용하도록 변경

## 장점
- Pydantic 검증 실패 방지
- 필요한 경우에만 확장 요청 (효율적)
- 길이 제약을 프롬프트로 명확히 전달

## 선택사항
- `max_length`는 유지 (안전장치) 또는 제거 후 수동 검증
