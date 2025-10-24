"""
PydanticOutputParser vs JsonOutputParser vs StrOutputParser 테스트
"""
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser, StrOutputParser

# 테스트할 모델 정의
class SelectedTopic(BaseModel):
    title: str = Field(description="Selected topic title")
    url: str = Field(description="Selected topic url")

class TopicSelection(BaseModel):
    topics: List[SelectedTopic] = Field(description="List of 3 topic candidates")

# 실제 LLM 응답 (에러 발생한 내용)
llm_response = """```json
{
  "topics": [
    {
      "title": "손흥민, 2025 MLS 신인왕 후보 올랐다",
      "url": "https://news.google.com/rss/articles/CBMiggFBVV95cUxNNVdyUy1iYTNrMnpiWHl2bjgyNTlPNH"
    },
    {
      "title": "하이브, 손흥민 소속 LAFC와 K-컬쳐 축제 개최",
      "url": "https://news.google.com/rss/articles/CBMiW0FVX3lxTE5qZjR3UE5zYlIxSUJsUThNaVl3cWpidl"
    },
    {
      "title": "유승민 대한체육회장 "부임 첫 전국체전 성공적…산업화 고민"",
      "url": "https://news.google.com/rss/articles/CBMiW0FVX3lxTE8yVTVYck51WlFrRVdraUFVbXREbTk4QU"
    }
  ]
}
```"""

def test_pydantic_output_parser():
    """PydanticOutputParser 테스트"""
    print("=" * 60)
    print("1. PydanticOutputParser 테스트")
    print("=" * 60)

    parser = PydanticOutputParser(pydantic_object=TopicSelection)

    try:
        result = parser.parse(llm_response)
        print("✅ 파싱 성공!")
        print(f"타입: {type(result)}")
        print(f"결과: {result}")
        print(f"topics 개수: {len(result.topics)}")
        for i, topic in enumerate(result.topics, 1):
            print(f"  {i}. {topic.title}")
            print(f"     URL: {topic.url}")
        return True
    except Exception as e:
        print(f"❌ 파싱 실패!")
        print(f"에러 타입: {type(e).__name__}")
        print(f"에러 메시지: {str(e)[:200]}")
        return False

def test_json_output_parser():
    """JsonOutputParser 테스트"""
    print("\n" + "=" * 60)
    print("2. JsonOutputParser 테스트")
    print("=" * 60)

    parser = JsonOutputParser(pydantic_object=TopicSelection)

    try:
        result = parser.parse(llm_response)
        print("✅ 파싱 성공!")
        print(f"타입: {type(result)}")
        print(f"결과: {result}")

        # dict로 반환되므로 직접 접근
        if isinstance(result, dict) and "topics" in result:
            print(f"topics 개수: {len(result['topics'])}")
            for i, topic in enumerate(result['topics'], 1):
                print(f"  {i}. {topic['title']}")
                print(f"     URL: {topic['url']}")
        return True
    except Exception as e:
        print(f"❌ 파싱 실패!")
        print(f"에러 타입: {type(e).__name__}")
        print(f"에러 메시지: {str(e)[:200]}")
        return False

def test_str_output_parser():
    """StrOutputParser 테스트"""
    print("\n" + "=" * 60)
    print("3. StrOutputParser 테스트")
    print("=" * 60)

    parser = StrOutputParser()

    try:
        result = parser.parse(llm_response)
        print("✅ 파싱 성공!")
        print(f"타입: {type(result)}")
        print(f"결과 길이: {len(result)} 글자")
        print(f"결과 (처음 200자):\n{result[:200]}...")
        return True
    except Exception as e:
        print(f"❌ 파싱 실패!")
        print(f"에러 타입: {type(e).__name__}")
        print(f"에러 메시지: {str(e)[:200]}")
        return False

if __name__ == "__main__":
    print("LLM 응답 내용:")
    print("-" * 60)
    print(llm_response)
    print("-" * 60)
    print()

    result1 = test_pydantic_output_parser()
    result2 = test_json_output_parser()
    result3 = test_str_output_parser()

    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    print(f"PydanticOutputParser: {'✅ 성공' if result1 else '❌ 실패'}")
    print(f"JsonOutputParser: {'✅ 성공' if result2 else '❌ 실패'}")
    print(f"StrOutputParser: {'✅ 성공' if result3 else '❌ 실패'}")
