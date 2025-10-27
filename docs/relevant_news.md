 # 겪고 있는 문제

 현재 하나의 기사만 소스로 사용해서 새로운 기사를 생성을 하면 혹시 저작권이나 이런 문제에 걸릴 수 있기에 최소 2개의 기사를 같이 제공을 해주고 있다.

 여기서 문제가 발생하는데

 1. 에세이
 **Science**의 경우에는 트렌드보다는 각 잡고 에세이를 쓰는 형식이기에 관련 기사를 찾기가 어렵다. 
 사실상 하나만을 참고하고 있다.

그리고 **Technology** 또한 Science 만큼은 아니지만 비슷한 양상을 띄고 있다.

2. 기사 제목이 기사 내용을 반영하지 못할 경우
**한국 스포츠**의 경우에 생각보다 관련 기사들이 안나오는 케이스들이 있다.
대표적인 예시로 한국 시리즈에 관한 기사들이 안나온 적이 있는데 아마 기사들이 없을 이유는 없고 기사 제목과 임베딩을 하기에 이 때문에 안나오는 경우가 있을 수 있다.

3. 관련 기사가 아닌 플랫폼을 통채로 들고 오는 경우
기사를 들고 와야 하는데 그 기사가 들고 있는 사이트의 홈페이지를 들고 오는 경우가 있다.
이런 경우에 사용이 불가능 핟.

# 해결책을 위한 브레인 스토밍

## 1번 문제
- 리스크를 걸고 1개만 처리한다.
- 최근 트렌드를 다루는 사이트들로 변경한다. -> technology는 그게 가능할 것 같은데 Science은 한번 찾아봐야겠다.
-> https://www.sciencenews.org/feed 과학은 이게 더 나은 것 같고 기술은 2번 해결책을 사용하면 될 듯

## 2번 문제
- 기사 제목이 아닌 기사 내용을 전달한다. -> tavily 테스트 필요 -> 기사 내용이 너무 길 경우를 대비해서 ai로 요약을 해야 할 수도 있다. 
*테스트 결과* : 내용을 요약해서 전달했을 때 제목만 전달하는 것보다 훨씬 좋은 결과가 나온다. 다만 쿼리의 최대 길이가 400자여서 llm을 통한 요약은 불가피할 듯

득: 기존에 3번 문제가 심해서 현재 일본 스포츠를 구글 뉴스가 아닌 영어로 일본 스포츠를 전달해주는 곳으로 바꿨는데(그렇게 좋은 사이트가 아님) 그걸 다시 일본 구글 스포츠 뉴스로 변경할 수 있다.(내용 기반이라서 그런지 3번 문제가 덜해짐)
실: llm으로 요약을 해야 해서 llm 비용이 나간다

내용을 요약하다 보니 쿼리가 길어져서 코사인 유사도 점수 기준을 낮춰줘야 할 것

## 3번 문제
- tavily에는 topic 필드가 있어서 거기서 news로 지정을 하면 뉴스들만 끌고 온다. -> 다만 영어 기사에 한정되어 있기에 한국이나 일본 맞춤형 주제에 대해서는 어려울 수 있다.
지금 한국 일본이 갈려 있는 부분은 1. 비지니스, 2. 문화, 3. 스포츠 이렇게 3가지 인데 그러면 이 3가지를 제외한 1. 기술 2. 과학 에만 적용이 가능하지 않을까

# todo list -> `fetch_topic_and_articles_node.py`
[x] Science, Sports(ja) 사이트 교체
[x] 요약해서 tavily 쿼리에 넣도록 수정 
    1. select_topic_with_llm을 한 뒤에 url이 llm이 너무 길면 잘라서 주기 때문에 headlines에서 해당 url을 포함하고 있는 사이트 찾아서 풀 url로 topic_candidates를 update하기

    2. 주제 요약 함수를 만들어서 거기서 diffbot을 이용해서 본문 추출하고 llm으로 200자 요약하라고 하고 반환 (400자가 제한이지만 안전빵으로) (혹시 400자가 넘으면 다시 돌리기)

    3. 그걸 query로 쏘기

[] tavily 조건
    1. Science는 날짜 제한 없애기
    2. Science, Technology는 topic = news로 설정

    ```python
    
    # To install: pip install tavily-python
from tavily import TavilyClient
client = TavilyClient("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")
response = client.search(
    query="Glioblastoma, a type of brain cancer, can erode the skull above tumors by activating bone-destroying cells, leading to about 20% bone loss. The impact of this bone loss on cancer growth remains unclear.﻿",
    topic="news"
)
print(response)
```