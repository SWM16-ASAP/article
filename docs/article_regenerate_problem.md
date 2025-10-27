# 문제

`generate_article_node.py`에서 아티클 pydantic_object를 아래와 같이 지정해놓았다.

```python
class ArticleGeneration(BaseModel):
    title: str = Field(max_length=80, description="headline for the article")
    content: str = Field(max_length=1500, description="The main content of the article, under 1500 characters")
```

여기서 문제가 json 형식으로는 잘 준다. 하지만 max_length 제한에 걸려서 outputFixingParser로 넘어가는 경우가 많고 거기서 다시 고치면 content가 너무 줄어들어서 다시 밑에서 `_expand_short_article` 함수를 통해 확장을 여러번 시키면서 토큰을 여러 번 소모하게 된다.


그래서 일단 pydantic_object에서 max_length는 제거를 하고 나온 다음에 길이에 대한 것을 맞추는 것이 맞는 것 같다. 

가능한 케이스
1. title이 long
2. content가 short
3. content가 long

pydantic model에 길이 제약을 떼고 생성된 title, content가 목표했던 길이(각각 10자 ~ 60자, 1000자 ~ 1500자)에 어느 정도 일치만 한다면 그렇게 두 면 되지 않을까...?

테스트를 해보니

위의 3개의 케이스에 대해서 처리를 해야 할 것 같은데

1, 3번은 요약이니까 full_text를 다시 제공 안하고 그냥 어느 정도로 줄여야 하는지 전달하고 그 정도로 요약해달라고 하고
2 번은 지금 하듯이 expand 함수 사용하면 될 것 같다.

title과 content를 length_validate하는 함수를 만들면 될 듯

length_validate 함수에서 title ㄱ