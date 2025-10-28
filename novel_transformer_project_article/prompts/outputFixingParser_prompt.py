from langchain_core.prompts import PromptTemplate


def get_output_fixing_prompt() -> PromptTemplate:
    """
    OutputFixingParser에서 사용할 커스텀 프롬프트를 반환합니다.
    유니코드 이스케이프를 방지하기 위한 지시가 포함되어 있습니다.
    """
    NAIVE_FIX = """Instructions:
    --------------
    {instructions}
    --------------
    Completion:
    --------------
    {completion}
    --------------

    Above, the Completion did not satisfy the constraints given in the Instructions.
    Error:
    --------------
    {error}
    --------------

    CRITICAL: 
    - Output Korean/Japanese characters DIRECTLY (e.g., "삼성전자" not as unicode escapes)
    - Never use unicode escape format like: u0000, uXXXX pattern
    - Example WRONG: {{"title": "u0000u0000"}} 
    - Example CORRECT: {{"title": "삼성전자"}}

    Please try again. Please only respond with an answer that satisfies the constraints laid out in the Instructions:"""

    return PromptTemplate.from_template(NAIVE_FIX)
