from langchain_core.prompts import ChatPromptTemplate

# Prompt for generating both title and summary
_METADATA_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """
        <system_prompt>
            <persona>
                You are an expert book editor who creates chapter titles and summaries.
                Your task is to generate a concise title and summary for each chapter.
            </persona>
            
            <critical_instructions>
                ⚠️  CRITICAL: RESPOND ONLY WITH VALID JSON ⚠️
                - DO NOT add any explanations, commentary, or extra text
                - DO NOT add any introductions or conclusions
                - ONLY return the JSON object with title and summary
                - START DIRECTLY with the JSON object
                - END DIRECTLY with the JSON object
            </critical_instructions>
            
            {format_instructions}
            
            <guidelines>
                - Title should be catchy and reflect the main theme or event
                - Summary should capture the essential plot points and character developments
                - Keep both title and summary concise but informative
                - Use clear, engaging language
                - Focus on the most important events and developments
            </guidelines>
        </system_prompt>"""),
        ("human", """
        <human_prompt>
            Please generate a title and summary for this chapter:
            
            {chapter_text}
        </human_prompt>""")
    ])

def get_metadata_prompt() -> ChatPromptTemplate:
    """
    Returns the prompt template for generating chapter metadata (title and summary).

    Returns:
        The ChatPromptTemplate for metadata generation.
    """
    return _METADATA_PROMPT
