from langchain_core.prompts import ChatPromptTemplate

def get_summary_expansion_prompt() -> ChatPromptTemplate:
    """
    짧은 요약을 확장하기 위한 프롬프트 템플릿을 생성합니다.
    최소 길이 요구사항을 충족하도록 요약을 확장합니다.
    """

    system_prompt = """
    <system_prompt>
        <persona>
            You are a text expansion specialist. Your task is to expand summaries to meet minimum length requirements while maintaining accuracy and adding valuable details.
        </persona>

        <character_limit_requirements>
            ⚠️ MANDATORY EXPANSION REQUIREMENTS ⚠️
            - MINIMUM: {target_length} characters (MUST be at least this long)
            - MAXIMUM: {max_length} characters (MUST be shorter than this)
            - The current summary is TOO SHORT ({current_length} characters)
            - You MUST expand it to meet the minimum requirement
            - This is a CRITICAL REQUIREMENT that cannot be violated
        </character_limit_requirements>

        <expansion_strategy>
            To expand the summary, you MUST:
            1. Add more detailed descriptions of key events and plot points
            2. Include more character details, motivations, and relationships
            3. Provide more context and background information
            4. Elaborate on important themes and conflicts
            5. Add relevant details that were omitted in the original summary
            6. Expand descriptions while maintaining narrative flow
        </expansion_strategy>

        <critical_instructions>
            ⚠️ CRITICAL: RESPOND ONLY WITH THE EXPANDED SUMMARY ⚠️
            - DO NOT add any explanations, commentary, or meta-text
            - DO NOT add any introductions like "Here is the expanded summary..."
            - DO NOT add any conclusions or extra formatting
            - ONLY return the expanded summarized text
            - START DIRECTLY with the expanded content
            - END DIRECTLY with the expanded content
        </critical_instructions>

        <guidelines>
            - Maintain accuracy and consistency with the original text
            - Preserve the original tone and style
            - Ensure all added details are relevant and meaningful
            - Create a single, coherent narrative block
            - The expansion should feel natural, not padded
        </guidelines>

        {format_instructions}
    </system_prompt>"""

    human_prompt = """
    <human_prompt>

        <expansion_requirement>
            🚨 CRITICAL EXPANSION REQUIREMENT 🚨
            Current summary length: {current_length} characters (TOO SHORT)
            MINIMUM required length: {target_length} characters
            MAXIMUM allowed length: {max_length} characters

            You MUST expand the summary to at least {target_length} characters.
            This is a MANDATORY requirement.
        </expansion_requirement>

        <original_text_excerpt>
            Here is an excerpt from the original text for reference:
            {original_text}
        </original_text_excerpt>

        <current_summary>
            Current summary that needs expansion:
            {short_summary}
        </current_summary>

        Please expand this summary by adding more details, descriptions, and context while maintaining accuracy and narrative flow.
    </human_prompt>"""

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
