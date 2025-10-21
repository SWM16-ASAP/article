from langchain_core.prompts import ChatPromptTemplate

def get_article_generation_prompt(is_lifestyle: bool = False) -> ChatPromptTemplate:
    """
    Creates and returns the prompt template for generating a viral news article.

    Args:
        is_lifestyle: If True, uses lifestyle-friendly prompt (Life, FunFact categories).
                     If False, uses news-based prompt (Business, Sports, Science categories).
    """
    
    guidelines = """
        <guidelines>
            **Article Style Guidelines:**
            - title MUST be simple clearly convey the content
            - title MUST be written at A1 level but A2 level words can be used ONLY when necessary for clarity or natural expression..
            - title MUST be 60 characters or less.
            - Use English not other languages.
            - content MUST be over 1000 characters and under 1500 characters.
            - content MUST NOT mention or suggest sharing on any other apps, websites, or social media platforms (e.g., TikTok, Instagram, X, Facebook).
            - The final output MUST be in a JSON format with 'title' and 'content' keys.
            {format_instructions}
        </guidelines>"""

    A1_level_requirements = """
    <A1_level_requirements>
        **A1 Level for title (Basic User - Beginner):**
        Can understand and use familiar everyday expressions and very basic phrases aimed at the satisfaction of needs of a concrete type. Can introduce him/herself and others and can ask and answer questions about personal details such as where he/she lives, people he/she knows and things he/she has. Can interact in a simple way provided the other person talks slowly and clearly and is prepared to help.
        
        **Guidelines for A1 Level:**
        - Use simple present tense and basic past tense
        - No future or perfect tenses (focus on 'here and now')
        - Use short, simple sentences
        - Use basic vocabulary (family, food, colors, common objects)
        - **Deconstruct Complex Ideas**: If an idea from the original text is too complex for simple A1 vocabulary (e.g., "he was relieved"), describe the feeling or action in a simpler way. For example, instead of "He was relieved," you could write "He was not worried anymore. He felt happy."
        - Use simple connectors (and, but, because)
        - Use basic question forms (What, Where, When, Who)
        - Use common everyday expressions
        - Use basic descriptive words (big, small, good, bad, happy, sad)
        - Focus on concrete, immediate experiences
        - Use simple action verbs (go, come, eat, sleep, work, play)
        - Use basic time expressions (today, tomorrow, yesterday)
    </A1_level_requirements>"""

    C1_level_requirements = """
    <C1_level_requirements>
        **C1 Level for content (Proficient User - Advanced):**
        Can understand a wide range of demanding, longer texts, and recognise implicit meaning. Can express him/herself fluently and spontaneously without much obvious searching for expressions. Can use language flexibly and effectively for social, academic and professional purposes. Can produce clear, well-structured, detailed text on complex subjects, showing controlled use of organisational patterns, connectors and cohesive devices.
        
        **Guidelines for C1 Level:**
        - You must not create content that does not exist in the original work.
        - Use all tenses with native-like fluency and accuracy
        - Use complex sentences with sophisticated structure and flow but it should not harm readability.
        - Use extensive vocabulary with precise nuance and subtlety
        - Express emotions and situations more delicately.
        - Use complex grammatical structures with natural fluency
        - Use advanced discourse markers and cohesive devices
        - Use idiomatic expressions and cultural references
        - Use academic and professional language with sophistication when it does not harm readability.
        - Use subtle humor and irony when it is appropriate
    </C1_level_requirements>"""
    
    meta_commentary_ban = """
    <meta_commentary_ban>
        ⚠️ CRITICAL: NO META-COMMENTARY OR EXTRA TEXT ⚠️
        - DO NOT add any text like "Here is the article:", "C1 Level:", or any explanations.
        - DO NOT add any introductions, commentary, or labels.
        - START DIRECTLY with the JSON output.
        - END DIRECTLY with the JSON output.
    </meta_commentary_ban>"""

    # Combine all instructions into the final system prompt for strong enforcement
    system_prompt = f"""
    <system_prompt>
        {guidelines}

        {A1_level_requirements}

        {C1_level_requirements}

        {meta_commentary_ban}
    </system_prompt>"""

    human_prompt_topic = """
    <topic_instructions>
        Based on the following multiple source materials, create interesting news article with a title and content.
        Synthesize the information from all sources to create a comprehensive and engaging article.

        title should be written at A1 level and 60 characters or less.
        content should be written at C1 level and over 1000 characters and under 1500 characters.

        Source Materials:
        "{topic_content}"
    </topic_instructions>"""

    # 카테고리별 프롬프트 분기
    if is_lifestyle:
        # 라이프스타일 카테고리 (Life, FunFact) - 배경 지식 허용, 친근한 톤
        human_prompt_persona = """
    <persona>
        You are a creative and engaging lifestyle writer specializing in interesting and relatable content.
        Your task is to create an engaging article based on the given topic.

        **Content Creation Approach:**
        - You MAY use general background knowledge and common facts to enrich the content.
        - Feel free to expand on the topic with relevant examples, analogies, and practical information.
        - Draw from widely-known information to make the content more comprehensive and valuable.
        - Ensure all added information is factually accurate and commonly accepted.

        **Tone & Style:**
        - Use a friendly, conversational tone that feels warm and approachable.
        - Write with a bright and positive atmosphere.
        - Create empathy and connection with readers.
        - Use relatable examples and vivid analogies to illustrate points.
        - Make the content feel personal and engaging, not academic or distant.
    </persona>"""
    else:
        # 일반 카테고리 (Business, Sports, Science) - 소스 기반, 전문적 톤
        human_prompt_persona = """
    <persona>
        You are a highly skilled journalist specializing in interesting and informative content.
        Your task is to analyze multiple source materials and synthesize them into a single, interesting and informative news article.
        You Must not add any background knowledge, external information, or details not present in the sources.
        Use the key information from all sources to create a comprehensive story that connects the different pieces of information.
        **Focus on finding common themes, contrasting viewpoints, or developing storylines that emerge from the combined sources.**
        Enrich the article with relevant context while staying true to the facts presented in the source materials.
        The final article must be a cohesive synthesis of all provided sources, not just a summary of individual pieces.
    </persona>"""

    human_prompt = f"""
    <human_prompt>
        {human_prompt_topic}
        
        {human_prompt_persona}
    </human_prompt>"""

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])