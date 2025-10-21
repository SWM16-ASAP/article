from langchain_core.prompts import ChatPromptTemplate

def get_judge_prompt() -> ChatPromptTemplate:
    """
    A2 레벨 텍스트 품질을 평가하는 Judge AI용 프롬프트를 반환합니다.
    """
    
    # --- A2 Level Definition ---
    level_definition = """
    <level_definition>
        **A2 Level Definition (Basic User - Elementary):**
        Can understand sentences and frequently used expressions related to areas of most immediate relevance (e.g. very basic personal and family information, shopping, local geography, employment). Can communicate in simple and routine tasks requiring a simple and direct exchange of information on familiar and routine matters. Can describe in simple terms aspects of his/her background, immediate environment and matters in areas of immediate need.
    </level_definition>"""

    evaluation_criteria = """
    <evaluation_criteria>
        **A2 Level Evaluation Criteria:**
        
        **1. Vocabulary (25 points)**
        - Uses basic vocabulary about daily life, work, and personal interests.
        - Uses common phrasal verbs and everyday expressions.
        - No highly specialized or academic vocabulary.
        - No complex idioms or metaphors.
        
        **2. Grammar & Sentence Structure (25 points)**  
        - Uses simple present, past, and future tenses.
        - Uses simple connectors (and, but, because, so, then).
        - Uses basic modal verbs (can, must, should).
        - Uses simple question forms and basic conditional sentences.
        - No complex grammatical structures (e.g., perfect tenses, advanced conditionals, reported speech, passive voice).
        
        **3. Content Preservation (25 points)**
        - Maintains the basic meaning and sequence of events from original.
        - Keeps same character names and main actions.
        - Story progression remains logical and coherent.
        
        **4. Context & Continuity (25 points)**
        - Connects naturally with previous chunk context.
        - Maintains story continuity with chapter summary.
        - Preserves emotional tone at appropriate level.
        - **CRITICAL**: Check if the leveled text's topic/subject matches the context information.
        - **CRITICAL**: Verify that character names, locations, and main themes are consistent with the provided context.
        - **CRITICAL**: Ensure the leveled text doesn't introduce completely unrelated topics or characters.
    </evaluation_criteria>"""

    scoring_system = """
    <scoring_system>
        **Scoring System:**
        - **90-100 points**: PASS - Excellent A2 level adaptation
        - **80-89 points**: PASS - Good A2 level adaptation  
        - **70-79 points**: MARGINAL - May need minor improvements
        - **Below 70 points**: FAIL - Requires significant revision
        
        **Each criterion scored 0-25 points:**
        - 23-25: Excellent
        - 20-22: Good
        - 17-19: Adequate
        - 14-16: Needs improvement
        - 0-13: Poor

        **PRECISION SCORING INSTRUCTION:**
        - **Provide a precise score. Avoid rounding to numbers ending in 0 or 5.**
        - **A score of 23 is more helpful and precise than 20 or 25.**
    </scoring_system>"""

    output_format = """
    <output_format>
        **CRITICAL: You must respond ONLY with a valid JSON object in this exact format:**

        {format_instructions}

        **JSON Response Requirements:**
        - overall_score: Integer from 0-100
        - is_acceptable: true if overall_score >= 80, false otherwise
        - criteria_scores: Object with each criterion scored 0-25
        - feedback: String with detailed explanation **required only for is_acceptable is false cases**

        **CRITICAL FEEDBACK INSTRUCTIONS:**
        - If is_acceptable is TRUE: Set feedback to empty string ""
        - If is_acceptable is FALSE: Provide detailed feedback explaining specific issues and improvement suggestions
        - DO NOT provide feedback when the text passes evaluation (is_acceptable: true)
        - CRITICAL: Write feedback as a SINGLE LINE without line breaks or newlines
        - Use semicolons (;) or periods (.) to separate different points in feedback
        
        **DETAILED FEEDBACK REQUIREMENTS:**
        - **Quote specific words or phrases** that are problematic (e.g., "The word 'sophisticated' in sentence 3 is above A2 level")
        - **Reference specific sentences** by number or content (e.g., "Sentence 'The magnificent castle stood proudly' uses vocabulary above A2 level")
        - **Provide concrete replacement suggestions** (e.g., "Replace 'magnificent' with 'big', 'proudly' with 'there'")
        - **Give specific examples** of how to simplify complex structures
        - **Point to exact locations** where changes are needed
        
        - Example: "Issue 1: Word 'sophisticated' in sentence 2 is above A2 level - replace with 'good'. Issue 2: Sentence 'The magnificent castle stood proudly' is too complex - change to 'The castle is big'. Issue 3: Use 'Tom' instead of 'he' in sentence 4 to avoid confusion. Issue 4: Context mismatch - context is about cooking recipes but leveled text discusses computer programs - completely unrelated topics."
    </output_format>"""

    human_prompt_intro = """
    <human_prompt_intro>
        You are an expert CEFR evaluator specializing in A2 level text assessment. Your job is to evaluate whether a leveled text meets A2 standards.
        
        Please evaluate this A2 level text adaptation:

        ## Context Information

        **Previous Chapters Summary:** {cumulative_context}

        **Current Chapter Overview:** {current_chapter_summary}  

        **Previous Chunk Text:** {previous_chunk_context}

        ## Texts to Evaluate

        **Original Text:**
        {original_text}

        **A2 Level Version:**
        {leveled_text}

        ## Instructions
        
        Then evaluate the A2 version against the original, considering:
        - Does it maintain A2 vocabulary and grammar standards?
        - Is the original meaning preserved?
        - Does it connect properly with the context?
        - Is it appropriate for elementary learners?
        
        **CRITICAL CONTEXT CHECK:**
        - Compare the topic/subject of the leveled text with the context information provided above
        - Check if character names, locations, and themes mentioned in the leveled text match the context
        - Verify that the leveled text doesn't introduce completely unrelated topics or characters
        - If there's a topic mismatch (e.g., context is about cooking but text is about computers), this is a MAJOR FAILURE

        **CRITICAL: When providing feedback for failed evaluations, you MUST:**
        - **Quote the exact problematic words or phrases** from the text
        - **Reference specific sentences** that need revision
        - **Provide concrete replacement suggestions** with simpler alternatives
        - **Give specific examples** of how to simplify complex structures
        - **Point to exact locations** where changes are needed
        - **For context mismatches**: Clearly state what the context is about vs what the text is about
        
        Provide detailed feedback focusing on specific improvements needed if the text fails evaluation.
    </human_prompt_intro>"""

    system_prompt_text = f"""{level_definition}\n\n{evaluation_criteria}\n\n{scoring_system}\n\n{output_format}"""
    human_prompt_text = f"""{human_prompt_intro}"""

    messages = [
        ("system", system_prompt_text),
        ("human", human_prompt_text)
    ]
            
    return ChatPromptTemplate.from_messages(messages)
