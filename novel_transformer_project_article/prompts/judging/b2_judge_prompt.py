from langchain_core.prompts import ChatPromptTemplate

def get_judge_prompt() -> ChatPromptTemplate:
    """
    B2 레벨 텍스트 품질을 평가하는 Judge AI용 프롬프트를 반환합니다.
    """
    
    # --- B2 Level Definition ---
    level_definition = """
    <level_definition>
        **B2 Level Definition (Independent User - Upper Intermediate):**
        Can understand the main ideas of complex text on both concrete and abstract topics, including technical discussions in his/her field of specialisation. Can interact with a degree of fluency and spontaneity that makes regular interaction with native speakers quite possible without strain for either party. Can produce clear, detailed text on a wide range of subjects and explain a viewpoint on a topical issue giving the advantages and disadvantages of various options.
    </level_definition>"""

    evaluation_criteria = """
    <evaluation_criteria>
        **B2 Level Evaluation Criteria:**
        
        **1. Vocabulary (25 points)**
        - Uses varied vocabulary including abstract concepts and technical terms relevant to the context.
        - Uses advanced phrasal verbs and idiomatic expressions appropriately.
        - Shows precision in word choice and stylistic variations.
        - Avoids overly simplistic or basic vocabulary that would be more appropriate for A or B1 levels.
        
        **2. Grammar & Sentence Structure (25 points)**  
        - Uses a wide range of tenses including passive voice and reported speech correctly.
        - Uses sophisticated connectors (furthermore, nevertheless, consequently, whereas) effectively.
        - Uses modal verbs for nuanced meaning (ought to, needn't, dare).
        - Uses complex conditional sentences (third conditional, mixed conditionals).
        - Shows good control over complex grammatical structures.
        
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
        - **90-100 points**: PASS - Excellent B2 level adaptation
        - **80-89 points**: PASS - Good B2 level adaptation  
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
        - **Quote specific words or phrases** that are problematic (e.g., "The word 'sophisticated' in sentence 3 is above B2 level")
        - **Reference specific sentences** by number or content (e.g., "Sentence 'The magnificent castle stood proudly' uses vocabulary above B2 level")
        - **Provide concrete replacement suggestions** (e.g., "Replace 'magnificent' with 'big', 'proudly' with 'there'")
        - **Give specific examples** of how to simplify complex structures
        - **Point to exact locations** where changes are needed
        
        - Example: "Issue 1: Word 'sophisticated' in sentence 2 is above B2 level - replace with 'good'. Issue 2: Sentence 'The magnificent castle stood proudly' is too complex - change to 'The castle is big'. Issue 3: Use 'Tom' instead of 'he' in sentence 4 to avoid confusion. Issue 4: Context mismatch - context is about cooking recipes but leveled text discusses computer programs - completely unrelated topics."
    </output_format>"""

    human_prompt_intro = """
    <human_prompt_intro>
        You are an expert CEFR evaluator specializing in B2 level text assessment. Your job is to evaluate whether a leveled text meets B2 standards.
        
        Please evaluate this B2 level text adaptation:

        ## Context Information

        **Previous Chapters Summary:** {cumulative_context}

        **Current Chapter Overview:** {current_chapter_summary}  

        **Previous Chunk Text:** {previous_chunk_context}

        ## Texts to Evaluate

        **Original Text:**
        {original_text}

        **B2 Level Version:**
        {leveled_text}

        ## Instructions
        
        Then evaluate the B2 version against the original, considering:
        - Does it maintain B2 vocabulary and grammar standards?
        - Is the original meaning preserved?
        - Does it connect properly with the context?
        - Is it appropriate for upper-intermediate learners?
        
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