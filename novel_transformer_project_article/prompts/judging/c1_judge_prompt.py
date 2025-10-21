from langchain_core.prompts import ChatPromptTemplate

def get_judge_prompt() -> ChatPromptTemplate:
    """
    C1 레벨 텍스트 품질을 평가하는 Judge AI용 프롬프트를 반환합니다.
    """
    
    # --- C1 Level Definition ---
    level_definition = """
    <level_definition>
        **C1 Level Definition (Proficient User - Advanced):**
        Can understand a wide range of demanding, longer texts, and recognise implicit meaning. Can express him/herself fluently and spontaneously without much obvious searching for expressions. Can use language flexibly and effectively for social, academic and professional purposes. Can produce clear, well-structured, detailed text on complex subjects, showing controlled use of organisational patterns, connectors and cohesive devices.
    </level_definition>"""

    evaluation_criteria = """
    <evaluation_criteria>
        **C1 Level Evaluation Criteria:**
        
        **1. Vocabulary & Eloquence (25 points)**
        - Uses sophisticated vocabulary and complex grammatical structures naturally and effectively.
        - Uses advanced vocabulary including academic and formal language appropriately.
        - Shows precise, context-appropriate vocabulary and nuanced expressions.
        - Demonstrates eloquence and effectiveness, not just complexity.
        - **Penalize unnatural word choices that result from simple thesaurus replacement.**
        - Avoids the 'Thesaurus Trap' - mechanical word substitution without natural flow.
        - Avoids overly simplistic or basic vocabulary that would be more appropriate for lower levels.
        - Uses extensive vocabulary with precise nuance and subtlety.
        - Expresses emotions and situations more delicately.
        - Uses academic and professional language with sophistication when it does not harm readability.
        - Uses subtle humor and irony when it is appropriate.
        - **CRITICAL: Text complexity should never come at the expense of readability.**
        
        **2. Grammar & Sentence Structure (25 points)**  
        - Uses complex connectors and discourse markers (albeit, notwithstanding, henceforth) effectively and appropriately.
        - Uses nuanced modal expressions and subtle grammatical forms.
        - Uses varied sentence structures including inversion and ellipsis with sophistication.
        - Shows controlled use of organizational patterns and cohesive devices.
        - **Evaluates entire sentence enhancement, not just individual word complexity.**
        - Demonstrates natural flow and readability despite advanced structures.
        - Uses all tenses with native-like fluency and accuracy.
        - Uses complex sentences with sophisticated structure and flow but it should not harm readability.
        - Uses complex grammatical structures with natural fluency.
        - Uses advanced discourse markers and cohesive devices.
        
        **3. Content Enhancement & Preservation (25 points)**
        - Maintains the basic meaning and sequence of events from original.
        - Keeps same character names and main actions.
        - Story progression remains logical and coherent.
        - **Goes beyond literal meaning to capture author's tone, implicit meaning, and intent.**
        - Enhances the original text's persuasiveness, detail, and stylistic sophistication.
        - Preserves emotional nuance while elevating language sophistication.
        - **Focus on telling a complete, sophisticated story rather than just isolated concepts.**
        - **Prioritize story completeness and sophistication over grammatical perfection.**
        - **It's better to be slightly grammatically incorrect but tell a sophisticated story than be perfect but basic.**
        - Break complex scenes into elegantly structured, sophisticated sentences that continue the story.
        - Use extensive vocabulary with precise nuance and subtlety while preserving the story flow.
        
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
        - **90-100 points**: PASS - Excellent C1 level adaptation
        - **80-89 points**: PASS - Good C1 level adaptation  
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
        - **Quote specific words or phrases** that are problematic (e.g., "The word 'sophisticated' in sentence 3 shows unnatural thesaurus replacement")
        - **Reference specific sentences** by number or content (e.g., "Sentence 'The magnificent castle stood proudly' lacks natural flow despite complexity")
        - **Provide concrete replacement suggestions** (e.g., "Replace 'magnificent' with 'grand', 'proudly' with 'majestically' for more natural flow")
        - **Give specific examples** of how to improve natural flow and eloquence
        - **Point to exact locations** where changes are needed
        
        - Example: "Issue 1: Word 'sophisticated' in sentence 2 shows unnatural thesaurus replacement - replace with 'refined' for natural flow. Issue 2: Sentence 'The magnificent castle stood proudly' lacks natural flow - change to 'The grand castle rose majestically'. Issue 3: Use more natural connectors in sentence 4 to improve flow. Issue 4: Context mismatch - context is about cooking recipes but leveled text discusses computer programs - completely unrelated topics."
    </output_format>"""

    human_prompt_intro = """
    <human_prompt_intro>
        You are an expert CEFR evaluator specializing in C1 level text assessment. Your job is to evaluate whether a leveled text meets C1 standards.
        
        Please evaluate this C1 level text adaptation:

        ## Context Information

        **Previous Chapters Summary:** {cumulative_context}

        **Current Chapter Overview:** {current_chapter_summary}  

        **Previous Chunk Text:** {previous_chunk_context}

        ## Texts to Evaluate

        **Original Text:**
        {original_text}

        **C1 Level Version:**
        {leveled_text}

        ## Instructions
        
        Then evaluate the C1 version against the original, considering:
        - Does it demonstrate eloquence and effectiveness, not just complexity?
        - Does it avoid the 'Thesaurus Trap' with natural, flowing enhancement?
        - Does it capture the author's tone, implicit meaning, and intent beyond literal translation?
        - Is the language sophistication natural and context-appropriate?
        - Does it sound like a well-educated native speaker expressing complex ideas fluently?
        
        **CRITICAL CONTEXT CHECK:**
        - Compare the topic/subject of the leveled text with the context information provided above
        - Check if character names, locations, and themes mentioned in the leveled text match the context
        - Verify that the leveled text doesn't introduce completely unrelated topics or characters
        - If there's a topic mismatch (e.g., context is about cooking but text is about computers), this is a MAJOR FAILURE

        **CRITICAL: When providing feedback for failed evaluations, you MUST:**
        - **Quote the exact problematic words or phrases** from the text
        - **Reference specific sentences** that need revision
        - **Provide concrete replacement suggestions** with more natural alternatives
        - **Give specific examples** of how to improve natural flow and eloquence
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
