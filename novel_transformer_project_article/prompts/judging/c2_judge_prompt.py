from langchain_core.prompts import ChatPromptTemplate

def get_judge_prompt() -> ChatPromptTemplate:
    """
    C2 레벨 텍스트 품질을 평가하는 Judge AI용 프롬프트를 반환합니다.
    """
    
    # --- C2 Level Definition ---
    level_definition = """
    <level_definition>
        **C2 Level Definition (Proficient User - Mastery):**
        Can understand with ease virtually everything heard or read. Can summarize information from different spoken and written sources, reconstructing arguments and accounts in a coherent presentation. Can express him/herself spontaneously, very fluently and precisely, differentiating finer shades of meaning even in the most complex situations.
    </level_definition>"""

    evaluation_criteria = """
    <evaluation_criteria>
        **C2 Level Evaluation Criteria:**
        
        **1. Vocabulary (25 points)**
        - Uses the most sophisticated vocabulary and idiomatic expressions with native-like accuracy.
        - Shows advanced lexical variety and precision, differentiating finer shades of meaning.
        - Avoids any simplistic or basic vocabulary that would be more appropriate for lower levels.
        - Uses advanced literary techniques and stylistic devices.
        - Uses complex grammatical structures with perfect control.
        - Uses sophisticated discourse markers and cohesive devices.
        - Uses advanced rhetorical techniques and persuasive language.
        - Uses extensive vocabulary with absolute precision and subtle nuance.
        - Conveys emotions and situations with greater depth and nuance, without sacrificing readability.
        - Uses idiomatic expressions and cultural references with mastery.
        - Uses formal and informal registers with perfect appropriateness.
        - Uses subtle humor, irony, and wit when appropriate.
        - Uses advanced lexical variety and precision.
        - **Focus on nuanced meaning and sophisticated expression.**
        - **Use context-appropriate register and tone with perfect judgment.**
        - **CRITICAL: Text complexity should never come at the expense of readability.**
        
        **2. Grammar & Sentence Structure (25 points)**  
        - Uses complex, varied sentence structures with perfect grammatical control.
        - Uses highly sophisticated sentences with complex structure and elegant flow.
        - Uses sophisticated discourse markers and cohesive devices effectively.
        - Uses advanced literary techniques and stylistic devices appropriately.
        - Shows native-like fluency and precision in complex grammatical structures.
        - Uses all tenses with absolute mastery and native-like fluency.
        - Uses complex grammatical structures with perfect control.
        - Uses sophisticated discourse markers and cohesive devices.
        - Uses advanced rhetorical techniques and persuasive language.
        - Uses varied sentence patterns for sophisticated rhythm and emphasis.
        
        **3. Content Preservation (25 points)**
        - Maintains the basic meaning and sequence of events from original.
        - Keeps same character names and main actions.
        - Story progression remains logical and coherent.
        - **Focus on telling a complete, sophisticated story rather than just isolated concepts.**
        - **Prioritize story completeness and sophistication over grammatical perfection.**
        - **It's better to be slightly grammatically incorrect but tell a sophisticated story than be perfect but basic.**
        - Break complex scenes into masterfully structured, sophisticated sentences that continue the story.
        - Use extensive vocabulary with absolute precision and subtle nuance while preserving the story flow.
        
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
        - **90-100 points**: PASS - Excellent C2 level adaptation
        - **80-89 points**: PASS - Good C2 level adaptation  
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
        - **Quote specific words or phrases** that are problematic (e.g., "The word 'sophisticated' in sentence 3 lacks C2 level sophistication")
        - **Reference specific sentences** by number or content (e.g., "Sentence 'The magnificent castle stood proudly' lacks masterful complexity")
        - **Provide concrete replacement suggestions** (e.g., "Replace 'magnificent' with 'awe-inspiring', 'proudly' with 'with regal dignity' for C2 level")
        - **Give specific examples** of how to achieve masterful complexity and sophistication
        - **Point to exact locations** where changes are needed
        
        - Example: "Issue 1: Word 'sophisticated' in sentence 2 lacks C2 level sophistication - replace with 'exquisitely refined'. Issue 2: Sentence 'The magnificent castle stood proudly' lacks masterful complexity - change to 'The awe-inspiring castle rose with regal dignity'. Issue 3: Use more sophisticated discourse markers in sentence 4 to achieve C2 level. Issue 4: Context mismatch - context is about cooking recipes but leveled text discusses computer programs - completely unrelated topics."
    </output_format>"""

    human_prompt_intro = """
    <human_prompt_intro>
        You are an expert CEFR evaluator specializing in C2 level text assessment. Your job is to evaluate whether a leveled text meets C2 standards.
        
        Please evaluate this C2 level text adaptation:

        ## Context Information

        **Previous Chapters Summary:** {cumulative_context}

        **Current Chapter Overview:** {current_chapter_summary}  

        **Previous Chunk Text:** {previous_chunk_context}

        ## Texts to Evaluate

        **Original Text:**
        {original_text}

        **C2 Level Version:**
        {leveled_text}

        ## Instructions
        
        Then evaluate the C2 version against the original, considering:
        - Does it maintain C2 vocabulary and grammar standards?
        - Is the original meaning preserved?
        - Does it connect properly with the context?
        - Is it appropriate for mastery level learners?
        
        **CRITICAL CONTEXT CHECK:**
        - Compare the topic/subject of the leveled text with the context information provided above
        - Check if character names, locations, and themes mentioned in the leveled text match the context
        - Verify that the leveled text doesn't introduce completely unrelated topics or characters
        - If there's a topic mismatch (e.g., context is about cooking but text is about computers), this is a MAJOR FAILURE

        **CRITICAL: When providing feedback for failed evaluations, you MUST:**
        - **Quote the exact problematic words or phrases** from the text
        - **Reference specific sentences** that need revision
        - **Provide concrete replacement suggestions** with more sophisticated alternatives
        - **Give specific examples** of how to achieve masterful complexity and sophistication
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