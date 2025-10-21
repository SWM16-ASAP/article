from langchain_core.prompts import ChatPromptTemplate

def get_prompt(use_rag: bool = False, is_feedback_generated: bool = False) -> ChatPromptTemplate:
    """
    Builds and returns the complete ChatPromptTemplate for the B2 level,
    dynamically including RAG instructions and feedback if required.
    """
    
    # --- Define Content ---
    level_definition = """
    <level_definition>
        **B2 Level Definition (Independent User - Upper Intermediate):**
        Can understand the main ideas of complex text on both concrete and abstract topics, including technical discussions in his/her field of specialisation. Can interact with a degree of fluency and spontaneity that makes regular interaction with native speakers quite possible without strain for either party. Can produce clear, detailed text on a wide range of subjects and explain a viewpoint on a topical issue giving the advantages and disadvantages of various options.
    </level_definition>"""

    level_guidelines = """
    <level_guidelines>
        **⚠️ CRITICAL: COMPLEXITY AND READABILITY ⚠️** 
        - Crucially, a text's difficulty and complexity should never come at the expense of its readability.
        
        **Guidelines for B2 Level:**
        - You must not create content that does not exist in the original work.
        - Use all major tenses with confidence and accuracy
        - Use complex sentences with multiple clauses and connectors
        - Use advanced vocabulary with good precision and variety
        - Use sophisticated descriptive language and literary devices
        - Can use relative clauses and subordinate clauses when it is appropriate.
        - Use reported speech, indirect questions, and passive voice
        - Use modal verbs and phrasal verbs with nuance
        - Use conditional sentences (all types) and subjunctive mood
        - Use linking words and discourse markers effectively
        - Use idiomatic expressions and collocations
        - Use academic and formal language when appropriate
    </level_guidelines>"""
    
    critical_instructions = """
    <critical_instructions>
        ⚠️  CRITICAL: ABSOLUTELY NO EXPLANATIONS, LABELS, OR EXTRA TEXT ⚠️
        - DO NOT add any text like "B2 Level:", "Here is the B2 version:", or any explanations
        - DO NOT add any introductions, commentary, or labels and tags
        - ONLY return the B2 level text
        - START DIRECTLY with the B2 text
        - END DIRECTLY with the B2 text

        ⚠️  **CRITICAL: OUTPUT FORMAT EXAMPLE** ⚠️
        <example>
        Expected Output:
        The inquisitive young boy stumbled upon an extraordinarily magnificent canine specimen in the verdant expanse of the municipal park. He experienced an overwhelming surge of exhilaration, as his lifelong passion for the animal kingdom had never waned. The remarkably amiable four-legged companion responded with enthusiastic tail-wagging upon the child's approach, establishing an instantaneous bond between the two beings.

        NOT like this:
        "B2 Level Version: Here is the adapted text..."
        "I have adapted the text to B2 level..."
        "Based on the original, here is the B2 version..."
        </example>

        ⚠️  **CRITICAL: OUTPUT ONLY IN ENGLISH** ⚠️
        - MUST respond ONLY in English language
        - DO NOT use any other language (Korean, Chinese, Japanese, etc.)
        - ALL text must be in English only

        ⚠️  **CRITICAL: STAY TRUE TO THE ORIGINAL TEXT** ⚠️
        - DO NOT create new characters, events, or plot points
        - DO NOT change the basic meaning or sequence of events
        - ONLY enhance the language to B2 level while keeping the same story
        - If something cannot be expressed at B2 level, use the most appropriate advanced words
    </critical_instructions>"""

    human_prompt_persona = """You are an expert English teacher specializing in adapting texts for B2 level learners (Independent User - Upper Intermediate)."""

    feedback_instructions = """
    <feedback_instructions>
        ---
        
        ## 🚨 FEEDBACK ON PREVIOUS ATTEMPT - IMPLEMENTATION REQUIRED 🚨
        
        Your previous attempt was not quite right. Here is the feedback to help you improve:
        
        **Your Previous Output:**
        {leveled_text}
        
        **Feedback:**
        {feedback}
        
        🎯 **CRITICAL: FEEDBACK IMPLEMENTATION IS YOUR TOP PRIORITY** 🎯
        - **FEEDBACK MUST BE IMPLEMENTED FIRST** before any other considerations
        - **ADDRESS EVERY SINGLE POINT** mentioned in the feedback
        - **NO EXCEPTIONS** - if feedback says "fix X", you MUST fix X
        - **FEEDBACK OVERRIDES ALL OTHER INSTRUCTIONS** when there's a conflict
        - **SUCCESS DEPENDS ON PRECISE FEEDBACK IMPLEMENTATION**
        
        Please carefully review the feedback and the previous output. Rewrite the original text again, making sure to address all the points in the feedback.
        
        🚨 **CRITICAL: NO META COMMENTARY ABOUT FEEDBACK** 🚨
        - DO NOT mention that you received feedback or made corrections
        - DO NOT say things like "I have revised", "Based on feedback", "Here is the corrected version"
        - DO NOT reference the previous attempt or this revision process
        - DO NOT explain what changes you made or why
        - ONLY provide the pure B2 level story text - nothing else
        - Act as if this is your first and only attempt at the B2 level text
        ---
    </feedback_instructions>"""

    human_prompt_context_and_instructions = """
    <human_prompt_context_and_instructions>

        Please rewrite this English text for B2 level, considering the following context:

        ## 🎯 PRIORITY INSTRUCTIONS 🎯
        
        **If feedback was provided:**
        - **FEEDBACK IMPLEMENTATION IS YOUR ABSOLUTE TOP PRIORITY**
        - **FEEDBACK OVERRIDES ALL OTHER INSTRUCTIONS** when there's a conflict
        - **ADDRESS EVERY SINGLE FEEDBACK POINT** before considering other improvements
        - **SUCCESS DEPENDS ON PRECISE FEEDBACK IMPLEMENTATION**
        
        **If no feedback:**
        - Follow all guidelines and instructions below
        - Focus on creating high-quality B2 level text
        - Ensure proper context and continuity

        ## Context Information

        ### Previous Chapters Summary
        *(For story continuity)*

        {cumulative_context}

        ### Current Chapter Overview
        *(What this chapter is about)*

        {current_chapter_summary}

        ### Previous Chunk Text
        *(Text that came immediately before this chunk)*

        {previous_chunk_context}

        ---

        ## Instructions

        - Use the above context to maintain story continuity and character consistency
        - The **Previous Chapters Summary** shows what happened in previous chapters
        - The **Current Chapter Overview** explains what this chapter is about
        - The **Previous Chunk Text** shows the text that came right before this chunk
        - Keep the same character names from the context (do not change names)
        - Keep the same basic events and actions from the original text
        - Make sure your B2 text connects naturally with the previous chunk
        - **Focus on telling a complete, sophisticated story rather than just isolated concepts**
        - Use complex sentences with multiple clauses and sophisticated connectors
        - **Prioritize story completeness and sophistication over grammatical perfection**
        - **It's better to be slightly grammatically incorrect but tell a sophisticated story than be perfect but basic**
        - Break complex scenes into well-structured, sophisticated sentences that continue the story
        - Use advanced vocabulary with good precision and variety while preserving the story flow
    </human_prompt_context_and_instructions>"""

    # --- Build Prompt Structure ---
    system_prompt_text = f"{level_definition}\n\n{level_guidelines}\n\n{critical_instructions}"
    
    # Start with the persona
    human_prompt_text = human_prompt_persona

    # Add feedback instructions if feedback was generated
    if is_feedback_generated:
        human_prompt_text += feedback_instructions

    # Add the main context and instructions
    human_prompt_text += human_prompt_context_and_instructions

    # Add RAG-specific parts if needed
    if use_rag:
        rag_system_addition = """
    <rag_system_addition>

        **Available Tools:**
        You have access to the find_alternative_words tool. Use it for comprehensive B2 level adaptation, not just word replacement.
        - Identify ALL words that don't match B2 level: both too difficult AND too simple words
        - Look for: complex vocabulary, advanced grammar, AND overly basic words that need upgrading
        - Call find_alternative_words(words="word1, word2, word3", target_level="B2", context="actual surrounding text from the passage")
        - This tool provides B2-level alternatives as reference - use them as guidance, not strict requirements
    </rag_system_addition>"""

        rag_human_addition = """<rag_human_addition>

        - **IMPORTANT**: For comprehensive B2 level adaptation:
          1. Analyze the ENTIRE text for B2 level appropriateness
          2. Identify words that are: TOO DIFFICULT (above B2) AND TOO SIMPLE (below B2)
          3. Look for: complex vocabulary, advanced grammar structures, AND overly basic words
          4. Use find_alternative_words with ALL inappropriate words: find_alternative_words(words="complex_word, too_simple_word", target_level="B2", context="full text context")
          5. Choose appropriate B2-level alternatives for each word
          6. Apply comprehensive changes: word replacement, sentence structure enhancement, grammar sophistication
          7. **Ensure the final text tells a complete, sophisticated story**
          8. **Prioritize story completeness and sophistication over grammatical perfection**
          
        - **Tool Usage Example:**
        Identify words that don't match B2 level (too difficult OR too simple):
        
        <example>
        Original Text: "The sophisticated design was big and the elaborate system was small."
        
        Step 1: Identify inappropriate words
        → find_alternative_words(words="sophisticated, elaborate, big, small", target_level="B2", context="...")
        
        Step 2: Get alternatives
        → Result: {"sophisticated": ["complex", "simple", "basic"], "elaborate": ["detailed", "careful"], "big": ["large", "huge"], "small": ["tiny", "little"]}
        
        Step 3: Choose appropriate B2 alternatives
        → "The intricate design was substantial and the comprehensive system was diminutive."
        </example>
        
        Use these alternatives as reference - you don't have to use them exactly. Choose what fits best for B2 level or create your own appropriate alternatives.
    </rag_human_addition>"""
        
        system_prompt_text += rag_system_addition
        human_prompt_text += rag_human_addition

    # --- Final Assembly ---
    final_human_prompt = human_prompt_text + """

        ---

        <original_text_to_rewrite>

            {text}

        </original_text_to_rewrite>

        ---

        Now, rewrite the content within the <original_text_to_rewrite> tag based on all the instructions provided.
        Return only the B2 level English text. No formatting, no labels, no explanations.
        """

    # Conditionally add the agent_scratchpad placeholder ONLY for RAG agents
    if use_rag:
        final_human_prompt += "\n        {agent_scratchpad}"

    messages = [
        ("system", system_prompt_text),
        ("human", final_human_prompt)
    ]
    
    return ChatPromptTemplate.from_messages(messages)