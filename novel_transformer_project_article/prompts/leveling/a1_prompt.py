from langchain_core.prompts import ChatPromptTemplate

def get_prompt(use_rag: bool = False, is_feedback_generated: bool = False) -> ChatPromptTemplate:
    """
    Builds and returns the complete ChatPromptTemplate for the A1 level,
    dynamically including RAG instructions and feedback if required.
    """
    
    # --- Define Content ---
    level_definition = """
    <level_definition>
        **A1 Level Definition (Basic User - Beginner):**
        Can understand and use familiar everyday expressions and very basic phrases aimed at the satisfaction of needs of a concrete type. Can introduce him/herself and others and can ask and answer questions about personal details such as where he/she lives, people he/she knows and things he/she has. Can interact in a simple way provided the other person talks slowly and clearly and is prepared to help.
    </level_definition>"""

    level_guidelines = """
    <level_guidelines>
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
    </level_guidelines>"""
    
    critical_instructions = """
    <critical_instructions>
        ⚠️  CRITICAL: ABSOLUTELY NO EXPLANATIONS, LABELS, OR EXTRA TEXT ⚠️
        - DO NOT add any text like "A1 Level:", "Here is the A1 version:", or any explanations
        - DO NOT add any introductions, commentary, or labels and tags
        - ONLY return the A1 level text
        - START DIRECTLY with the A1 text
        - END DIRECTLY with the A1 text

        ⚠️  **CRITICAL: OUTPUT FORMAT EXAMPLE** ⚠️
        <example>
        Expected Output:
        The boy sees a dog. The dog is big. The boy is happy. He wants to play with the dog.

        NOT like this:
        "A1 Level Version: Here is the adapted text..."
        "I have adapted the text to A1 level..."
        "Based on the original, here is the A1 version..."
        </example>

        ⚠️  **CRITICAL: OUTPUT ONLY IN ENGLISH** ⚠️
        - MUST respond ONLY in English language
        - DO NOT use any other language (Korean, Chinese, Japanese, etc.)
        - ALL text must be in English only

        ⚠️  **CRITICAL: STAY TRUE TO THE ORIGINAL TEXT** ⚠️
        - DO NOT create new characters, events, or plot points
        - DO NOT change the basic meaning or sequence of events
        - ONLY simplify the language while keeping the same story
        - If something cannot be expressed at A1 level, use the simplest possible words
    </critical_instructions>"""

    human_prompt_persona = """You are an expert English teacher specializing in adapting texts for A1 level learners (Basic User - Beginner)."""

    feedback_instructions = """
    <feedback_instructions>
        ---
        
        🚨 **CRITICAL FEEDBACK ANALYSIS REQUIRED** 🚨
        
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
        - ONLY provide the pure A1 level story text - nothing else
        - Act as if this is your first and only attempt at the A1 level text
        ---
    </feedback_instructions>"""

    human_prompt_context_and_instructions = """
    <human_prompt_context_and_instructions>

        Please rewrite this English text for A1 level, considering the following context:

        ## 🎯 PRIORITY INSTRUCTIONS 🎯
        
        **If feedback was provided:**
        - **FEEDBACK IMPLEMENTATION IS YOUR ABSOLUTE TOP PRIORITY**
        - **FEEDBACK OVERRIDES ALL OTHER INSTRUCTIONS** when there's a conflict
        - **ADDRESS EVERY SINGLE FEEDBACK POINT** before considering other improvements
        - **SUCCESS DEPENDS ON PRECISE FEEDBACK IMPLEMENTATION**
        
        **If no feedback:**
        - Follow all guidelines and instructions below
        - Focus on creating high-quality A1 level text
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
        - Make sure your A1 text connects naturally with the previous chunk
        - **Focus on telling a complete, simple story rather than just isolated concepts**
        - Use short, simple sentences (5-8 words) that tell the story clearly
        - **Prioritize story completeness over grammatical correctness**
        - **It's better to be grammatically incorrect but tell a clear story than be correct and confusing**
        - Break complex scenes into simple sentences that continue the story
        - Use the most basic words possible while preserving the story flow
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
        You have access to the find_alternative_words tool. Use it for comprehensive A1 level adaptation, not just word replacement.
        - Identify ALL words that don't match A1 level: both too difficult AND too simple words
        - Look for: complex vocabulary, advanced grammar, AND overly basic words that need upgrading
        - Call find_alternative_words(words="word1, word2, word3", target_level="A1", context="actual surrounding text from the passage")
        - This tool provides A1-level alternatives as reference - use them as guidance, not strict requirements
    </rag_system_addition>"""

        rag_human_addition = """<rag_human_addition>

        - **IMPORTANT**: For comprehensive A1 level adaptation:
          1. Analyze the ENTIRE text for A1 level appropriateness
          2. Identify words that are: TOO DIFFICULT (above A1) AND TOO SIMPLE (below A1)
          3. Look for: complex vocabulary, advanced grammar structures, AND overly basic words
          4. Use find_alternative_words with ALL inappropriate words: find_alternative_words(words="complex_word, too_simple_word", target_level="A1", context="full text context")
          5. Choose appropriate A1-level alternatives for each word
          6. Apply comprehensive changes: word replacement, sentence structure simplification, grammar adjustment
          7. **Ensure the final text tells a complete, simple story**
          8. **Prioritize story completeness over grammatical correctness**
          
        - **Tool Usage Example:**
        Identify words that don't match A1 level (too difficult OR too simple):
        
        <example>
        Original Text: "The sophisticated design was big and the elaborate system was small."
        
        Step 1: Identify inappropriate words
        → find_alternative_words(words="sophisticated, elaborate, big, small", target_level="A1", context="...")
        
        Step 2: Get alternatives
        → Result: {"sophisticated": ["complex", "simple", "basic"], "elaborate": ["detailed", "careful"], "big": ["large", "huge"], "small": ["tiny", "little"]}
        
        Step 3: Choose appropriate A1 alternatives
        → "The simple design was big. The careful system was small."
        </example>
        
        Use these alternatives as reference - you don't have to use them exactly. Choose what fits best for A1 level or create your own appropriate alternatives.
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
        Return only the A1 level English text. No formatting, no labels, no explanations.
        """

    # Conditionally add the agent_scratchpad placeholder ONLY for RAG agents
    if use_rag:
        final_human_prompt += "\n        {agent_scratchpad}"

    messages = [
        ("system", system_prompt_text),
        ("human", final_human_prompt)
    ]
    
    return ChatPromptTemplate.from_messages(messages)