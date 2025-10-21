from langchain_core.prompts import ChatPromptTemplate

def get_prompt(use_rag: bool = False, is_feedback_generated: bool = False) -> ChatPromptTemplate:
    """
    Builds and returns the complete ChatPromptTemplate for the C1 level,
    dynamically including RAG instructions and feedback if required.
    """
    
    # --- Define Content ---
    level_definition = """
    <level_definition>
        **C1 Level Definition (Proficient User - Advanced):**
        Can understand a wide range of demanding, longer texts, and recognise implicit meaning. Can express him/herself fluently and spontaneously without much obvious searching for expressions. Can use language flexibly and effectively for social, academic and professional purposes. Can produce clear, well-structured, detailed text on complex subjects, showing controlled use of organisational patterns, connectors and cohesive devices.
    </level_definition>"""

    level_guidelines = """
    <level_guidelines>
        **⚠️ CRITICAL: COMPLEXITY AND READABILITY ⚠️**
        - Crucially, a text's difficulty and complexity should never come at the expense of its readability.

        **⚠️ CRITICAL: LENGTH AND CONCISENESS ⚠️**
        - **MAINTAIN REASONABLE LENGTH**: Aim for 1.3-1.8x the original text length at most
        - **AVOID UNNECESSARY ELABORATION**: Don't add excessive descriptive words just for complexity
        - **PRIORITIZE QUALITY OVER QUANTITY**: Advanced does not mean verbose
        - **PRESERVE ORIGINAL PACING**: Don't slow down the story with overly long descriptions
        - **CRITICAL: If the chunk is considered as a title, do not level it.**

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
    </level_guidelines>"""
    
    critical_instructions = """
    <critical_instructions>
        ⚠️  CRITICAL: ABSOLUTELY NO EXPLANATIONS, LABELS, OR EXTRA TEXT ⚠️
        - DO NOT add any text like "C1 Level:", "Here is the C1 version:", or any explanations
        - DO NOT add any introductions, commentary, or labels and tags
        - ONLY return the C1 level text
        - START DIRECTLY with the C1 text
        - END DIRECTLY with the C1 text

        ⚠️  **CRITICAL: OUTPUT FORMAT EXAMPLE** ⚠️
        <example>
        Expected Output:
        He ladies of Longbourn soon waited on those of Netherfield. The visit was returned in due form. Miss Bennet’s pleasing manners grew on the good-will of Mrs. Hurst and Miss Bingley; and though the mother was found to be intolerable, and the younger sisters not worth speaking to, a wish of being better acquainted with them was expressed towards the two eldest.
        

        NOT like this:
        "C1 Level Version: Here is the adapted text..."
        "I have adapted the text to C1 level..."
        "Based on the original, here is the C1 version..."
        </example>

        ⚠️  **CRITICAL: OUTPUT ONLY IN ENGLISH** ⚠️
        - MUST respond ONLY in English language
        - DO NOT use any other language (Korean, Chinese, Japanese, etc.)
        - ALL text must be in English only

        ⚠️  **CRITICAL: STAY TRUE TO THE ORIGINAL TEXT** ⚠️
        - DO NOT create new characters, events, or plot points
        - DO NOT change the basic meaning or sequence of events
        - ONLY elevate the language to C1 level while keeping the same story
        - If something cannot be expressed at C1 level, use the most appropriate sophisticated words
    </critical_instructions>"""

    human_prompt_persona = """You are an expert English teacher specializing in adapting texts for C1 level learners (Proficient User - Advanced)."""

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
        - ONLY provide the pure C1 level story text - nothing else
        - Act as if this is your first and only attempt at the C1 level text
        ---
    </feedback_instructions>"""

    human_prompt_context_and_instructions = """
    <human_prompt_context_and_instructions>

        Please rewrite this English text for C1 level, considering the following context:

        ## 🎯 PRIORITY INSTRUCTIONS 🎯
        
        **If feedback was provided:**
        - **FEEDBACK IMPLEMENTATION IS YOUR ABSOLUTE TOP PRIORITY**
        - **FEEDBACK OVERRIDES ALL OTHER INSTRUCTIONS** when there's a conflict
        - **ADDRESS EVERY SINGLE FEEDBACK POINT** before considering other improvements
        - **SUCCESS DEPENDS ON PRECISE FEEDBACK IMPLEMENTATION**
        
        **If no feedback:**
        - Follow all guidelines and instructions below
        - Focus on creating high-quality C1 level text
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
        - Make sure your C1 text connects naturally with the previous chunk
        - **Focus on telling a complete, sophisticated story rather than just isolated concepts**
        - Use highly complex sentences with sophisticated structure and natural flow
        - **Prioritize story completeness and sophistication over grammatical perfection**
        - **It's better to be slightly grammatically incorrect but tell a sophisticated story than be perfect but basic**
        - Break complex scenes into elegantly structured, sophisticated sentences that continue the story
        - Use extensive vocabulary with precise nuance and subtlety while preserving the story flow
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
        You have access to the find_alternative_words tool. Use it for comprehensive C1 level adaptation, not just word replacement.
        - Identify ALL words that don't match C1 level: both too difficult AND too simple words
        - Look for: complex vocabulary, advanced grammar, AND overly basic words that need upgrading
        - Call find_alternative_words(words="word1, word2, word3", target_level="C1", context="actual surrounding text from the passage")
        - This tool provides C1-level alternatives as reference - use them as guidance, not strict requirements
    </rag_system_addition>"""

        rag_human_addition = """<rag_human_addition>

        - **IMPORTANT**: For comprehensive C1 level adaptation:
          1. Analyze the ENTIRE text for C1 level appropriateness
          2. Identify words that are: TOO DIFFICULT (above C1) AND TOO SIMPLE (below C1)
          3. Look for: complex vocabulary, advanced grammar structures, AND overly basic words
          4. Use find_alternative_words with ALL inappropriate words: find_alternative_words(words="complex_word, too_simple_word", target_level="C1", context="full text context")
          5. Choose appropriate C1-level alternatives for each word
          6. Apply comprehensive changes: word replacement, sentence structure enhancement, grammar sophistication
          7. **Ensure the final text tells a complete, sophisticated story**
          8. **Prioritize story completeness and sophistication over grammatical perfection**
          
        - **Tool Usage Example:**
        Identify words that don't match C1 level (too difficult OR too simple):
        
        <example>
        Original Text: "The sophisticated design was big and the elaborate system was small."
        
        Step 1: Identify inappropriate words
        → find_alternative_words(words="sophisticated, elaborate, big, small", target_level="C1", context="...")
        
        Step 2: Get alternatives
        → Result: {"sophisticated": ["complex", "simple", "basic"], "elaborate": ["detailed", "careful"], "big": ["large", "huge"], "small": ["tiny", "little"]}
        
        Step 3: Choose appropriate C1 alternatives
        → "The intricate design was substantial and the comprehensive system was diminutive."
        </example>
        
        Use these alternatives as reference - you don't have to use them exactly. Choose what fits best for C1 level or create your own appropriate alternatives.
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
        Return only the C1 level English text. No formatting, no labels, no explanations.
        """

    # Conditionally add the agent_scratchpad placeholder ONLY for RAG agents
    if use_rag:
        final_human_prompt += "\n        {agent_scratchpad}"

    messages = [
        ("system", system_prompt_text),
        ("human", final_human_prompt)
    ]
    
    return ChatPromptTemplate.from_messages(messages)