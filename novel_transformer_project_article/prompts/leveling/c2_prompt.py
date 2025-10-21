from langchain_core.prompts import ChatPromptTemplate

def get_prompt(use_rag: bool = False, is_feedback_generated: bool = False) -> ChatPromptTemplate:
    """
    Builds and returns the complete ChatPromptTemplate for the C2 level,
    dynamically including RAG instructions and feedback if required.
    """
    
    # --- Define Content ---
    level_definition = """
    <level_definition>
        **C2 Level Definition (Proficient User - Mastery):**
        Can understand with ease virtually everything heard or read. Can summarize information from different spoken and written sources, reconstructing arguments and accounts in a coherent presentation. Can express him/herself spontaneously, very fluently and precisely, differentiating finer shades of meaning even in the most complex situations.
    </level_definition>"""

    level_guidelines = """
    <level_guidelines>
        **⚠️ CRITICAL: COMPLEXITY AND READABILITY ⚠️**
        - Crucially, a text's difficulty and complexity should never come at the expense of its readability.

        **⚠️ CRITICAL: LENGTH AND CONCISENESS ⚠️**
        - **MAINTAIN REASONABLE LENGTH**: Aim for 1.5-2.0x the original text length at most
        - **AVOID UNNECESSARY ELABORATION**: Don't add excessive descriptive words just for complexity
        - **PRIORITIZE QUALITY OVER QUANTITY**: Sophisticated does not mean verbose
        - **PRESERVE ORIGINAL PACING**: Don't slow down the story with overly long descriptions
        - **CRITICAL: If the chunk is considered as a title, do not level it.**

        **Guidelines for C2 Level:**
        - You must not create content that does not exist in the original work.
        - Convey emotions and situations with greater depth and nuance, without sacrificing readability.
        - Use all tenses with absolute mastery and native-like fluency
        - Use highly sophisticated sentences with complex structure and elegant flow
        - Use extensive vocabulary with absolute precision and subtle nuance
        - Use advanced literary techniques and stylistic devices
        - Use complex grammatical structures with perfect control
        - Use sophisticated discourse markers and cohesive devices
        - Use advanced rhetorical techniques and persuasive language
        - Use idiomatic expressions and cultural references with mastery
        - Use formal and informal registers with perfect appropriateness
        - Use subtle humor, irony, and wit when appropriate
        - Use varied sentence patterns for sophisticated rhythm and emphasis
        - Focus on nuanced meaning and sophisticated expression
        - Use context-appropriate register and tone with perfect judgment
        - Use advanced lexical variety and precision
    </level_guidelines>"""
    
    critical_instructions = """
    <critical_instructions>
        ⚠️  CRITICAL: ABSOLUTELY NO EXPLANATIONS, LABELS, OR EXTRA TEXT ⚠️
        - DO NOT add any text like "C2 Level:", "Here is the C2 version:", or any explanations
        - DO NOT add any introductions, commentary, or labels and tags
        - ONLY return the C2 level text
        - START DIRECTLY with the C2 text
        - END DIRECTLY with the C2 text

        ⚠️  **CRITICAL: OUTPUT FORMAT EXAMPLE** ⚠️
        <example>
        Expected Output:
        Amidst the verdant expanse of the municipal park, the inquisitive young boy serendipitously encountered an extraordinarily magnificent specimen of the canine species, whose presence seemed to radiate an almost ethereal quality that transcended the ordinary. Overwhelmed by an unprecedented surge of exhilaration that coursed through his very being with an intensity that defied description, he found himself utterly captivated, for his lifelong passion for the animal kingdom had never diminished in its fervor. The remarkably amiable four-legged companion responded with an exuberant display of tail-wagging enthusiasm upon the child's approach, thereby establishing an instantaneous and profound connection between the two beings that transcended mere physical proximity and spoke to the fundamental bond between humans and animals.

        NOT like this:
        "C2 Level Version: Here is the adapted text..."
        "I have adapted the text to C2 level..."
        "Based on the original, here is the C2 version..."
        </example>

        ⚠️  **CRITICAL: OUTPUT ONLY IN ENGLISH** ⚠️
        - MUST respond ONLY in English language
        - DO NOT use any other language (Korean, Chinese, Japanese, etc.)
        - ALL text must be in English only

        ⚠️  **CRITICAL: STAY TRUE TO THE ORIGINAL TEXT** ⚠️
        - DO NOT create new characters, events, or plot points
        - DO NOT change the basic meaning or sequence of events
        - ONLY elevate the language to native-speaker level sophistication while keeping the same story
        - Use the most advanced vocabulary and complex structures appropriate for C2 level
    </critical_instructions>"""

    human_prompt_persona = """You are an expert English teacher specializing in adapting texts for C2 level learners (Proficient User - Mastery)."""

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
        - ONLY provide the pure C2 level story text - nothing else
        - Act as if this is your first and only attempt at the C2 level text
        ---
    </feedback_instructions>"""

    human_prompt_context_and_instructions = """
    <human_prompt_context_and_instructions>

        Please rewrite this English text for C2 level, considering the following context:

        ## 🎯 PRIORITY INSTRUCTIONS 🎯
        
        **If feedback was provided:**
        - **FEEDBACK IMPLEMENTATION IS YOUR ABSOLUTE TOP PRIORITY**
        - **FEEDBACK OVERRIDES ALL OTHER INSTRUCTIONS** when there's a conflict
        - **ADDRESS EVERY SINGLE FEEDBACK POINT** before considering other improvements
        - **SUCCESS DEPENDS ON PRECISE FEEDBACK IMPLEMENTATION**
        
        **If no feedback:**
        - Follow all guidelines and instructions below
        - Focus on creating high-quality C2 level text
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
        - Make sure your C2 text connects naturally with the previous chunk
        - **Focus on telling a complete, sophisticated story rather than just isolated concepts**
        - Use highly sophisticated sentences with complex structure and elegant flow
        - **Prioritize story completeness and sophistication over grammatical perfection**
        - **It's better to be slightly grammatically incorrect but tell a sophisticated story than be perfect but basic**
        - Break complex scenes into masterfully structured, sophisticated sentences that continue the story
        - Use extensive vocabulary with absolute precision and subtle nuance while preserving the story flow
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
        You have access to the find_alternative_words tool. Use it for comprehensive C2 level adaptation, not just word replacement.
        - Identify ALL words that don't match C2 level: both too difficult AND too simple words
        - Look for: complex vocabulary, advanced grammar, AND overly basic words that need upgrading
        - Call find_alternative_words(words="word1, word2, word3", target_level="C2", context="actual surrounding text from the passage")
        - This tool provides C2-level alternatives as reference - use them as guidance, not strict requirements
    </rag_system_addition>"""

        rag_human_addition = """<rag_human_addition>

        - **IMPORTANT**: For comprehensive C2 level adaptation:
          1. Analyze the ENTIRE text for C2 level appropriateness
          2. Identify words that are: TOO DIFFICULT (above C2) AND TOO SIMPLE (below C2)
          3. Look for: complex vocabulary, advanced grammar structures, AND overly basic words
          4. Use find_alternative_words with ALL inappropriate words: find_alternative_words(words="complex_word, too_simple_word", target_level="C2", context="full text context")
          5. Choose appropriate C2-level alternatives for each word
          6. Apply comprehensive changes: word replacement, sentence structure enhancement, grammar sophistication
          7. **Ensure the final text tells a complete, sophisticated story**
          8. **Prioritize story completeness and sophistication over grammatical perfection**
          
        - **Tool Usage Example:**
        Identify words that don't match C2 level (too difficult OR too simple):
        
        <example>
        Original Text: "The sophisticated design was big and the elaborate system was small."
        
        Step 1: Identify inappropriate words
        → find_alternative_words(words="sophisticated, elaborate, big, small", target_level="C2", context="...")
        
        Step 2: Get alternatives
        → Result: {"sophisticated": ["complex", "simple", "basic"], "elaborate": ["detailed", "careful"], "big": ["large", "huge"], "small": ["tiny", "little"]}
        
        Step 3: Choose appropriate C2 alternatives
        → "The intricate design was substantial and the comprehensive system was diminutive."
        </example>
        
        Use these alternatives as reference - you don't have to use them exactly. Choose what fits best for C2 level or create your own appropriate alternatives.
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
        Return only the C2 level English text. No formatting, no labels, no explanations.
        """

    # Conditionally add the agent_scratchpad placeholder ONLY for RAG agents
    if use_rag:
        final_human_prompt += "\n        {agent_scratchpad}"

    messages = [
        ("system", system_prompt_text),
        ("human", final_human_prompt)
    ]
    
    return ChatPromptTemplate.from_messages(messages)