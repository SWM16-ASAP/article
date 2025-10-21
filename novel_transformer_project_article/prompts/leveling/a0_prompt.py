from langchain_core.prompts import ChatPromptTemplate

def get_prompt(use_rag: bool = False, is_feedback_generated: bool = False) -> ChatPromptTemplate:
    """
    Builds and returns the complete ChatPromptTemplate for the A0 level,
    dynamically including RAG instructions and feedback if required.
    """
    
    # --- Define Content ---
    level_definition = """
    <level_definition>
        **A0 Level Definition (A Simple Story for Beginners):**
        Can understand a very simple story with a clear plot. Knows the main characters and the main actions.
        The goal is to tell a complete story using the simplest English possible. The story must be easy to read and follow.
    </level_definition>"""


    level_guidelines = """
    <level_guidelines>
        **Writing Guidelines for A0 Level:**
        - **Simple and Clear Story**: The most important rule is to create a simple and clear story that is easy to follow.
        - **Sentence Structure**: Use short, complete sentences. Example: "The boy sees a dog."
        - **Grammar**: Focus on simplicity over correctness. It's better to be grammatically incorrect but simple than correct and complex.
        - **Tense**: Try to use simple present tense when possible, but grammar mistakes are acceptable if it makes the text simpler.
        - **Vocabulary**: Use only very common, everyday words.
        - **Deconstruct Complex Concepts**: If a concept is too difficult for a single A0 word (e.g., "apologize"), break it down into simple actions. For example, instead of "He apologized," write "He said, 'I am sorry.'".
        - **No Connectors**: Do not use 'and', 'but', 'so'. Each sentence should be one simple idea.
        - **Use Names for Clarity**: To avoid confusion, it is better to repeat names instead of using pronouns like 'he', 'she', or 'they'.
        - **Basic Descriptions**: Use only very basic adjectives like 'big', 'small', 'happy', or 'sad' when necessary for the story.
        - **Literal Language Only**: Do not use any metaphors, similes, or idioms. The text must be direct and literal.
    </level_guidelines>"""

    critical_instructions = """
    <critical_instructions>
        ⚠️  **CRITICAL: ABSOLUTELY NO EXPLANATIONS, LABELS, OR EXTRA TEXT** ⚠️
        - DO NOT add any text like "A0 Level:", "Here is the A0 version:", or any explanations
        - DO NOT add any introductions, commentary, or labels and tags
        - ONLY return the A0 level text
        - START DIRECTLY with the A0 text
        - END DIRECTLY with the A0 text

        ⚠️  **CRITICAL: OUTPUT FORMAT EXAMPLE** ⚠️
        <example>
        Expected Output:
        The boy sees a dog. The dog is big. The boy is happy. He wants to play with the dog.

        NOT like this:
        "A0 Level Version: Here is the adapted text..."
        "I have adapted the text to A0 level..."
        "Based on the original, here is the A0 version..."
        </example>

        ⚠️  **CRITICAL: OUTPUT ONLY IN ENGLISH** ⚠️
        - MUST respond ONLY in English language
        - DO NOT use any other language (Korean, Chinese, Japanese, etc.)
        - ALL text must be in English only

        ⚠️  **CRITICAL: STAY TRUE TO THE ORIGINAL TEXT** ⚠️
        - DO NOT create new characters, events, or plot points
        - DO NOT change the basic meaning or sequence of events
        - ONLY simplify the language while keeping the same story
        - If something cannot be expressed at A0 level, use the simplest possible words
    </critical_instructions>"""
    
    human_prompt_persona = """You are an expert English teacher specializing in adapting texts for A0 level learners (Pre-A1 - Breakthrough)."""

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
        - ONLY provide the pure A0 level story text - nothing else
        - Act as if this is your first and only attempt at the A0 level text
        ---
    </feedback_instructions>"""

    human_prompt_context_and_instructions = """
    <human_prompt_context_and_instructions>

        Please rewrite this English text for A0 level, considering the following context:

        ## 🎯 PRIORITY INSTRUCTIONS 🎯
        
        **If feedback was provided:**
        - **FEEDBACK IMPLEMENTATION IS YOUR ABSOLUTE TOP PRIORITY**
        - **FEEDBACK OVERRIDES ALL OTHER INSTRUCTIONS** when there's a conflict
        - **ADDRESS EVERY SINGLE FEEDBACK POINT** before considering other improvements
        - **SUCCESS DEPENDS ON PRECISE FEEDBACK IMPLEMENTATION**
        
        **If no feedback:**
        - Follow all guidelines and instructions below
        - Focus on creating high-quality A0 level text
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
        - Make sure your A0 text connects naturally with the previous chunk
        - **Focus on telling a complete, simple story rather than just isolated concepts**
        - Use short, complete sentences that tell the story clearly
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
        You have access to the find_alternative_words tool. Use it for comprehensive A0 level concept adaptation.
        - Identify ALL words that don't match A0 concept recognition level
        - Look for: complex vocabulary, advanced grammar, abstract concepts
        - Focus on finding concrete, observable, immediate concepts
        - Call find_alternative_words(words="word1, word2, word3", target_level="A0", context="actual surrounding text from the passage")
        - This tool provides A0-level alternatives as reference - prioritize concept comprehension over grammar
    </rag_system_addition>"""

        rag_human_addition = """<rag_human_addition>

        - **IMPORTANT**: For comprehensive A0 level adaptation:
          1. Analyze the ENTIRE text for A0 level appropriateness
          2. Identify words that are: TOO DIFFICULT (above A0) AND TOO SIMPLE (below A0)
          3. Look for: complex vocabulary, advanced grammar structures, AND overly basic words
          4. Use find_alternative_words with ALL inappropriate words: find_alternative_words(words="complex_word, too_simple_word", target_level="A0", context="full text context")
          5. Choose appropriate A0-level alternatives for each word
          6. Apply comprehensive changes: word replacement, break into simple complete sentences, remove grammar complexity
          7. **Ensure the final text tells a complete, simple story**
          8. **Prioritize story completeness over grammatical correctness**
          
        - **Tool Usage Example:**
        Identify words that don't match A0 level (too difficult OR too simple):
        
        Example: "The sophisticated design was big and the elaborate system was small."
        → find_alternative_words(words="sophisticated, elaborate, design, system", target_level="A0", context="...")
        → Result: {{\"sophisticated\": ["nice", "good"], "elaborate": ["big", "many"], "design": ["thing", "look"], "system": ["box", "thing"]}}
        
        Break into A0 story sentences: "The thing is good. The thing is big. The box is big. The box is small."
        Focus on telling a simple, complete story with concrete, observable actions.
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
        Return only the A0 level English text. No formatting, no labels, no explanations.
        """

    # Conditionally add the agent_scratchpad placeholder ONLY for RAG agents
    if use_rag:
        final_human_prompt += "\n        {agent_scratchpad}"

    messages = [
        ("system", system_prompt_text),
        ("human", final_human_prompt)
    ]
    
    # Let LangChain infer the input variables automatically from the final prompt string.
    # This is safer and cleaner than managing the list manually.
    return ChatPromptTemplate.from_messages(messages)