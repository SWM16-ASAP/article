from typing import List, TypedDict, Optional, Dict

from .utils.data_models import FolderData

# --- Data Structures for Final Output ---
# This structure mirrors the desired output JSON format.

class LeveledChunk(TypedDict):
    """Represents a single leveled text chunk with its number."""
    chunkNum: int
    isImage: bool
    chunkText: str
    description: Optional[str]

class LeveledChapter(TypedDict):
    """Represents a chapter containing multiple leveled chunks."""
    chapterNum: int
    chunks: List[LeveledChunk]

class LevelResult(TypedDict):
    """Represents the complete leveled text for a specific CEFR level."""
    textLevel: str  # e.g., "A1", "A2"
    chapters: List[LeveledChapter]

class ChapterMetadata(TypedDict):
    """Represents metadata for a chapter including title and summary."""
    chapterNum: int
    title: Optional[str]
    summary: str

class ModelUsage(TypedDict):
    """Represents usage metrics for a single model."""
    input_tokens: int
    output_tokens: int
    cost: float

# UsageMetrics is now a dictionary mapping model_id to its ModelUsage
UsageMetrics = Dict[str, ModelUsage]

# --- Main Graph State ---
class BookState(TypedDict):
    """Represents the complete state of the book transformation process."""

    full_text: str

    # --- Input Data (from JSON) ---
    id: str
    content_type: str
    title: Optional[str]
    author: Optional[str]
    tags: Optional[List[str]]
    target_language_code: str
    
    # --- Processing State ---
    original_text_level: str  # The evaluated level of the original text
    chapters: List[str]  # Original text split into chapters
    chapter_chunks: List[List[str]]  # Original chapters split into chunks [chapter_idx][chunk_idx]
    current_chapter_index: int
    current_chunk_index: int
    
    # --- Chapter Metadata ---
    chapter_metadata: List[ChapterMetadata]  # Metadata for each chapter (title, summary, level)
    
    # --- Final Output Data ---
    # This list will be populated during the workflow and directly used to build the final JSON.
    leveled_results: List[LevelResult]

    # article topic url
    origin_url: str

    # --- Usage Metrics ---
    usage_metrics: UsageMetrics # Changed from model_usage to usage_metrics for consistency with previous discussions

    cover_image_url: Optional[str]

    total_cost: float
