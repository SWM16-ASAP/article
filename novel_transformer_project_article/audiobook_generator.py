import json
import os
import boto3
from pathlib import Path
from botocore.exceptions import NoCredentialsError, ClientError

# Import the core function from your existing utility file
from novel_transformer_project.utils.audio_generator import create_chapter_audiobook, initialize_tts_client, clean_text
from novel_transformer_project.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- S3 Helper Functions ---

def download_from_s3(bucket: str, key: str, local_path: str) -> bool:
    """Downloads a file from S3 to a local path."""
    s3_client = boto3.client("s3")
    try:
        logger.info(f"Downloading s3://{bucket}/{key} to {local_path}...")
        s3_client.download_file(bucket, key, local_path)
        logger.info("Download complete.")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.error(f"File not found in S3: s3://{bucket}/{key}")
        else:
            logger.error(f"Failed to download from S3: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during download: {e}")
        return False

def upload_directory_to_s3(local_directory: str, bucket: str, s3_prefix: str):
    """Uploads the contents of a local directory to an S3 prefix, preserving structure."""
    s3_client = boto3.client("s3")
    try:
        logger.info(f"Uploading contents of {local_directory} to s3://{bucket}/{s3_prefix}")
        for root, _, files in os.walk(local_directory):
            for filename in files:
                local_path = os.path.join(root, filename)
                # Create a relative path to preserve the folder structure in S3
                relative_path = os.path.relpath(local_path, local_directory)
                s3_key = os.path.join(s3_prefix, relative_path)
                
                logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}...")
                s3_client.upload_file(local_path, bucket, s3_key)
        logger.info("All files uploaded successfully.")
        return True
    except NoCredentialsError:
        logger.error("AWS credentials not found. Cannot upload to S3.")
        return False
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False

# --- Main Orchestration Logic ---

def run_audio_generation(input_json_path: str, local_base_dir: str) -> dict | None:
    """
    Main orchestration function to generate audiobooks.
    
    This function reads the final leveled JSON file and generates an MP3
    and a timings file for every chapter of every CEFR level.
    """
    # Initialize the Google TTS client ONCE for the entire job.
    try:
        tts_client = initialize_tts_client()
    except Exception as e:
        logger.error(f"Failed to initialize Google TTS client: {e}", exc_info=True)
        return None

    # 1. Read the final JSON data file
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            leveled_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {input_json_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error: Could not parse JSON from {input_json_path}")
        return None

    # Handle both 'id' and 'novel_id' fields for backward compatibility
    content_id = leveled_data.get("id")
    content_type = leveled_data.get("content_type", "literature") # Default to literature
    logger.info(f"Starting audiobook generation for content: {content_id} (Type: {content_type})")

    # Define a base directory for all audio output for this content
    audio_output_dir = os.path.join(local_base_dir, content_type, content_id)
    
    # 2. Iterate through each CEFR level in the data
    # text level 별로 불러온다.
    for level_result in leveled_data.get("leveled_results", []):
        text_level = level_result.get("textLevel")
        logger.info(f"--- Processing Level: {text_level} ---")

        # Create a specific directory for this level's audio
        level_specific_dir = os.path.join(audio_output_dir, text_level)
        Path(level_specific_dir).mkdir(parents=True, exist_ok=True)

        # 3. Iterate through each chapter for the current level
        for chapter_data in level_result.get("chapters", []):
            chapter_num = chapter_data.get("chapterNum")
            logger.info(f"  - Generating audio for Chapter {chapter_num}...")

            # 4. Call the utility function to do the actual work
            # This is the entry point into your existing audio_generator.py
            try:
                mp3_path, timings_path = create_chapter_audiobook(
                    tts_client=tts_client, # Pass the initialized client
                    leveled_chapter_data=chapter_data,
                    cefr_level=text_level,
                    output_dir=level_specific_dir
                )
                logger.info(f"    Successfully created: {os.path.basename(mp3_path)}")
                logger.info(f"    Successfully created: {os.path.basename(timings_path)}")
            except Exception as e:
                # Make the system resilient: if one chapter fails, log it and continue
                logger.error(f"    Failed to generate audio for Chapter {chapter_num} at level {text_level}. Error: {e}", exc_info=True)
                continue
    
    logger.info("All audio generation tasks complete.")
    return {
        "output_path": audio_output_dir,
        "content_type": content_type,
        "content_id": content_id
    }


if __name__ == '__main__':
    # This block is the main entry point when the container starts.
    # It reads configuration from environment variables set by ECS.
    
    # Get configuration from environment variables
    input_bucket = os.environ.get("INPUT_JSON_S3_BUCKET")
    input_key = os.environ.get("INPUT_JSON_S3_KEY")
    output_bucket = os.environ.get("AUDIO_OUTPUT_S3_BUCKET")

    if not all([input_bucket, input_key, output_bucket]):
        logger.error("Missing one or more required environment variables:")
        logger.error(f"  - INPUT_JSON_S3_BUCKET: {input_bucket}")
        logger.error(f"  - INPUT_JSON_S3_KEY: {input_key}")
        logger.error(f"  - AUDIO_OUTPUT_S3_BUCKET: {output_bucket}")
        exit(1) # Exit with an error code

    # Create a temporary local directory
    local_temp_dir = "/tmp/audio_processing"
    Path(local_temp_dir).mkdir(parents=True, exist_ok=True)

    # Make the local filename dynamic based on the S3 key
    json_filename = os.path.basename(input_key)
    
    if not json_filename:
        logger.warning(f"Could not determine filename from S3 key '{input_key}'. Defaulting to 'leveled_output.json'.")
        json_filename = "leveled_output.json"
    local_json_path = os.path.join(local_temp_dir, json_filename)

    # Download the source JSON from S3
    if download_from_s3(input_bucket, input_key, local_json_path):
        # Run the main generation process
        generation_result = run_audio_generation(
            input_json_path=local_json_path,
            local_base_dir=local_temp_dir
        )

        # If generation was successful, upload the results
        if generation_result and generation_result.get("output_path"):
            # The S3 prefix will be content_type/novel_id
            s3_upload_prefix = os.path.join(generation_result["content_type"], generation_result["content_id"])
            local_audio_path = generation_result["output_path"]
            
            logger.info(f"Uploading results to S3 prefix: {s3_upload_prefix}")
            upload_directory_to_s3(local_audio_path, output_bucket, s3_upload_prefix)
