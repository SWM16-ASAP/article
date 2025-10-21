"""
JSON Handler Module for Text Leveling System

This module handles JSON input parsing, validation, and error handling
for the text leveling pipeline according to PROJECT_OVERVIEW specifications.
"""

import json
import logging
import boto3
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import uuid

# Configure logging
from .logging_config import get_logger
logger = get_logger(__name__)

@dataclass
class NovelData:
    """Structured representation of novel input data"""
    title: str
    author: str
    text: str
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

class JSONValidationError(Exception):
    """Custom exception for JSON validation errors"""
    pass

class JSONHandler:
    """
    Handles JSON file operations for the text leveling system
    
    Features:
    - S3 JSON file retrieval
    - JSON parsing and validation
    - Text length and format validation
    - Error handling and logging
    """
    
    def __init__(self, s3_bucket: str, aws_region: str = "us-east-1"): 
        """
        Initialize JSON Handler
        
        Args:
            s3_bucket: S3 bucket name containing input JSON files
            aws_region: AWS region for S3 client
        """
        self.s3_bucket = s3_bucket
        self.aws_region = aws_region
        self.s3_client = boto3.client('s3', region_name=aws_region)
        
        # Validation constraints
        self.MIN_TEXT_LENGTH = 100
        self.MAX_TEXT_LENGTH = 500_000
        self.MAX_TITLE_LENGTH = 200
        self.MAX_AUTHOR_LENGTH = 100
        
    def download_json_from_s3(self, json_file_key: str) -> Dict[str, Any]:
        """
        Download and parse JSON file from S3
        
        Args:
            json_file_key: S3 object key for the JSON file
            
        Returns:
            Parsed JSON data as dictionary
            
        Raises:
            JSONValidationError: If file cannot be downloaded or parsed
        """
        try:
            logger.info(f"Downloading JSON file: {json_file_key} from bucket: {self.s3_bucket}")
            
            # Download file from S3
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=json_file_key
            )
            
            # Read and decode content
            content = response['Body'].read().decode('utf-8')
            logger.info(f"Successfully downloaded {len(content)} characters from S3")
            
            # Parse JSON
            json_data = json.loads(content)
            logger.info("JSON parsing successful")
            
            return json_data
            
        except self.s3_client.exceptions.NoSuchKey:
            error_msg = f"JSON file not found in S3: {json_file_key}"
            logger.error(error_msg)
            raise JSONValidationError(error_msg)
            
        except self.s3_client.exceptions.NoSuchBucket:
            error_msg = f"S3 bucket not found: {self.s3_bucket}"
            logger.error(error_msg)
            raise JSONValidationError(error_msg)
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {str(e)}"
            logger.error(error_msg)
            raise JSONValidationError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error downloading JSON: {str(e)}"
            logger.error(error_msg)
            raise JSONValidationError(error_msg)
    
    def validate_json_structure(self, json_data: Dict[str, Any]) -> None:
        """
        Validate JSON structure and required fields
        
        Args:
            json_data: Parsed JSON data to validate
            
        Raises:
            JSONValidationError: If validation fails
        """
        logger.info("Starting JSON structure validation")
        
        # Check required fields
        required_fields = ["title", "author", "text"]
        missing_fields = [field for field in required_fields if field not in json_data]
        
        if missing_fields:
            error_msg = f"Missing required fields: {missing_fields}"
            logger.error(error_msg)
            raise JSONValidationError(error_msg)
        
        # Check field types
        if not isinstance(json_data["title"], str):
            raise JSONValidationError("Field 'title' must be a string")
            
        if not isinstance(json_data["author"], str):
            raise JSONValidationError("Field 'author' must be a string")
            
        if not isinstance(json_data["text"], str):
            raise JSONValidationError("Field 'text' must be a string")
        
        logger.info("JSON structure validation passed")
    
    def validate_content(self, json_data: Dict[str, Any]) -> None:
        """
        Validate content length and format
        
        Args:
            json_data: Parsed JSON data to validate
            
        Raises:
            JSONValidationError: If content validation fails
        """
        logger.info("Starting content validation")
        
        title = json_data["title"].strip()
        author = json_data["author"].strip()
        text = json_data["text"].strip()
        
        # Validate title
        if not title:
            raise JSONValidationError("Title cannot be empty")
        if len(title) > self.MAX_TITLE_LENGTH:
            raise JSONValidationError(f"Title too long (max {self.MAX_TITLE_LENGTH} characters)")
        
        # Validate author
        if not author:
            raise JSONValidationError("Author cannot be empty")
        if len(author) > self.MAX_AUTHOR_LENGTH:
            raise JSONValidationError(f"Author name too long (max {self.MAX_AUTHOR_LENGTH} characters)")
        
        # Validate text content
        if not text:
            raise JSONValidationError("Text content cannot be empty")
        if len(text) < self.MIN_TEXT_LENGTH:
            raise JSONValidationError(f"Text too short (minimum {self.MIN_TEXT_LENGTH} characters)")
        if len(text) > self.MAX_TEXT_LENGTH:
            raise JSONValidationError(f"Text too long (maximum {self.MAX_TEXT_LENGTH} characters)")
        
        # Check for valid characters (basic Korean/English text validation)
        if not re.search(r'[가-힣a-zA-Z]', text):
            raise JSONValidationError("Text must contain valid Korean or English characters")
        
        logger.info(f"Content validation passed - Title: '{title}', Author: '{author}', Text length: {len(text)} characters")
    
    def parse_and_validate(self, json_file_key: str) -> NovelData:
        """
        Complete JSON parsing and validation pipeline
        
        Args:
            json_file_key: S3 object key for the JSON file
            
        Returns:
            NovelData object with validated content
            
        Raises:
            JSONValidationError: If any validation step fails
        """
        logger.info(f"Starting JSON processing pipeline for: {json_file_key}")
        
        # Step 1: Download and parse JSON
        json_data = self.download_json_from_s3(json_file_key)
        
        # Step 2: Validate structure
        self.validate_json_structure(json_data)
        
        # Step 3: Validate content
        self.validate_content(json_data)
        
        # Step 4: Create structured data object
        novel_data = NovelData(
            title=json_data["title"].strip(),
            author=json_data["author"].strip(),
            text=json_data["text"].strip()
        )
        
        logger.info(f"JSON processing completed successfully - Generated ID: {novel_data.id}")
        return novel_data
    
    def create_sample_json(self, title: str, author: str, text: str) -> Dict[str, Any]:
        """
        Create a sample JSON structure for testing
        
        Args:
            title: Novel title
            author: Author name
            text: Novel text content
            
        Returns:
            Dictionary in the expected JSON format
        """
        return {
            "title": title,
            "author": author,
            "text": text
        }
    
    def save_json_to_s3(self, data: Dict[str, Any], output_bucket: str, output_key: str) -> None:
        """
        Save JSON data to S3 output bucket
        
        Args:
            data: JSON data to save
            output_bucket: S3 bucket for output
            output_key: S3 key for output file
        """
        try:
            json_content = json.dumps(data, ensure_ascii=False, indent=2) # data를 json 형식으로 변환
            
            self.s3_client.put_object(
                Bucket=output_bucket,
                Key=output_key,
                Body=json_content.encode('utf-8'),
                ContentType='application/json'
            )
            
            logger.info(f"=== JSON 결과 저장 완료: s3://{output_bucket}/{output_key} ===")
            
        except Exception as e:
            error_msg = f"Failed to save JSON to S3: {str(e)}"
            logger.error(error_msg)
            raise JSONValidationError(error_msg)

