# Image to QnA

This example demonstrates how to build an intelligent Question and Answer (QnA) system for images using the GraSP framework. The system extracts text from images, generates thoughtful questions about the content, and then provides detailed answers to those questions.

> **Key Features**:
> `multimodal LLMs`, `text extraction`, `question generation`, `image processing`, `multi-step reasoning`

## Overview

The Image to QnA example is designed to:

- **Process image content**: Extract text and structured information from images using multimodal LLMs
- **Generate insightful questions**: Create diverse, challenging questions based on extracted content
- **Answer questions intelligently**: Provide detailed answers grounded in the image content
- **Handle multiple images**: Process collections of images as a cohesive document set
- **Support different question types**: Generate both objective and subjective questions requiring different types of reasoning

## Directory Contents

- `graph_config.yaml`: Configuration file defining the workflow for image processing, question generation, and answer generation
- `task_executor.py`: Python code implementing custom processors, conditions, and functions for the QnA pipeline

## How It Works

1. **Text Extraction**:
   - The system loads images from the HuggingFaceM4/Docmatix dataset
   - For each image, a multimodal LLM (Qwen VL 72B) extracts text content
   - Extracted text is stored in a structured format for further processing
   - Multiple images are processed sequentially through a loop mechanism

2. **Question Generation**:
   - Once all images are processed, the extracted text is combined into a document
   - The system checks if enough text was extracted to proceed
   - A large language model (Qwen3 32B) generates 3 diverse questions:
     - At least one objective question (specific answer)
     - At least one subjective question (analytical reasoning)
     - Each with supporting evidence from the documents

3. **Answer Generation**:
   - For each generated question:
     - The system initializes the answer generation process
     - The question is passed to an LLM along with the document content
     - The model provides both its thinking process and final answer
     - The answer loop continues until all questions are addressed

4. **Output Collection**:
   - The system assembles a comprehensive output including:
     - Original image references and extracted text
     - Generated questions with their types (objective/subjective)
     - Detailed answers and reasoning for each question