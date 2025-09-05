# Structured Output

This example demonstrates how to implement structured output from LLM responses within the GraSP framework. It showcases how to extract specific information from LLM responses in a standardized JSON format using structured output that can be reliably parsed and processed in downstream applications.

> **Key Features**:
> `structured JSON output`, `schema definition`, `post-processing`, `code taxonomy`, `response normalization`

## Overview

The Structured Output example is designed to:

- **Extract structured data**: Automatically respond in a well-defined schema for easy parsing and processing

## Directory Contents

- `task_executor.py`: Core functionality for processing and extracting structured data from LLM responses
- `graph_config.yaml`: Configuration file defining the workflow graph and structured output schema

## How It Works

1. **Data Source**:
   - The system loads code snippets from the "glaiveai/glaive-code-assistant-v2" HuggingFace dataset
   - Each entry contains a programming question that includes code

2. **Schema Definition**:
   - The `structured_output` section in the model configuration defines a schema with:
     - `category`: String field for the main category of the code snippet
     - `sub_category`: String field for a more specific categorization

3. **LLM Processing**:
   - The `generate_taxonomy` node sends the code snippet to the LLM
   - The LLM identifies the appropriate category and subcategory based on the code content
   - The response is formatted according to the defined schema

4. **Post-Processing**:
   - The `GenerateTaxonomyPostProcessor` extracts the structured data from the LLM response
   - It handles potential JSON parsing errors through regex fallbacks
   - The extracted data is added to the state for downstream use

5. **Output Formatting**:
   - The system formats the final output with:
     - Original question ID and content
     - Extracted category and subcategory information

## Example Output

```json
[
    {
        "id": "20705cc57af2ec0d2ea976de82a4c833f915d6a0bd6b3e3b508c3a4edf213743",
        "question": "I have a field that contains a string of numbers like \"2002 2005 2001 2006 2008 2344\". I know how to select a substring using the `SELECT substr(years,1,4)  FROM TABLE` query, but I'm struggling to check each substring individually. My goal is to find the row that contains the number closest to 0. Is there a way to achieve this using SQL?",
        "category": "Database",
        "sub_category": "SQL Query"
    },
    {
        "id": "299392dcbd6a991853554c2869deeff6f97b13db6a3a4c7f6e25fb41778dafc2",
        "question": "I want to create a function in Python that checks whether a given substring is present in a given string. How can I do that?",
        "category": "Programming",
        "sub_category": "String Manipulation"
    },
    {
        "id": "305421a8148e4bb08fc7802b9b882e208c5b450b2c1c3d74694bf489ba346b4b",
        "question": "<p>When an 8086 or 8088 first powers up, what address does the processor begin executing at? I know the Z80 starts at 0, and the 6809 looks to FFFF for an address to start at, but I have often wondered how the x86 starts off.</p>\n<p>EDIT:</p>\n<p>This is assuming no BIOS intervention. I.E. if I had my own EEPROM to boot from, where should it be located in RAM to start the machine running?</p>",
        "category": "Hardware",
        "sub_category": "Microprocessors"
    }
]
```