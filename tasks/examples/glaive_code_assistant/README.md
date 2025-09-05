# Glaive Code Assistant

This example demonstrates how to build a self-improving code assistant using the GraSP framework. It implements a feedback loop where an LLM generates code solutions and then critiques and refines its own answers through multiple iterations.

> **Key Features**:
> `self-critique`, `iterative refinement`, `code generation`, `feedback loop`, `stopping conditions`

## Overview

The Glaive Code Assistant example is designed to:

- **Generate code solutions**: Create solutions to coding problems using LLMs
- **Implement self-critique**: Evaluate generated code against reference solutions
- **Enable iterative improvement**: Refine solutions based on critique feedback

## Directory Contents

- `task_executor.py`: Contains the processor classes and logic for the critique cycle
- `graph_config.yaml`: Configuration file defining the workflow graph for code generation and critique

## How It Works

1. **Data Source**:
   - The system loads coding problems from the "glaiveai/glaive-code-assistant-v2" HuggingFace dataset
   - Each problem includes a question and a reference solution

2. **Solution Generation**:
   - The `generate_answer` node produces an initial code solution for the problem
   - The model responds directly with code, avoiding conversational text

3. **Critique Cycle**:
   - The `critique_answer` node evaluates the solution against the reference
   - It identifies issues and provides specific recommendations for improvement
   - The critique is passed back to the solution generator for refinement

4. **Iterative Refinement**:
   - The system cycles between generation and critique nodes
   - With each iteration, the solution is improved based on feedback
   - The `ShouldContinueCondition` determines when to stop refinement

5. **Completion**:
   - The cycle ends when either:
     - The critique responds with "NO MORE FEEDBACK" (indicating a correct solution)
     - The maximum number of iterations is reached (8 rounds)

## Example Output

```json
[
    {
        "id": "299392dcbd6a991853554c2869deeff6f97b13db6a3a4c7f6e25fb41778dafc2",
        "conversation": [
            {
                "role": "user",
                "content": "I want to create a function in Python that checks whether a given substring is present in a given string. How can I do that?"
            },
            {
                "role": "assistant",
                "content": "```python\ndef is_substring_present(main_string, substring):\n    return substring in main_string\n```"
            },
            {
                "role": "user",
                "content": "NO MORE FEEDBACK"
            }
        ],
        "taxonomy": [
            {
                "category": "Coding"
            }
        ],
        "annotation_type": [
            "mistral-large"
        ],
        "language": "en",
        "tags": [
            "glaiveai/glaive-code-assistant-v2",
            "reannotate",
            "self-critique"
        ]
    }
]
```

