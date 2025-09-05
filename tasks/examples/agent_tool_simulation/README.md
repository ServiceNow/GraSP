# Agent Tool Simulation

This example demonstrates how to integrate external tools with AI agents using the GraSP framework. It showcases how to structure and use function calling with agents to perform tasks requiring specific capabilities, in this case mathematical operations.

> **Key Features**:
> `function calling`, `tool integration`, `mathematical operations`, `class-based tools`, `module-based tools`

## Overview

The agent tool simulation is designed to:

- **Demonstrate tool integration**: Show how to connect external functions as tools to agents
- **Showcase multiple tool sources**: Illustrate different approaches to defining and importing tools
- **Implement tool-based reasoning**: Enable agents to decompose problems and select appropriate tools

## Directory Contents

- `task_executor.py`: Contains post-processing logic for the tool-using agent
- `graph_config.yaml`: Configuration file defining the workflow graph and tool imports
- `tools.py`: Basic tools defined as standalone functions
- `tools_from_class.py`: Tools defined as methods within a class
- `tools_from_module.py`: Module-level tools that can be imported as a group
- `user_queries.json`: Sample mathematical queries to test the agent with tools

## How It Works

1. **Query Loading**: The system loads math questions from `user_queries.json`

2. **Tool Configuration**: The agent is configured with access to multiple math operation tools:
   - Addition function from standalone module
   - Subtraction and division methods from a class
   - Multiplication function from another module

3. **Agent Processing**:
   - The agent receives a mathematical query
   - It analyzes which tools are needed to solve the problem
   - It calls the appropriate tools with the correct parameters
   - The agent combines tool outputs to produce a final answer

4. **Result Capture**:
   - The `MathAgentPostProcessor` extracts the calculated result
   - The system stores this in the state for structured output

## Example Output

```json
[
    {
        "id": "6b9e75c4fe9b803f0c6f1dd59081750fa100db56874a12b0b83683b7ea9a0c8b",
        "user_query": "Can you give me answer for 3/2?",
        "math_result": "The result of \\( \\frac{3}{2} \\) is 1.5."
    },
    {
        "id": "e79908ac2f64bbdf7611c307ba3b73f647a2bbcf06d42e9c16c6321aa9ac3e1d",
        "user_query": "Can you give me answer for (2+3)*5?",
        "math_result": "The answer to \\((2+3) \\times 5\\) is \\(25\\)."
    },
    {
        "id": "6af6cadbfaa3eda3b9eb678dece24dac211c8bd3cd62e8742d8364422cfeecb2",
        "user_query": "Can you give me answer for multiplying 2 and 3 and then subtract 1 from answer?",
        "math_result": "The result of multiplying 2 and 3 and then subtracting 1 is 5."
    }
]
```

