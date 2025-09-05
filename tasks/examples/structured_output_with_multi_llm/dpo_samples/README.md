# DPO Samples Task using Multi-LLM Node

> **Key Features**:
> `multi-LLM processing`, `response evaluation`, `parallel model inference`, `quality rating`, `structured output schemas`

## Introduction

The DPO (Direct Preference Optimization) Samples task is designed to generate and evaluate multiple model responses to the same prompt, then rate them based on quality. This task demonstrates how to use multiple LLMs in parallel, evaluate their outputs using a judge model, and collect structured responses across different quality buckets for training data generation.

This implementation uses the GraspState framework to manage state across multiple iterations, ensuring proper data flow and state persistence throughout the task execution.

## Flow Structure

The DPO Samples task follows this workflow:

1. **Data Preparation**:
   - Extract user prompt and baseline response from the conversation
   - Initialize state variables for tracking ratings

2. **Generation Phase**:
   - Send the user prompt to multiple LLMs (gpt4, gpt-4o, gpt-4o-mini)
   - Collect structured responses from each model

3. **Rating Phase**:
   - Format all model responses for evaluation
   - Send to a judge model (gpt4) to rate each response
   - Parse and store ratings

4. **Iteration Control**:
   - Check if we have collected responses across different quality buckets
   - Continue generating more samples or end the flow based on criteria

5. **Output Generation**:
   - Compile all rated responses
   - Sort by rating score
   - Format the final output

## Configuration Details

### Data Transforms

The task uses three custom data transforms:

1. **ExtractUserPromptTransform**: Extracts the user prompt from the conversation
2. **ExtractResponseScaleTransform**: Extracts the baseline response from the conversation
3. **InitializeStateVariablesTransform**: Initializes state variables for tracking

### Graph Configuration

The graph consists of two main nodes:

1. **generate_samples (multi_llm)**:
   - Uses three different models: gpt4, gpt-4o, and gpt-4o-mini
   - Each model has different structured output configurations:
     - gpt4: YAML-defined schema
     ```yaml
     structured_output:
      schema:
        fields:
          message:
            type: str
            description: "Response message"
          success:
            type: bool
            description: "Operation success status"
     
     ```
     - gpt-4o: Class-based schema (SimpleResponse)
     ```yaml
     structured_output:
      schema: "models.structured_output.schemas_factory.SimpleResponse"
     
     ```
     - gpt-4o-mini: YAML-defined schema with structured output disabled
     ```yaml
     structured_output:
      enabled: false
      schema:
        fields:
          message:
            type: str
            description: "Response message"
          success:
            type: bool
            description: "Operation success status"
     
     ```

2. **rate_samples (llm)**:
   - Uses gpt4 as the judge model
   - Takes formatted model responses and rates them on a scale of 1-10
   - Returns ratings in JSON format

### Edge Conditions

The task uses a custom edge condition (`ShouldContinueCondition`) to determine whether to:
- Continue generating more samples
- End the flow

The condition checks:
- If we have collected samples across all quality buckets (low: 1-4, medium: 5-7, high: 8-10)
- If we've reached the maximum number of iterations (3)

## State Management

The task maintains these key state variables:

- **user_prompt**: The user's question extracted from the conversation
- **baseline_response**: The reference response extracted from the conversation
- **samples_ratings**: List of ratings for each iteration's model responses
- **current_model_responses**: The current batch of model responses

## Output Format

The final output includes:

1. The original user prompt
2. A list of model responses with:
   - The model name
   - The generated response (preserving structured output)
   - The judge's rating (1-10)
   - The judge's explanation for the rating

Responses are sorted by rating in descending order.

## Running the Task

To run the task:

1. Ensure you have the required models configured in your models.yaml
2. Your dataset is in conversation format. For example:
```json
{
  "id": "test_id",
  "annotation_type": ["scale", "gpt4", "gpt-4o", "gpt-4o-mini"],
  "language": "en",
  "tags": ["dpo_samples_rating"],
  "conversation": [
    {
      "role": "user",
      "content": "What are the key considerations when designing a sustainable urban transportation system?"
    },
    {
      "role": "assistant",
      "content": "Designing a sustainable urban transportation system requires..."
    }
  ]
}
```
```shell 
   python main.py --task examples.structured_output_with_multi_llm.dpo_samples --num_records 1
```

## Example Output

```json
{
  "id": "test_id",
  "taxonomy": ["test_taxonomy"],
  "annotation_type": ["scale", "gpt4", "gpt-4o", "gpt-4o-mini"],
  "language": "en",
  "tags": ["dpo_samples_rating"],
  "conversation": [
    {
      "role": "user",
      "content": "What are the key considerations when designing a sustainable urban transportation system?"
    },
    {
      "role": "assistant",
      "content": [
        {
          "generation": {
            "message": "Designing a sustainable urban transportation system requires...",
            "success": true
          },
          "model": "gpt-4o",
          "judge_rating": 9,
          "judge_explanation": "This response provides comprehensive coverage of sustainability factors..."
        },
        {
          "generation": {
            "message": "When designing a sustainable urban transportation system...",
            "success": true
          },
          "model": "gpt4",
          "judge_rating": 8,
          "judge_explanation": "The response covers most key considerations..."
        },
        {
          "generation": "A sustainable urban transportation system should focus on...",
          "model": "gpt-4o-mini",
          "judge_rating": 6,
          "judge_explanation": "The response covers basic aspects but lacks depth..."
        }
      ]
    }
  ]
}
```
