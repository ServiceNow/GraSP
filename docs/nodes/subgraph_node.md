## Subgraph Node

GraSP supports modular graph design through the **subgraph** node. Subgraphs allow you to encapsulate and reuse a sequence of nodes and edges as a single unit within a main graph. This makes complex workflows more organized, reusable, and easier to maintain.

You can include a subgraph in your main graph and customize the configuration of its internal nodes as needed for your specific workflow.

### Example Configuration

#### Subgraph Definition (`generate_question` subgraph)

```yaml
# Example: tasks/examples/custom_subgraphs/generate_question_subgraph/graph_config.yaml
graph_config:
  nodes:
    persona_sampler:
      node_type: weighted_sampler
      attributes:
        num_turns:
          values: [2, 3, 4, 5]
        tone1:
          values: [professional, casual, friendly]
        persona1:
          values: [high school teacher, software engineer]

    paraphrase_question:
      node_type: llm
      output_keys: text
      prompt:
        - system: |
            Assume you are {persona1} persona.
            Paraphrase the question while preserving all details.
        - user: |
            QUESTION: {p}
      model:
        name: gpt4o
        parameters:
          temperature: 0.1

  edges:
    - from: START
      to: persona_sampler
    - from: persona_sampler
      to: paraphrase_question
    - from: paraphrase_question
      to: END
```

#### Main Graph Including the Subgraph

```yaml
nodes:
  generate_question:
    node_type: subgraph
    subgraph: tasks.examples.custom_subgraphs.generate_question_subgraph
    node_config_map:
      paraphrase_question:
        output_keys: rephrased_text
        prompt_placeholder_map:
          p: prompt
        model:
          name: gpt4o
          parameters:
            temperature: 1.0
```

### Configuration Fields

- **`node_type`**:  
  Should be set to `subgraph` to indicate this node is a subgraph reference.

- **`subgraph`**:  
  The import path or module reference to the subgraph definition. This should point to the graph configuration file that defines the subgraph.

- **`node_config_map`**(Optional) :  
  Dictionary to customize nodes inside the subgraph for this usage.
  - The keys are node names within the subgraph (e.g., `paraphrase_question`).
  - The values are configuration fields you wish to modify for that specific subgraph node, such as `output_keys`, `model`, `prompt`, etc.
  - **Note:** Only the fields explicitly provided in `node_config_map` are overridden for the specified subgraph node. All other settings for that node remain as defined in the original subgraph.

    Supported fields include (but are not limited to):
    - `output_keys`: Override the output keys for the subgraph node.
    - `prompt_placeholder_map`: Remap input placeholders in prompts. For example, if the subgraph uses `{p}` but your main graph uses `prompt`, you can set `prompt_placeholder_map: { p: prompt }`.
    - `model`: Override model name or parameters.
    - Any other valid node configuration field.

#### Example: Using `prompt_placeholder_map`

If a subgraph node’s prompt uses `p` but your main graph contains `prompt` as input state variable, you can map this as follows:

```yaml
prompt_placeholder_map:
  p: prompt
```
This will update all occurrences of `{p}` in the subgraph prompt of that node to `{prompt}`.

### Notes

- **Customization**: With `node_config_map`, you can adapt subgraph nodes to fit your main graph’s needs—change output keys, update model parameters, or map input variables as needed. Only the fields you specify are changed; all others remain as originally set in the subgraph.
- **Prompt Placeholder Mapping**: Use `prompt_placeholder_map` to easily align variable names between your main graph and subgraph's prompt, without changing the original subgraph.
- **Reusability**: Subgraphs help you avoid duplication by making it easy to reuse common sequences or workflows across multiple graphs.
- **Compatibility**: Adding subgraph nodes does not affect existing graphs that don’t use this feature.

---

**Tip:** Use subgraphs to modularize common tasks such as data preprocessing, prompt engineering, or multi-step LLM workflows, and include them in different main graphs with custom settings as needed.

---