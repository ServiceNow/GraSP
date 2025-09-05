# Evol Instruct subgraph
This graph is designed to reuse in other scenarios just by adding it as a subgraph.
The prompt and code flow is taken from https://github.com/nlpxucan/WizardLM/tree/main/Evol_Instruct
There are 4 depth and 1 breadth prompt. The prompt can evolved in breadth or depth or mix of both, however, 
the code is currently supporting randomly selected one.

### Input Parameter:
    text : The text which should be evolved with instruction.
    algorithm: Currently it does random prompt out of 5 prompts. Defaults to random.

### Output:
    evolved_text: The evolved text.

### Example:
Make sure the state has `text` key either coming from dataset 
or created as intermediate variable or it should be mapped.
```YAML
graph_config:
  nodes:
    evol_text:
      node_type: subgraph
      subgraph: grasp.recipes.evol_instruct
    query_llm:
      node_type: llm
      prompt:
        - user: |
           {evolved_text}
      model:
        name: gpt-4o-mini
        parameters:
          temperature: 1.0

  edges:
    - from: START
      to: evol_text
    - from: evol_text
      to: query_llm
    - from: query_llm
      to: END
```