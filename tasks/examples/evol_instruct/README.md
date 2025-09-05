# Evol-Instruct

This example demonstrates how to use a subgraph within the GraSP framework. It showcases usage of [Evol-Instruct](https://arxiv.org/pdf/2304.12244) recipe which is a evolution techniques that can transform simple prompts into more complex, nuanced, and challenging instructions through automated prompt engineering.

> **Key Features**:
> `prompt evolution`, `subgraphs`, `automated prompt engineering`, `depth/breadth techniques`, `prompt transformation`

## Overview

The Evol-Instruct example is designed to:

- **Transform simple prompts**: Evolve basic instructions into more sophisticated variants
- **Apply various evolution techniques**: Implement depth-based (constraints, deepening, concretizing, reasoning) and breadth-based transformations

## Directory Contents

- `task_executor.py`: Core functionality for outputting evolved text
- `graph_config.yaml`: Configuration file defining the workflow graph
- `test.json`: Sample input prompts for evolution

## How It Works

1. **Input Prompts**:
   - The system loads simple prompts from `test.json`
   - Each prompt is a basic instruction or question (e.g., "tell me a story about a jungle and river")

2. **Evolution Process**:
   - The prompt is passed to the `evol_text` subgraph (defined in [grasp/recipes/evol_instruct](../../../grasp/recipes/evol_instruct/Readme.md))
   - The `EvolInstructPromptGenerator` lambda function transforms the input prompt
   - A random evolution technique is applied from available methods:
     - **Depth techniques**: Add constraints, increase depth/breadth, concretize concepts, require multi-step reasoning
     - **Breadth technique**: Create a new prompt in the same domain but more rare/specialized

3. **Transformation**:
   - The evolved prompt is processed by the LLM in the subgraph
   - The result is a more complex version of the original prompt
   - The evolved text is passed to the main graph's `query_llm` node

4. **Response Generation**:
   - The LLM responds to the evolved prompt
   - The system captures both the original text, evolved text, and the LLM response

## Example Output

```json
[
    {
        "id": 1,
        "text": "tell me a story about a jungle and river.",
        "evolved_text": "Compose a story set in a jungle with a river, incorporating a mythical creature or legend as a key element.",
        "llm_response": "In the heart of the lush Amazon jungle, where the canopy formed an almost impenetrable roof of green, a river wound its twisting path through the verdant landscape. Known to the locals as the Serpent's Vein, the river was said to be imbued with the spirits of the jungle and guarded by a mystical creature known only through whispered legends—the Yacumama.\n\nThe Yacumama, or \"Mother of the Waters,\" was an enormous serpent believed to dwell in the depths of the river. Tales spoke of her as the protector of the jungle, the guardian of all its creatures. It was said that anyone who dared to pollute her domain or disrespect the jungle's delicate balance would face her wrath.\n\nAmara, a young woman of the native Ewari tribe, had grown up with stories of the Yacumama. Her grandmother, the wise elder of the tribe, spoke often of encounters with the creature, filling Amara's dreams with visions of the great serpent's shimmering scales and eyes as deep and endless as the night sky.\n\nOne day, a dangerous threat came to their land. Foreign prospectors, intent on extracting gold from the jungle, arrived with roaring machines and avaricious intentions. The machines began their destruction, polluting the Serpent's Vein with toxic waste, turning the clear waters murky, and frightening the creatures of the jungle into silence.\n\nThe tribe, led by Amara's grandmother, gathered at the sacred grove near the river's edge, seeking guidance from the jungle spirits. As chants filled the air, a gentle ripple passed along the river's surface, and a mist began to rise. Amara, guided by an inexplicable pull, stepped closer to the water, where she felt a presence—a deep, resonant pulse that matched the rhythm of her heart.\n\nDetermined to save her home, Amara embarked on a journey downriver, hoping to confront the invaders and implore the Yacumama for aid. As she paddled a small canoe through the dense fog, the jungle seemed to come alive around her, vines whispering secrets and birds calling out as if urging her onward.\n\nEventually, she reached the site of destruction. There, the river was choked with debris, and the once-green banks lay bare under the incessant assault of machinery. Taking a deep breath, Amara called upon the spirit of the Yacumama, pouring all her hope and desperation into her plea.\n\nAt first"
    },
    {
        "id": 2,
        "text": "what happened during first world war in India?",
        "evolved_text": "During the First World War, how did events in India evolve over time, including political, social, and economic impacts?",
        "llm_response": "During the First World War, India, then a part of the British Empire, experienced significant political, social, and economic changes.\n\n### Political Impact:\n1. **Increased Nationalism**: The war accelerated Indian demands for self-governance. Indian leaders hoped that their support for the British in the war would be rewarded with political concessions.\n2. **Montagu Declaration (1917)**: In response to growing nationalist sentiment, the British government made a historic move with the Montagu Declaration, which promised \"increasing association of Indians in every branch of the administration and the gradual development of self-governing institutions.\"\n3. **Rise of Political Movements**: The Indian National Congress became more assertive in its demands. Leaders like Bal Gangadhar Tilak and Mahatma Gandhi emerged as prominent figures during this period.\n4. **Rowlatt Act (1919)**: Post-war, the Rowlatt Act was passed, allowing the government to imprison suspects without trial. It fueled widespread discontent and protests.\n\n### Social Impact:\n1. **Social Strain**: The war effort strained Indian society. Millions of men were recruited into the British Indian Army, affecting families and villages.\n2. **Influx of Ideas**: Exposure to global ideas during the war contributed to a cultural and ideological awakening. Many Indians, serving overseas, returned with new perspectives.\n3. **Gandhi’s Emergence**: Mahatma Gandhi's non-violent civil disobedience began to galvanize Indian society, advocating for social reforms in addition to political change.\n\n### Economic Impact:\n1. **Economic Exploitation**: India was a crucial financial and military resource for the British during the war. There was heavy taxation and requisitioning of resources to support the war.\n2. **Industrial Growth**: The war led to certain industrial developments in India due to disruption in imports from Europe, which fostered domestic industries, especially textiles.\n3. **Inflation and Hardships**: The war caused significant inflation, resulting in hardships for many Indians. The economic disparity between different classes widened.\n\nOverall, while the First World War intensified India's contributions to global military efforts, it also laid the groundwork for a more organized and forceful push towards Indian self-rule. The after-effect was a nation more politically conscious and determined to achieve independence, setting the stage for the next phase of its freedom struggle."
    }
]
```