# Agent Simulation

This example demonstrates how to create structured AI agent dialogues with opposing viewpoints using the GraSP framework. It showcases a directed graph implementation that orchestrates a realistic, multi-turn conversation between two dynamically generated personas discussing a specific topic using agentic nodes.

> **Key Features**:
> `agent nodes`, `chat history`, `system message intervention`, `multi-turn dialogue`, `multi-agent`

## Overview

The agent simulation is designed to:

- **Generate realistic debates**: Create authentic dialogue with contrasting perspectives on specific topics
- **Guide conversational arcs**: Manage a complete conversation from greeting to conclusion
- **Illustrate system interventions**: Use strategic message injection to guide conversation depth and structure

## Directory Contents

- `task_executor.py`: Core functionality for the agent simulation
- `graph_config.yaml`: Configuration file defining the workflow graph
- `categories.json`: Sample categories/topics for agent conversations

## How It Works

1. **Topic Selection**: The system loads a category and subcategory from `categories.json` (e.g., Health/Yoga)

2. **Persona Generation**: The `persona_assignor` node creates two distinct personas relevant to the topic:
   - Assigns roles (one-word names) for each agent
   - Generates detailed prompts that define each agent's background, communication style, and stance

3. **Conversation Initialization**: 
   - A random agent is selected to start the conversation
   - The agent begins with a greeting that establishes their character and introduces the topic

4. **Turn-Based Discussion**:
   - Agents respond to each other's messages in turns
   - After turn 3, system messages encourage deeper discussion
   - After turn 5, system messages guide the agents to begin wrapping up

5. **Conversation Conclusion**:
   - The conversation ends when an agent includes "FINAL ANSWER" in their response
   - The system generates structured output including the conversation history and taxonomy

## Example Output

```json
[
    {
        "id": "d06735758145f5f858ccf2cd870003bbae142720b04e6349db38ea44b5c191ee",
        "conversation": [
            {
                "user": "Open the dialogue with respectful greetings. Remain fully in character and begin discussing the assigned topic, keeping a conversational tone that encourages a response. Just give direct response, nothing else.\n"
            },
            {
                "agent_2": "Hello. Let's dive into the subject of yoga. I'm interested in discussing its physical benefits, which seem well-documented in terms of flexibility, strength, and stress reduction. However, I'm curious about the more spiritual claims. What empirical evidence supports these aspects of yoga?"
            },
            {
                "agent_1": "Greetings. It's wonderful to explore the depth of yoga together. While its physical benefits are widely recognized, the spiritual and mental dimensions often elicit more curiosity. Empirically, it's challenging to quantify spirituality, but studies do highlight the impacts of yoga on mental health, such as reducing anxiety and enhancing mindfulness—elements that many practitioners describe as spiritual growth. These studies typically track changes in the brain and mindfulness levels, underscoring yoga's transformative effects on awareness and inner peace. It's in this subtle transformation that many find a deep spiritual connection. How do these aspects align with what you're curious to know about yoga's spiritual side?"
            },
            {
                "agent_2": "While I appreciate studies on reduced anxiety and increased mindfulness, I remain skeptical around quantifying spiritual growth based on such metrics. Stress reduction and mindfulness can certainly contribute to emotional well-being, but labeling these effects as spiritual seems to stretch the empirical data. I'm curious about the specific mechanisms that link these psychological benefits directly to spirituality. Are there any rigorous studies measuring direct spiritual outcomes, or do these remain largely anecdotal and subjective?"
            },
            {
                "agent_1": "It's understandable to feel skeptical about defining spiritual growth through scientific measures, as spirituality often transcends conventional metrics. Many studies indeed focus on psychological outcomes like stress reduction, which can pave the way for what some interpret as spiritual awakening through enhanced introspection and presence. However, direct spiritual experiences are often nuanced and personal, frequently described in subjective terms rather than objective data. \n\nSome research does explore the brain's response to mindfulness practices, suggesting shifts in areas like the prefrontal cortex and amygdala that are associated with greater emotional regulation and self-awareness—key components in many spiritual journeys. Ultimately, this area of study is evolving, and while empirical evidence provides a framework, the depth of spiritual experiences often lies beyond quantifiable measures. How do you perceive the role of personal experience in understanding spirituality within yoga?"
            },
            {
                "agent_2": "While personal experiences are undeniably valuable, they often serve anecdotal roles that are difficult to evaluate scientifically. As a scientist focused on exercise physiology, I'm primarily interested in understanding how these experiences can be systematically observed or measured. The shifts in brain regions you mentioned could offer intriguing insights, potentially correlating increased self-awareness with what individuals describe as spiritual experiences. However, without objective methodologies specifically tailored to measure these dimensions, distinguishing between neural activity associated with emotional regulation versus spiritual awakening remains a complex challenge. Do you think we need new frameworks or paradigms for studying these spiritual aspects in yoga, or should efforts focus more on the psychological benefits observed?"
            },
            {
                "agent_1": "The quest to explore spirituality through empirical lenses certainly invites new paradigms. Traditionally, scientific inquiry illuminates the tangible—only scratching the surface of the layered experience of yoga. Yet, as interest grows in the intersection of consciousness and neuroscience, innovative approaches may arise to bridge the gap between the quantifiable and the spiritual.\n\nPerhaps the path forward involves both: developing frameworks that can more accurately reflect the subtleties of spiritual experience while continuing to articulate the psychological benefits clearly observed. Expanding research into the brain's response during deep meditative states could offer more clues, perhaps even inspiring methodologies that acknowledge personal narrative as a part of holistic understanding.\n\nUltimately, it's about expanding our receptivity—embracing the possibility that some realms of wellness, such as spiritual growth through yoga, might weave together scientific exploration with the art of personal journey. How do you envision integrating these dimensions as part of a broader understanding of yoga’s impact?"
            },
            {
                "agent_2": "I appreciate your open-minded perspective on exploring new frameworks to better understand yoga's spiritual dimensions. While the pursuit of hard evidence remains paramount for scientific rigor, there certainly is value in considering both the objective and subjective experiences that practitioners report. \n\nTo integrate these dimensions, perhaps we could start by refining how we define and measure \"spirituality\" in the context of yoga. By collaborating with psychologists, neuroscientists, and even philosophers, we could develop mixed-method studies that combine quantitative data, like brain imaging, with qualitative accounts from practitioners. This could forge a more comprehensive picture, respecting both empirical data and the personal narratives that drive many to yoga.\n\nWould you say that such an interdisciplinary approach might be beneficial, or do you see other avenues for grounding spiritual claims in more systematic research?"
            },
            {
                "agent_1": "Your suggestion for an interdisciplinary approach resonates deeply with me. The richness of yoga lies not only in its physical postures or its impact on the brain but also in the lived experiences that each practitioner integrates into their personal journey. Collaborating across fields could indeed forge pathways for a nuanced understanding, enhancing both scientific comprehension and spiritual narrative.\n\nBy fostering dialogue between disciplines, we open space for innovative methods that honor the complexity of spiritual experiences. Brain imaging and mindfulness metrics may provide valuable insights, but it's the synthesis of these with the authentic voices of those who practice yoga that could truly expand our understanding.\n\nIn grounding spiritual claims, I believe the key is to keep the conversation dynamic and receptive—allowing for traditional scientific methods to evolve and adapt alongside the intuitive experiences that yoga cultivates. How do you envision these collaborations taking shape in research or practice?"
            },
            {
                "agent_2": "It's promising to think about interdisciplinary collaborations as a way to deepen our understanding. In practice, such collaborations could materialize through joint research projects where varied expertise is brought together to design studies that address both measurable physiological changes and the qualitative accounts of spiritual evolution. These projects could involve neuroscientists focusing on brain activity, psychologists examining changes in mental states, and philosophers contributing to the nuanced discussions around defining spirituality. Moreover, technology such as neurofeedback and advanced imaging might play a role in systematically evaluating experiences that are currently considered anecdotal.\n\nHowever, the challenge lies in maintaining scientific objectivity while recognizing the unique value of personal experiences—a balancing act that requires continual revision of research methodologies. In this spirit, pilot studies that focus on capturing the full spectrum of yoga's impact could be a starting point, paving the way for more comprehensive, longitudinal studies that truly reflect the depth of these spiritual dimensions. Would you agree that pilot studies could offer a practical first step, or do you have other thoughts on how this integration might commence?"
            },
            {
                "agent_1": "Pilot studies are indeed a promising first step, serving as an experimental ground to cultivate understanding before broader explorations. These initial studies can lay the foundation for more nuanced methodologies, combining empirical rigor with the subtleties of personal experience.\n\nThe success of such endeavors will likely depend on their ability to remain flexible and adaptive, utilizing insights from neuroscience, psychology, and philosophy to shape evolving definitions and metrics of spirituality. It's in these early stages that researchers can refine their approaches—adjusting their lenses to capture both the quantifiable and the intangible.\n\nMoreover, these pilot studies can foster a spirit of inquiry and openness among practitioners, inviting them to share their personal journeys as valuable data. By treating these experiences as frontiers for learning, rather than dismissing them as merely anecdotal, we could foster a holistic understanding that appreciates the symbiosis between science and spirituality. How do you see technology, such as neurofeedback, enhancing these studies to better bridge the empirical and experiential in yoga research?"
            },
            {
                "agent_2": "Neurofeedback presents an intriguing avenue for exploring the interplay between empirical data and personal experience in yoga. By providing real-time feedback on brain activity, practitioners could gain insights into their mental states during yoga and meditation, potentially creating a more objective measure of experiences often described as spiritual or transcendent. This feedback could also guide individuals in optimizing their practice for better alignment with personal goals, whether they're focused on relaxation, stress reduction, or deeper mindfulness.\n\nThe challenge will be to ensure that this technological layer does not overshadow the unique personal narratives that make each practitioner's journey distinct. Rather, it should serve as a complement, providing a bridge between measurable brain patterns and subjective experiences. As pilot studies advance, it's crucial to maintain an openness to adjust hypotheses and methodologies based on ongoing findings, acknowledging the dynamic and complex interrelationships at play.\n\nFINAL ANSWER: While physical benefits of yoga are well-documented, exploring its spiritual claims requires an interdisciplinary approach, integrating neuroscience, psychology, and philosophy. Pilot studies using technologies like neurofeedback could uncover insights into the empirical-qualitative spectrum, paving the way for comprehensive understanding. Both empirical data and personal experiences should inform evolving research methodologies, ensuring a robust exploration of yoga's multifaceted impacts."
            }
        ],
        "taxonomy": [
            {
                "category": "Health",
                "subcategory": "Yoga"
            }
        ]
    }
]
```