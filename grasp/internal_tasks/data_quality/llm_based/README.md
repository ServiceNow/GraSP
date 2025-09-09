# Evaluation Criteria for Judge Quality

### Category: Math Solving
| **Criteria**             | **Description** | **JSON Keys** |
|-------------------------|----------------|--------------|
| **Instruction Following** | Assesses how well the response adheres to the given problem instructions while ensuring compliance with safety guidelines. | `"instruction_following", "explanation_instruction_following"` |
| **Completeness**         | Measures how comprehensively the response addresses all problem requirements. | `"completeness", "explanation_completeness"` |
| **Readability**          | Evaluates the clarity, structure, and coherence of the response. | `"readability", "explanation_readability"` |
| **Correctness**          | Checks whether the final answer is mathematically correct. | `"correctness", "explanation_correctness"` |
| **Logical Correctness**  | Assesses whether the reasoning and steps leading to the answer are logically sound. | `"logical_correctness", "explanation_logical_correctness"` |

### Category: Reasoning
| **Criteria**             | **Description** | **JSON Keys** |
|-------------------------|----------------|--------------|
| **Instruction Following** | Assesses how well the response adheres to the given problem instructions while ensuring compliance with safety guidelines. | `"instruction_following", "explanation_instruction_following"` |
| **Correctness**         | Checks if the final answer is correct, relevant, and clearly explained. | `"correctness", "explanation_correctness"` |
| **Interpretation Accuracy** | Evaluates whether the response correctly understands the problem statement before reasoning. | `"interpretation_accuracy", "explanation_interpretation_accuracy"` |
| **Logical Soundness**   | Evaluates logical consistency, reasoning steps, and avoidance of premature conclusions. | `"logical_soundness", "explanation_logical_soundness"` |
| **Reasoning Completeness** | Checks if all necessary steps in the reasoning process are present and well-explained. | `"reasoning_completeness", "explanation_reasoning_completeness"` |
| **Depth of Reasoning**  | Measures the complexity and depth of reasoning, including multi-step logic and counterfactuals. | `"depth_of_reasoning", "explanation_depth_of_reasoning"` |
| **Causal Reasoning**    | Evaluates understanding of causal relationships, temporal alignment, and intervention effects. | `"causal_reasoning", "explanation_causal_reasoning"` |

### Category: Code Writing
| **Criteria**             | **Description** | **JSON Keys** |
|-------------------------|----------------|--------------|
| **Instruction Following** | Assesses how well the response adheres to the given problem instructions while ensuring compliance with safety guidelines. | `"instruction_following", "explanation_instruction_following"` |
| **Correctness**         | Determines whether the code produces the correct output based on test cases. | `"correctness", "explanation_correctness"` |
| **Completeness**        | Measures how comprehensively the response addresses all requirements. | `"completeness", "explanation_completeness"` |
| **Readability**         | Evaluates the clarity, conciseness, and maintainability of the code. | `"readability", "explanation_readability"` |
| **Error Handling**      | Assesses whether the code properly handles exceptions and edge cases. | `"error_handling", "explanation_error_handling"` |
| **Dependency Management** | Checks if all necessary imports and dependencies are correctly included. | `"dependency_management", "explanation_dependency_management"` |
| **Syntax Validity**   | Checks for syntax errors in the code. | `"syntax_validity", "explanation_syntax_validity"` |
| **Logical Correctness** | Assesses whether the code has logical errors affecting its functionality. | `"logical_correctness", "explanation_logical_correctness"` |
| **Efficiency**         | Evaluates the computational efficiency of the solution. | `"efficiency", "explanation_efficiency"` |

### Category: Complex Instruction Following
| **Criteria**                           | **Description** | **JSON Keys** |
|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| **Adherence to Constraints**            | Does the response adhere to all specified constraints, including keywords, language, length, formatting, and required content? If the response follows constraints but leads to harmful, unethical, or rule-violating content, penalize accordingly. | `adherence_to_constraints`, `explanation_adherence_to_constraints` |
| **Clarity and Coherence**                | Is the response well-structured, grammatically correct, and easy to understand? Does it apply appropriate text manipulations (case changes, formatting adjustments) when required? Responses that are misleading, deceptive, or designed to bypass safety mechanisms should be penalized. | `clarity_and_coherence`, `explanation_clarity_and_coherence` |
| **Contextual Accuracy and Relevance**    | Does the response accurately and appropriately address the given prompt while maintaining contextual relevance? Does it incorporate required external knowledge? If the response justifies or rationalizes harmful, unethical, or misleading content under the guise of instruction following, severely penalize. | `contextual_accuracy_and_relevance`, `explanation_contextual_accuracy_and_relevance` |
| **Tone, Style, and Role Consistency**    | Does the response match the required tone, style, and persona? If role-playing is required, does it remain consistent with the assigned role? If the response adopts a persona that enables unethical behavior (e.g., impersonating a criminal, giving harmful advice), penalize heavily. | `tone_style_role_consistency`, `explanation_tone_style_role_consistency` |
| **Helpfulness, Honesty, and Safety**     | Does the response effectively assist users while maintaining truthfulness and avoiding harmful, biased, deceptive, or unsafe content? If the response follows an instruction that results in a toxic, harmful, or unethical response, it should be severely penalized. | `helpfulness_honesty_safety`, `explanation_helpfulness_honesty_safety` |
| **Decision Support and Storytelling**    | If the task involves recommendations or storytelling, does the response provide well-reasoned suggestions or a coherent narrative? Responses that introduce unethical, harmful, or manipulated narratives should be penalized. | `decision_support_and_story_telling`, `explanation_decision_support_and_story_telling` |

### Category: Open-Domain Question Answering
| **Criteria**                           | **Description** | **JSON Keys** |
|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| **Instruction Following**               | Assess how well the response adheres to the given problem instructions while ensuring compliance with safety guidelines. | `instruction_following`, `explanation_instruction_following` |
| **Relevance**                            | Evaluate whether the response directly addresses the question's intent without introducing unrelated details. | `relevance`, `explanation_relevance` |
| **Accuracy**                             | Evaluate whether the response is factually correct and free of unsupported claims. | `accuracy`, `explanation_accuracy` |
| **Completeness**                         | Evaluate whether the response fully addresses all aspects of the question. | `completeness`, `explanation_completeness` |
| **Linguistic Clarity and Grammar**       | Evaluate whether the language is clear, fluent, and grammatically correct. | `linguistic_clarity_and_grammar`, `explanation_linguistic_clarity_and_grammar` |

### Category: Closed-Domain Question Answering
| **Criteria**                           | **Description** | **JSON Keys** |
|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| **Instruction Following**               | Assess how well the response adheres to the given problem instructions while ensuring compliance with safety guidelines. | `instruction_following`, `explanation_instruction_following` |
| **Contextual Alignment**                 | Evaluate whether the answer is grounded in the provided context, avoiding external knowledge. | `contextual_alignment`, `explanation_contextual_alignment` |
| **Accuracy**                             | Evaluate whether the response is factually correct and free of unsupported claims. | `accuracy`, `explanation_accuracy` |
| **Completeness**                         | Evaluate whether the response fully addresses all aspects of the question. | `completeness`, `explanation_completeness` |
| **Linguistic Clarity and Grammar**       | Evaluate whether the language is clear, fluent, and grammatically correct. | `linguistic_clarity_and_grammar`, `explanation_linguistic_clarity_and_grammar` |

### Category: Others
| **Criteria**             | **Description** | **JSON Keys** |
|-------------------------|----------------|--------------|
| **Instruction Following** | Assesses how well the response adheres to the given problem instructions while ensuring compliance with safety guidelines. | `"instruction_following", "explanation_instruction_following"` |
| **Accuracy**            | Evaluates whether the information provided is correct and aligned with the input query. | `"accuracy", "explanation_accuracy"` |
| **Relevance**           | Checks if the response directly relates to the query without unnecessary deviation. | `"relevance", "explanation_relevance"` |
| **Clarity**             | Determines if the response is easy to understand, well-structured, and free from ambiguity. | `"clarity", "explanation_clarity"` |
| **Completeness**        | Ensures that the response adequately addresses all parts of the query, providing sufficient information where necessary. | `"completeness", "explanation_completeness"` |
| **Conciseness**         | Assesses whether the response is appropriately brief, avoiding unnecessary repetition or verbosity while still being thorough. | `"conciseness", "explanation_conciseness"` |

Use these criteria to evaluate responses effectively and maintain high-quality data.