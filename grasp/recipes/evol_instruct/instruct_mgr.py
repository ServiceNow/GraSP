import random

from grasp.recipes.evol_instruct import breadth, depth


def get_instruction(instruction: str, algorithm="random") -> str:
    evol_prompts: list[str] = []
    evol_prompts.append(depth.createConstraintsPrompt(instruction))
    evol_prompts.append(depth.createDeepenPrompt(instruction))
    evol_prompts.append(depth.createConcretizingPrompt(instruction))
    evol_prompts.append(depth.createReasoningPrompt(instruction))
    evol_prompts.append(breadth.createBreadthPrompt(instruction))

    if algorithm == "random":
        return random.choice(evol_prompts)
    else:
        raise ValueError("Invalid algorithm")
