from typing import Any

from grasp.core.graph.grasp_state import GraspState
from grasp.processors.output_record_generator import BaseOutputGenerator


class EvolTextGenerator(BaseOutputGenerator):
    def generate(self, state: GraspState) -> dict[str, Any]:
        print(state["text"], state["evolved_text"])
        return state
