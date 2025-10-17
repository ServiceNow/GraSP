from sygra.core.graph.functions.node_processor import NodePostProcessorWithState
from sygra.core.graph.sygra_message import SygraMessage
from sygra.core.graph.sygra_state import SygraState


class MathAgentPostProcessor(NodePostProcessorWithState):
    def apply(self, resp: SygraMessage, state: SygraState) -> SygraState:
        answer = resp.message.content
        state["math_result"] = answer
        return state
