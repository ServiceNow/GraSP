from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from grasp.core.graph.functions.edge_condition import EdgeCondition
from grasp.core.graph.functions.node_processor import (
    NodePostProcessor,
    NodePreProcessor,
)
from grasp.core.graph.grasp_state import GraspState
from grasp.processors.output_record_generator import BaseOutputGenerator
from grasp.utils import constants, utils


class CritiqueAnsNodePreProcessor(NodePreProcessor):
    """
    Old code transformed to new format
    """

    def apply(self, state: GraspState) -> GraspState:
        if not state["messages"]:
            state["messages"] = []

        # We need to convert user turns to assistant and vice versa
        cls_map = {"ai": HumanMessage, "human": AIMessage}
        translated = [
            cls_map[msg.type](content=msg.content) for msg in state["messages"]
        ]
        state.update({"messages": translated})
        return state


class CritiqueAnsNodePostProcessor(NodePostProcessor):
    def apply(self, response: AIMessage) -> GraspState:
        return {
            "messages": [HumanMessage(response.message.content)],
        }


class ShouldContinueCondition(EdgeCondition):
    def apply(state: GraspState) -> str:
        # End after 4 iterations or the last feedback response contains "NO MORE FEEDBACK"
        messages = state["messages"]
        if len(messages) > 8 or (
            len(messages) > 1 and "no more feedback" in messages[-1].content.lower()
        ):
            return constants.GRASP_END
        return "generate_answer"


class CodeGenOutputGenerator(BaseOutputGenerator):
    @staticmethod
    def build_conversation(data: Any, state: GraspState) -> list[dict]:
        chat_format_messages = utils.convert_messages_from_langchain_to_chat_format(
            data
        )
        chat_format_messages.insert(0, {"role": "user", "content": state["question"]})

        return chat_format_messages
