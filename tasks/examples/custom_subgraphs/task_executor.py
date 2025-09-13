from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from grasp.core.graph.functions.edge_condition import EdgeCondition
from grasp.core.graph.functions.node_processor import NodePreProcessor
from grasp.core.graph.grasp_state import GraspState
from grasp.processors.output_record_generator import BaseOutputGenerator
from grasp.utils import utils


class CritiqueAnswerNodePreProcessor(NodePreProcessor):
    def apply(self, state: GraspState) -> GraspState:
        if not state.get("messages"):
            state["messages"] = []

        cls_map = {"ai": HumanMessage, "human": AIMessage}
        translated = [
            cls_map[msg.type](content=msg.content) for msg in state["messages"]
        ]
        state.update({"messages": translated})
        return state


class ShouldContinue(EdgeCondition):
    def apply(state: GraspState) -> str:
        """
        Decide whether to continue iterating with 'generate_answer'
        or end the pipeline. If we have done too many rounds or we
        see 'no more feedback', then we end.
        """
        messages = state["messages"]
        if len(messages) > 8 or (
            len(messages) > 1 and "no more feedback" in messages[-1].content.lower()
        ):
            return END
        return "generate_answer"


class CustomSubgraphsOutputGenerator(BaseOutputGenerator):
    """
    Custom output generator for custom subgraph example
    """

    @staticmethod
    def build_conversation(data: Any, state: GraspState) -> list[dict[str, str]]:
        """
        Builds the conversation from the messages in the state.
        """
        if "messages" not in state:
            return []

        # Convert from LangChain message objects back to chat dict format
        chat_format_messages = utils.convert_messages_from_langchain_to_chat_format(
            state["messages"]
        )

        # If the last message doesn't contain "no more feedback", we do NOT finalize
        if (
            len(chat_format_messages) < 1
            or "no more feedback"
            not in chat_format_messages[-1]["content"].lower().strip()
        ):
            return []

        # Remove the final "NO MORE FEEDBACK" message from the conversation
        chat_format_messages = chat_format_messages[:-1]

        # Replace the conversation's first turn with the original paraphrased question
        # or you could change it to the 'evolved_prompt' if you prefer
        if "rephrased_text" in state and state["rephrased_text"]:
            chat_format_messages.insert(
                0,
                {
                    "role": "user",
                    "content": state["rephrased_text"].replace(
                        "PARAPHRASED QUESTION: ", ""
                    ),
                },
            )

        return chat_format_messages

    @staticmethod
    def build_metadata(data: Any, state: GraspState) -> dict[str, Any]:
        """
        Builds the metadata from the state.
        """
        metadata = {
            "original_question": state.get("prompt", ""),
            "rephrased_text": state.get("rephrased_text", ""),
            "taxonomy": [{"category": "Coding", "subcategory": ""}],
            "annotation_type": ["gpt-4o"],
            "language": ["en"],
            "tags": ["mbpp", "reannotate", "self-critique"],
        }
        return metadata
