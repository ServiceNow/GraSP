from typing import Any

from datasets import load_dataset
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END

from grasp.core.base_task_executor import BaseTaskExecutor
from grasp.utils import utils


def critique_answer_node_pre_process(state):
    """
    Converts stored 'messages' from the custom format into
    the LangChain message format for the critique_answer node.
    """
    if not state["messages"]:
        state["messages"] = []

    # We need to convert user turns to assistant and vice versa
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    translated = [cls_map[msg.type](content=msg.content) for msg in state["messages"]]
    state.update({"messages": translated})
    return state


def should_continue(state):
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


class TaskExecutor(BaseTaskExecutor):
    """
    TaskExecutor orchestrates:
      1. Loading MBPP dataset
      2. Executing the new pipeline that includes:
         - persona_sampler
         - paraphrase_question
         - evolve_instruct
         - generate_answer
         - critique_answer
      3. Deciding if we loop or end via should_continue
      4. Converting final run into an output record
    """

    def init_dataset(self) -> list[dict]:
        """
        Loads the MBPP dataset from Hugging Face (both train, validation, prompt splits)
        and attaches a unique 'id' to each record.
        """
        final_dataset = []
        for split in ["train", "validation", "prompt"]:
            dataset = load_dataset("mbpp", "sanitized", split=split)
            dataset = dataset.to_list()
            for data in dataset:
                data["id"] = data["task_id"]
            final_dataset.extend(dataset)
        return final_dataset

    def output_record_generator(self, state) -> dict[str, Any] | None:
        """
        Generates the final data record only if:
          - We have 'messages' in state
          - The last message from the critic is 'NO MORE FEEDBACK'

        The record includes:
          - 'conversation' (reconstructed from messages in chat format)
          - 'id'
          - Additional metadata (taxonomy, annotation_type, etc.)
        """
        if "messages" not in state:
            return None

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
            return None

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

        return {
            "id": state.get("id", ""),
            "conversation": chat_format_messages,
            "original_question": state.get("prompt", ""),
            "rephrased_text": state["rephrased_text"],
            "taxonomy": [{"category": "Coding", "subcategory": ""}],
            "annotation_type": ["gpt-4o"],
            "language": ["en"],
            "tags": ["mbpp", "reannotate", "self-critique"],
        }