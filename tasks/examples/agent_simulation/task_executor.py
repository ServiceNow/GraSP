import json
import random
import re
from typing import Any


from grasp.core.graph.functions.edge_condition import EdgeCondition
from grasp.core.graph.functions.node_processor import (
    NodePostProcessorWithState,
    NodePreProcessor,
)
from grasp.core.graph.grasp_message import GraspMessage
from grasp.core.graph.grasp_state import GraspState
from grasp.processors.output_record_generator import BaseOutputGenerator

start_conversation_prompts = [
    "Begin the discussion with polite greetings. Stay in character and ease into the topic assigned to you. Make sure to invite your partner into the conversation. Just give direct response, nothing else.",
    "Start the conversation by greeting your counterpart. Establish your role clearly and introduce the topic in a natural, engaging way. Leave space for your partner to respond. Just give direct response, nothing else.",
    "Open the dialogue with respectful greetings. Remain fully in character and begin discussing the assigned topic, keeping a conversational tone that encourages a response. Just give direct response, nothing else.",
    "Greet your conversation partner and briefly introduce the subject of discussion. Express curiosity or enthusiasm while ensuring your message prompts a reply. Just give direct response, nothing else.",
    "Begin by acknowledging your counterpart with a greeting. Stay immersed in your persona and bring up the topic of interest in a way that invites your partner to share their thoughts. Just give direct response, nothing else.",
]


def safe_json_extract(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in the LLM response.")
    json_str = match.group(0)
    return json.loads(json_str)


def sanitize_name(name: str) -> str:
    return name.replace(" ", "_")


class PersonaPostProcessor(NodePostProcessorWithState):
    """
    PostProcessor that extracts the roles and prompts for agents
    """

    def apply(self, response: GraspMessage, state: GraspState) -> GraspState:
        parsed = safe_json_extract(response.message.content)
        parsed["agent_1_role"] = sanitize_name(parsed["agent_1_role"])
        parsed["agent_2_role"] = sanitize_name(parsed["agent_2_role"])
        state["agent_1_role"] = parsed["agent_1_role"]
        state["agent_2_role"] = parsed["agent_2_role"]
        state["agent_1_prompt"] = parsed["agent_1_prompt"]
        state["agent_2_prompt"] = parsed["agent_2_prompt"]
        return state


class ShouldContinueToNextAgent(EdgeCondition):
    def apply(state: GraspState) -> str:
        if not state.get("__chat_history__") or len(state["__chat_history__"]) == 0:
            return "FINAL_ANSWER"
        last_resp_history = state["__chat_history__"][-1]
        node_name = last_resp_history.get("name", "")
        last_agent_resp = last_resp_history.get("response", "")
        if "FINAL ANSWER" in last_agent_resp:
            return "FINAL_ANSWER"
        if node_name == "agent_1":
            return "agent_2"
        else:
            return "agent_1"


class StartConversation(EdgeCondition):
    def apply(state: GraspState) -> str:
        agent_chosen = random.choice([0, 1])
        if agent_chosen == 0:
            return "agent_1"
        elif agent_chosen == 1:
            return "agent_2"


class Agent1PreProcess(NodePreProcessor):
    def apply(self, state: GraspState) -> GraspState:
        state["_agent_name"] = state["agent_1_role"]
        state["_agent_prompt"] = state["agent_1_prompt"]
        if not state.get("__chat_history__"):
            state["agent_2_response"] = random.choice(start_conversation_prompts)
        return state


class Agent2PreProcess(NodePreProcessor):
    def apply(self, state: GraspState) -> GraspState:
        state["_agent_name"] = state["agent_2_role"]
        state["_agent_prompt"] = state["agent_2_prompt"]
        if not state.get("__chat_history__"):
            state["agent_1_response"] = random.choice(start_conversation_prompts)
        return state


class ConversationOutputGenerator(BaseOutputGenerator):
    @staticmethod
    def build_conversation(data: Any, state: GraspState) -> list[dict]:
        conversation = []

        for chat in data:
            conversation.append({chat["name"]: chat["response"]})
        conversation.insert(0, {"user": data[0]["request"][0]["content"]})
        return conversation

    @staticmethod
    def build_taxonomy(data: Any, state: GraspState) -> list[dict]:
        taxonomy = []
        taxonomy.append(
            {"category": state.get("category"), "subcategory": state.get("subcategory")}
        )
        return taxonomy
