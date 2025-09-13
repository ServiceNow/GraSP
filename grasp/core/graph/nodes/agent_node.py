from inspect import isclass, signature
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from grasp.core.graph.grasp_message import GraspMessage
from grasp.core.graph.nodes.llm_node import LLMNode
from grasp.logger.logger_config import logger
from grasp.utils import constants, utils


class AgentNode(LLMNode):
    """
    This node creates and executes an agent using `create_react_agent`.
    Inherits from LLMNode
    """

    REQUIRED_KEYS: list[str] = ["model", "prompt"]

    def __init__(self, node_name: str, config: dict):
        super().__init__(node_name, config)

        # Currently we support direct passing of tools which have decorator @tool and of type Langgraph's BaseTool Class.
        self.tools = []
        tool_paths = self.node_config.get("tools", [])
        if tool_paths:
            from grasp.utils.tool_utils import load_tools

            self.tools = load_tools(tool_paths)

        self.inject_system_messages = self.node_config.get("inject_system_messages", [])

    def _initialize_model(self):
        """
        Initialize the LLM model using ModelFactory by overriding the base behavior from llm_node.

        This method calls `ModelFactory.create_model` to initialize the model using the
        configuration from the node configuration.
        """

        from grasp.core.models.model_factory import ModelFactory

        self.model = ModelFactory.create_model(self.node_config["model"], constants.BACKEND)

    async def _exec_wrapper(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_factory = utils.get_graph_factory(constants.BACKEND)

        # Pre-processing
        state = (
            self.pre_process().apply(state)
            if isclass(self.pre_process)
            else self.pre_process(state)
        )
        chat_history_enabled = self.node_config.get("chat_history", False)

        # Generate and inject prompt
        prompt_tmpl = self._generate_prompt(state)
        prompt = prompt_tmpl.invoke(state)
        prompt = self._inject_history(state, prompt)

        request_msgs = graph_factory.convert_to_chat_format(prompt.to_messages())
        chat_history = state.get(constants.VAR_CHAT_HISTORY, [])
        logger.info(f"Chat history length: {len(chat_history)} at {self.name}")

        # Construct system prompt
        full_agent_prompt = self._compose_agent_prompt(state, prompt)

        # Create the ReAct agent
        agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=full_agent_prompt,
            name=state.get("_agent_name"),
        )

        # Remove redundant system message at the beginning
        if isinstance(prompt.messages[0], SystemMessage):
            prompt.messages = prompt.messages[1:]

        # Run the agent
        response = await agent.ainvoke({"messages": prompt.to_messages()})
        ai_response = None
        # pick the last AIMessage
        for m in response["messages"]:
            if isinstance(m, AIMessage):
                ai_response = m
        responseMsg = GraspMessage(ai_response)

        # Post-processing
        post_process_sig = (
            signature(self.post_process().apply)
            if isclass(self.post_process)
            else signature(self.post_process)
        )
        if len(post_process_sig.parameters) == 1:
            updated_state = (
                self.post_process().apply(responseMsg)
                if isclass(self.post_process)
                else self.post_process(ai_response)  # type: ignore
            )
        else:
            updated_state = (
                self.post_process().apply(responseMsg, state)
                if isclass(self.post_process)
                else self.post_process(ai_response, state)  # type: ignore
            )

        # Store chat history
        if chat_history_enabled:
            if not updated_state.get(constants.VAR_CHAT_HISTORY):
                updated_state[constants.VAR_CHAT_HISTORY] = []
            updated_state[constants.VAR_CHAT_HISTORY].append(
                {
                    constants.KEY_NAME: self.name,
                    constants.KEY_REQUEST: request_msgs,
                    constants.KEY_RESPONSE: graph_factory.get_message_content(responseMsg),
                }
            )

        return updated_state

    def _compose_agent_prompt(self, state: dict[str, Any], prompt) -> str:
        """
        Combines the base agent prompt with conditional injections based on chat history length.
        """
        messages = prompt.to_messages()
        base_prompt_candidate = next(
            (msg.content for msg in messages if isinstance(msg, SystemMessage)), None
        ) or state.get("_agent_prompt", "")
        # Ensure we always operate on a string; content can sometimes be a list for multimodal
        base_prompt_str: str = (
            base_prompt_candidate
            if isinstance(base_prompt_candidate, str)
            else str(base_prompt_candidate)
        )
        chat_history = state.get(constants.VAR_CHAT_HISTORY, [])

        for inject in self.inject_system_messages:
            for turn_index, message in inject.items():
                if len(chat_history) // 2 == turn_index:
                    base_prompt_str = f"{base_prompt_str}\n{message}"
                    break

        return base_prompt_str

    def to_backend(self) -> Any:
        """
        Convert the Node object to backend platform specific Runnable object.

        Returns:
             Any: platform specific runnable object like Runnable in LangGraph.
        """
        return utils.backend_factory.create_llm_runnable(self._exec_wrapper)

    def validate_node(self):
        """
        Override the method to add required validation for this Node type
        It throws Exception
        Returns:
            None
        """

        # validate the required keys
        self.validate_config_keys(self.REQUIRED_KEYS, self.node_type, self.node_config)
