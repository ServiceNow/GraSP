from inspect import isclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from sygra.core.graph.nodes.base_node import BaseNode
from sygra.core.graph.sygra_message import SygraMessage
from sygra.logger.logger_config import logger
from sygra.utils import constants, utils
from sygra.utils.audio_utils import expand_audio_item
from sygra.utils.image_utils import expand_image_item


class LLMNode(BaseNode):
    """
    This node is used to call LLM model from graph.
    """

    # keys required in the llm node configuration
    REQUIRED_KEYS: list[str] = ["model", "prompt"]

    def __init__(self, node_name: str, config: dict):
        """
        LLMNode constructor.

        Args:
            node_name: Name of the node, defined as key under "nodes" in the YAML file.
            config: Node configuration defined in YAML file as the node value.
        """
        super().__init__(node_name, config)

        self.input_key = self.node_config.get("input_key", "messages")
        self.output_keys = self.node_config.get(constants.GRAPH_OUTPUT_KEY, "messages")
        self.output_role = self.node_config.get("output_role", "assistant")
        assert (
            self.node_config.get("output_vars") is None
        ), f"output_vars is not supported, use {constants.GRAPH_OUTPUT_KEY}."

        if self.output_keys and isinstance(self.output_keys, list):
            assert (
                "post_process" in self.node_config
            ), "Post processor is needed for multiple output keys."

        self.role_cls_map = {
            "user": HumanMessage,
            "assistant": AIMessage,
            "system": SystemMessage,
        }

        self.pre_process = self._default_llm_pre_process
        if "pre_process" in self.node_config:
            self.pre_process = utils.get_func_from_str(self.node_config["pre_process"])

        self.post_process = self._default_llm_post_process
        if "post_process" in self.node_config:
            self.post_process = utils.get_func_from_str(self.node_config["post_process"])

        self._initialize_model()

        self.task_name = utils.current_task
        self.graph_properties = utils.get_graph_properties(self.task_name)

        # Currently we support direct passing of tools which have decorator @tool and of type Langgraph's BaseTool Class.
        self.tools = []
        tool_paths = self.node_config.get("tools", [])
        if tool_paths:
            from sygra.utils.tool_utils import load_tools

            self.tools = load_tools(tool_paths)

    def _initialize_model(self):
        """
        Initialize the LLM model using ModelFactory.

        This method calls `ModelFactory.get_model` to initialize the model using the
        configuration from the node configuration.
        """
        from sygra.core.models.model_factory import ModelFactory

        self.model = ModelFactory.get_model(self.node_config["model"])

    def _default_llm_pre_process(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.input_key not in state or not state[self.input_key]:
            state[self.input_key] = []
        return state

    def _default_llm_post_process(
        self, response: AIMessage, state: dict[str, Any]
    ) -> dict[str, Any]:
        # only if post processor is not defined
        output_dict: dict[str, Any] = {}
        if self.output_keys == "messages":
            output_dict["messages"] = [
                self.role_cls_map[self.output_role](response.content, name=self.name)
            ]
        else:
            output_dict[self.output_keys] = response.content
        return output_dict

    def _generate_prompt(self, state: dict[str, Any]):
        messages = utils.convert_messages_from_config_to_chat_format(self.node_config["prompt"])
        return self._generate_prompt_tmpl_from_msg(state, messages)

    def _inject_history_multiturn(self, state: dict[str, Any], msg_list, window_size: int = 5):
        updated_msg_list = []
        chat_history = state.get(constants.VAR_CHAT_HISTORY, [])

        # Only keep the last `window_size` turns
        recent_history = chat_history[-window_size:]

        for entry in recent_history:
            user_msg_content = entry[constants.KEY_REQUEST][-1]["content"]
            assistant_msg_content = entry[constants.KEY_RESPONSE]

            user_message = HumanMessage(content=user_msg_content)
            assistant_message = AIMessage(content=assistant_msg_content)

            updated_msg_list.extend([user_message, assistant_message])

        # Append the incoming messages to the end
        updated_msg_list.extend(msg_list)
        return updated_msg_list

    def _inject_history_singleturn(self, state: dict[str, Any], msg_list, window_size: int = 5):
        prompt_lines = constants.PREFIX_SINGLETURN_CONV.copy()
        updated_msg_list = []
        chat_history = state[constants.VAR_CHAT_HISTORY]
        ## Adding chat_history[0] with window sized history because organizer instruction is at chat_history[0]
        recent_history = [chat_history[0]] + chat_history[-window_size:] if chat_history else []
        for i, record in enumerate(recent_history):
            agent_name = record["name"]
            if i == 0:
                user_msg = next(
                    (r["content"] for r in record["request"] if r["role"] == "user"), ""
                )
                prompt_lines.append(f"Organizer: {user_msg}")
            prompt_lines.append(f"{agent_name}: {record['response']}")
        prompt_lines.append(constants.SUFFIX_SINGLETURN_CONV.format(self.name))
        logger.info(f"Injecting history with prompt count: {len(prompt_lines)}")
        final_prompt_text = "\n".join(prompt_lines)
        final_prompt_msg = HumanMessage(content=final_prompt_text)
        updated_msg_list.append(final_prompt_msg)
        return updated_msg_list

    def _inject_history(self, state: dict[str, Any], prompt):
        msg_list = prompt.to_messages()
        updated_msg_list = msg_list
        chat_conversation = self.graph_properties.get("chat_conversation", None)
        chat_history_window_size = self.graph_properties.get("chat_history_window_size", 5)
        chat_history_enabled = self.node_config.get("chat_history", False)
        # Inject Chat history into msg_list
        if chat_conversation == constants.CHAT_CONVERSATION_MULTITURN:
            if chat_history_enabled and len(state[constants.VAR_CHAT_HISTORY]) > 0:
                updated_msg_list = self._inject_history_multiturn(
                    state, msg_list, chat_history_window_size
                )
        elif chat_conversation == constants.CHAT_CONVERSATION_SINGLETURN:
            if chat_history_enabled and len(state[constants.VAR_CHAT_HISTORY]) > 0:
                updated_msg_list = self._inject_history_singleturn(
                    state, msg_list, chat_history_window_size
                )
        prompt.messages = updated_msg_list
        return prompt

    def _generate_prompt_tmpl_from_msg(
        self, state: dict[str, Any], chat_frmt_messages
    ) -> ChatPromptTemplate:
        """
        Build a ChatPromptTemplate from message config and expand list-based image_url variables.

        Args:
            state: full graph state
            chat_frmt_messages: internal message format from YAML
        """
        for message in chat_frmt_messages:
            contents = message["content"]
            # if it's a normal text conversation then no need to expand
            if isinstance(contents, str):
                continue
            expanded_contents = []
            for item in contents:
                if item["type"] == "image_url":
                    expanded_contents.extend(expand_image_item(item, state))
                elif item["type"] == "audio_url":
                    expanded_contents.extend(expand_audio_item(item, state))
                else:
                    expanded_contents.append(item)

            message["content"] = expanded_contents

        messages = utils.convert_messages_from_chat_format_to_langchain(chat_frmt_messages)
        prompt = ChatPromptTemplate.from_messages(
            [*messages, MessagesPlaceholder(variable_name=self.input_key)],
        )
        return prompt.partial(**state)

    async def _exec_wrapper(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Entry method when node is executed in the graph.

        Args:
            state: State of the node.

        Returns:
            None
        """
        graph_factory = utils.get_graph_factory(constants.BACKEND)
        # preprocessor - if it is a class, call apply method
        state = (
            self.pre_process().apply(state)
            if isclass(self.pre_process)
            else self.pre_process(state)
        )
        chat_history_enabled = self.node_config.get("chat_history", False)
        prompt_tmpl = self._generate_prompt(state)
        # get the prompt from template
        prompt = prompt_tmpl.invoke(state)
        prompt = self._inject_history(state, prompt)

        # convert the request into chat format to store for multi turn
        request_msgs = graph_factory.convert_to_chat_format(prompt.to_messages())
        # now call the llm server
        response = await self.model.ainvoke(prompt, tools=self.tools, tool_choice="auto")
        # wrap the message to pass to the class - new implementation
        responseMsg = SygraMessage(response)

        # Call post-processor with best effort: try (resp, state) then fallback to (resp)
        try:
            updated_state = (
                self.post_process().apply(responseMsg, state)
                if isclass(self.post_process)
                else self.post_process(response, state)
            )
        except TypeError:
            updated_state = (
                self.post_process().apply(responseMsg)
                if isclass(self.post_process)
                else self.post_process(response)  # type: ignore
            )

        # Store chat history if enabled for this node
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
