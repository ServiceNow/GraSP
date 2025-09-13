import re
from typing import Any

from transformers import GPT2TokenizerFast

from grasp.core.graph.functions.edge_condition import EdgeCondition
from grasp.core.graph.functions.lambda_function import LambdaFunction
from grasp.core.graph.functions.node_processor import (
    NodePostProcessorWithState,
    NodePreProcessor,
)
from grasp.core.graph.grasp_message import GraspMessage
from grasp.core.graph.grasp_state import GraspState
from grasp.logger.logger_config import logger
from grasp.processors.data_transform import DataTransform
from grasp.utils import constants

tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/gpt-4")


class ImagesMetadata(DataTransform):
    def name(self) -> str:
        return "ImagesMetadata"

    def transform(
        self, data: list[dict[str, Any]], params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Store metadata for images in the state.
        """
        image_field = params.get("image_field", None)
        if not image_field:
            raise ValueError("image_field parameter is required")
        for item in data:
            item["num_images"] = (
                len(item[image_field]) if isinstance(item[image_field], list) else 1
            )
        return data


class ImagesPreProcessor(NodePreProcessor):
    def apply(self, state: GraspState) -> GraspState:
        """
        Prepares the state for processing images by initializing necessary fields.
        """
        state["image"] = state["images"][state["loop_count"]]
        return state


class ImageLoopChecker(EdgeCondition):
    def apply(state: GraspState) -> str:
        if state["loop_count"] < state["num_images"]:
            return "extract_text"
        return "token_checker"


class ExtractTextPostProcessor(NodePostProcessorWithState):
    def apply(self, response: GraspMessage, state: GraspState) -> GraspState:
        ocr_text = self.parse_non_json_text(response.message.content)
        if not state.get("ocr_texts"):
            state["ocr_texts"] = []
        state["ocr_texts"].append(ocr_text)
        return state

    @staticmethod
    def parse_non_json_text(text, tag="ANSWER"):
        matches = re.search(f"<{tag}>([\s\S]*?)<\/{tag}>", text)
        if matches:
            return matches.group(1).strip()


class UpdateLoopCount(LambdaFunction):
    def apply(lambda_node_dict: dict, state: GraspState):
        """
        Updates the loop count in the state.
        """
        state["loop_count"] += 1
        return state


# -----------------------------------------------------------------------------------------
# check if text extracted, calculate the token count
# also join the documents extracted from multiple images
class TokenChecker(LambdaFunction):
    def apply(lambda_node_dict: dict, state: GraspState):
        texts = state["ocr_texts"]
        documents = ""
        if not texts or len(texts) == 0 or texts[0] is None:
            token_count = 0
        else:
            for i, text in enumerate(texts, start=1):
                documents += f"DOCUMENT NUMBER {i}: \n\n{text}\n\n"
            state["documents"] = documents

            tokens = tokenizer.encode(documents)
            token_count = len(tokens)

        # add 200 tokens for extra prompt from yaml node
        state["token_count"] = token_count + 200
        state["documents"] = documents
        return state


class ShouldGenerateQuestion(EdgeCondition):
    def apply(state: GraspState) -> str:
        if state["token_count"] == 0:
            return constants.GRASP_END
        return "generate_question"


class QuestionExtractProcessor(NodePostProcessorWithState):
    def apply(self, response: GraspMessage, state: GraspState) -> GraspState:
        response_text = response.message.content
        parsed = []
        question_types = []
        for i in range(1, 4):
            q_start_tag = f"<question{i}>"
            q_end_tag = f"</question{i}>"
            type_start_tag = "<question_type>"
            type_end_tag = "</question_type>"
            ev_end_tag = "</supporting_evidence>"
            try:
                parsed.append(
                    response_text.split(q_start_tag)[1]
                    .split(q_end_tag)[0]
                    .split(ev_end_tag)[1]
                    .strip()
                )
                question_types.append(
                    response_text.split(q_start_tag)[1]
                    .split(type_start_tag)[1]
                    .split(type_end_tag)[0]
                    .strip()
                )
            except IndexError:
                # If the expected format is not found, return an empty string or handle the error as needed
                continue
        state["generated_questions"] = parsed
        state["question_response_text"] = response_text
        state["question_types"] = question_types
        state["num_questions"] = len(parsed)
        return state


######### Answer node related classes #########################
class SetCurrentQuestion(NodePreProcessor):
    def apply(self, state: GraspState) -> GraspState:
        current_pointer = state["question_counter"]
        logger.info(f"Current question pointer: {current_pointer}")
        state["question"] = state["generated_questions"][current_pointer]
        return state


class AnswerExtractProcessor(NodePostProcessorWithState):
    def apply(self, response: GraspMessage, state: GraspState) -> GraspState:
        # initialization
        response_text = response.message.content
        if not state.get("generated_responses"):
            state["generated_responses"] = []
        if not state.get("answers"):
            state["answers"] = []
        if not state.get("thinking"):
            state["thinking"] = []

        # try to extract think token content
        try:
            thinking = response_text.split("<think>")[1].split("</think>")[0]
            try:
                answer = response_text.split("</think>")[1]
            except Exception:
                # could not generate completely
                response_text = "TOO LONG"
                thinking = "Response too long"
                answer = "TOO LONG"
        except Exception:
            # If the expected format is not found, set thinking as empty and remaining part as answer
            thinking = ""
            answer = response_text
        state["generated_responses"].append(response_text)
        state["answers"].append(answer)
        state["thinking"].append(thinking)
        return state


####### Looping related classes ##########
class QuestionLoopChecker(EdgeCondition):
    def apply(state: GraspState) -> str:
        if state["question_counter"] < state["num_questions"]:
            logger.info(
                f"More questions to ask. Go to answer generation. Question counter: {state['question_counter']}"
            )
            return "generate_more_answers"
        return constants.GRASP_END


class UpdateQuestionCounter(LambdaFunction):
    def apply(lambda_node_dict: dict, state: GraspState):
        """
        Updates the loop count in the state.
        """
        state["question_counter"] += 1
        logger.info(f"Updated question counter: {state['question_counter']}")
        return state
