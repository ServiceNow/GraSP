import json
import os
from argparse import Namespace

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from grasp.logger.logger_config import logger
from grasp.utils import constants, utils


class ConversationPreTokenizationTask:
    """
    A task for generating conversation_pretokenized strings from input messages + response.

    Args:
        input_file (str): Path to the input JSONL dataset file.
        output_dir (str): Directory where output file will be saved.
        num_records (int): Total number of records to process.
        **kwargs: Additional task-specific parameters.
    """

    def __init__(self, input_file: str, output_dir: str, num_records: int, **kwargs):
        self.input_file = input_file
        self.output_dir = output_dir
        self.num_records = num_records
        self.task_params = kwargs

    def execute(self) -> str:
        """
        Executes the conversation pretokenization task.

        Returns:
            str: Path to the output file containing the pretokenized data.
        """
        args = self._construct_args()

        tokenizer = self._set_tokenizer(args.hf_chat_template_model_id, args.hf_token)

        if self.input_file.endswith(".json"):
            data = utils.load_json_file(self.input_file)
        elif self.input_file.endswith(".jsonl"):
            data = utils.load_jsonl_file(self.input_file)
        else:
            raise ValueError("Unsupported file format for input_file.")
        df = pd.DataFrame(data)
        records = df.to_dict(orient="records")
        logger.info(f"Loaded {len(records)} records from {args.input_file}")

        pretokenized_records = []
        for record in tqdm(
            records[: args.num_records], desc="Generating conversation_pretokenized"
        ):
            inputs_pretokenized, targets_pretokenized, conversation_pretokenized = (
                self._generate_pretokenized(tokenizer, record)
            )
            record["conversation_pretokenized"] = conversation_pretokenized
            record["inputs_pretokenized"] = inputs_pretokenized
            record["targets_pretokenized"] = targets_pretokenized
            pretokenized_records.append(record)

        output_file = os.path.join(args.output_dir, "conversation_pretokenized_output.jsonl")
        with open(output_file, "w") as f:
            for record in pretokenized_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(pretokenized_records)} records to {output_file}")
        return output_file

    def _generate_pretokenized(self, tokenizer, record) -> tuple:
        """
        Helper to generate pretokenized string for a single record.
        """
        messages = record["conversation"]
        inputs = messages[:-1]  # All messages except the last one
        messages[-1]  # The last message as the target

        inputs_pretokenized = tokenizer.apply_chat_template(
            inputs, tokenize=False, add_generation_prompt=True
        ).strip()
        conversation_pretokenized = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        targets_pretokenized = conversation_pretokenized.replace(inputs_pretokenized, "")

        return inputs_pretokenized, targets_pretokenized, conversation_pretokenized

    def _set_tokenizer(self, model_id: str, token: str) -> PreTrainedTokenizerBase:
        """
        Sets the tokenizer based on the model ID.

        Args:
            model_id (str): The model ID for the tokenizer or the local path to the tokenizer.
            token (str): The Hugging Face token for authentication.

        Returns:
            PreTrainedTokenizerBase: The initialized tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        return tokenizer

    def _construct_args(self) -> Namespace:
        """
        Constructs arguments for task execution.

        Returns:
            Namespace: Config object for the task.
        """
        args = {
            "task": "data_preprocessing.conversation_pretokenization",
            "input_file": self.input_file,
            "output_dir": self.output_dir,
            "num_records": self.num_records,
            "hf_chat_template_model_id": self.task_params["hf_chat_template_model_id"],
            "hf_token": os.environ.get(constants.HF_TOKEN),
            "run_name": "conversation_pretokenization",
        }
        return Namespace(**args)
