import sys

from grasp.core.base_task_executor import DefaultTaskExecutor
from grasp.logger.logger_config import configure_logger
from grasp.core.models.custom_models import ModelParams
from grasp.core.models.model_factory import ModelFactory
from grasp.utils import utils
import argparse
import time
import ast
import json
from pathvalidate import is_valid_filename
import os
from grasp.utils.dotenv import load_dotenv

# Sometimes there is SSL retry error; to fix it: https://github.com/huggingface/transformers/issues/17611
CURL_CA_BUNDLE = os.environ.get("CURL_CA_BUNDLE", "")
REQUESTS_CA_BUNDLE = os.environ.get("REQUESTS_CA_BUNDLE", "")
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

load_dotenv(dotenv_path=".env", override=True)


def check_model_availability(task_name):
    # get all the models used in this task
    model_config_this_task = utils.get_models_used(task_name)
    # test if all the models are active, else abort the process
    for mn, mc in model_config_this_task.items():
        if mc is None:
            logger.error(f"Model {mn} has no model configuration. Exiting the process.")
            sys.exit(1)
        mc["name"] = mn
        backend = mc.get("backend", "default")

        # create model object for inference
        mod = ModelFactory.create_model(mc, backend)
        model_param = ModelParams(url=mc.get("url"), auth_token=mc.get("auth_token"))
        # inference
        status = mod.ping()
        if status != 200:
            logger.error(f"Model({mn}) is down. Aborting the process.")
            sys.exit(1)
    logger.info("Required models are up and running.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        help="Task name to synthesize data for",
        required=True,
    )
    parser.add_argument(
        "--start_index",
        "-si",
        type=int,
        default=0,
        help="Start index of the input dataset to pick records from",
    )
    parser.add_argument(
        "--num_records",
        "-n",
        type=int,
        default=10,
        help="Num of records to pick from the input dataset",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=25,
        help="Num of records to process in a batch",
    )
    parser.add_argument(
        "--checkpoint_interval",
        "-ci",
        type=int,
        default=100,
        help="Num of records after which an output file checkpoint is saved",
    )
    parser.add_argument(
        "--debug",
        "-d",
        type=ast.literal_eval,
        default=False,
        help="Enable debug mode",
    )
    parser.add_argument(
        "--clear_logs",
        "-cl",
        type=ast.literal_eval,
        default=False,
        help="Clear output logs before running the script",
    )
    parser.add_argument(
        "--output_with_ts",
        "-owt",
        type=ast.literal_eval,
        default=True,
        help="Generate output file with timestamp suffix",
    )
    parser.add_argument(
        "--run_name",
        "-rn",
        type=str,
        default="",
        help="Name of the run to be used in output file name and logs",
    )
    parser.add_argument(
        "--run_args",
        "-ra",
        type=json.loads,
        default="{}",
        help='Custom args for the run as a JSON string, e.g. \'{"key1": "value1", "key2": "value1"}\'',
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=ast.literal_eval,
        default=None,
        help="Override resumable config to force resume (True) or disable resume (False)",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        help="Output directory for the synthesized data, if not specified, defaults to the current task directory",
        default=None,
    )

    parser.add_argument(
        "--oasst",
        "-ost",
        type=bool,
        help="Boolean flag to run the OASST mapper",
        default=False,
    )

    parser.add_argument(
        "--quality",
        "-q",
        type=bool,
        help="Boolean flag to enable quality metrics",
        default=False,
    )

    args = parser.parse_args()

    start = time.time()
    task_name = args.task

    # initialize logger with/without debug mode
    configure_logger(args.debug, args.clear_logs, args.run_name)
    # this import cannot be moved to the top because logger is not yet initialized
    from grasp.logger.logger_config import logger

    logger.info("------------------------------------")
    logger.info(f"STARTING SYNTHESIS FOR TASK: {task_name}")
    logger.info("------------------------------------")
    logger.info(f"SCRIPT ARGS: {args}")

    if args.run_name:
        assert is_valid_filename(args.run_name), f"Invalid run name: {args.run_name}"

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info(f"Output directory set to: {args.output_dir}")

    # check models are available and normalize task name
    if not task_name.startswith("tasks."):
        #check if task_name is valid or "tasks.{task_name}" is valid. Whichever is valid, use that as args.task and full_task_name
        if utils.is_valid_task_name(task_name):
            full_task_name = task_name
        elif utils.is_valid_task_name(f"tasks.{task_name}"):
            full_task_name = f"tasks.{task_name}"
        else:
            logger.error(f"Invalid task name: {task_name}. Exiting the process.")
            sys.exit(1)
        check_model_availability(full_task_name)
        args.task = full_task_name
        utils.current_task = full_task_name  # Set current_task to the full task name with prefix
    else:
        check_model_availability(task_name)
        utils.current_task = task_name

    specialized_executor_cls = None
    executor_path = f"tasks.{task_name}.task_executor.TaskExecutor"
    try:
        specialized_executor_cls = utils.get_func_from_str(executor_path)
        logger.info(f"Found specialized TaskExecutor at {executor_path}")
    except (ModuleNotFoundError, AttributeError) as e:
        logger.info(
            f"No specialized TaskExecutor found for task: {task_name}. Using DefaultTaskExecutor."
        )

    task_executor = specialized_executor_cls or DefaultTaskExecutor

    logger.info(f"Running {task_executor} for task {args.task}")
    task_executor(args).execute()

    logger.info("------------------------------------")
    logger.info(
        f"SYNTHESIS COMPLETE FOR TASK: {args.task} IN {(time.time() - start):0.2f} secs"
    )
    logger.info("------------------------------------")
    os.environ["CURL_CA_BUNDLE"] = CURL_CA_BUNDLE
    os.environ["REQUESTS_CA_BUNDLE"] = REQUESTS_CA_BUNDLE
