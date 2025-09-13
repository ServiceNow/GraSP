# mixtral API Errors
import os
import sys

MIXTRAL_API_RATE_LIMIT_ERROR = "rate limit"
MIXTRAL_API_MODEL_OVERLOAD_ERROR = "model has no capacity"
INTERMEDIATE = "_intermediate."

BACKEND = "langgraph"
GRASP_START = sys.intern("__start__")
GRASP_END = sys.intern("__end__")
START = "START"
END = "END"

# Note: Root dir depends on the location of this file. Update the below variable if the file is moved.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRASP_CONFIG = os.path.join(ROOT_DIR, "config", "configuration.yaml")
PREFIX_SINGLETURN_CONV = [
    "You are participating in a roleplay conversation. Stay in character and respond naturally based on your persona.",
    "Here is the conversation so far between [Organizer, Agents]:",
]
SUFFIX_SINGLETURN_CONV = "\nNow, respond as {}. Generate only the next agent response. Do not include any label, notes or system messages. Stay in character and respond in a way that keeps the dialogue flowing naturally."
CHAT_CONVERSATION_MULTITURN = "multiturn"
CHAT_CONVERSATION_SINGLETURN = "singleturn"

# special state variables
VAR_REQUESTS = "__requests__"
VAR_RESPONSES = "__responses__"
VAR_LAST_NODES = "__last_node__"
VAR_CHAT_HISTORY = "__chat_history__"
KEY_NAME = "name"
KEY_REQUEST = "request"
KEY_RESPONSE = "response"

HF_TOKEN = "HF_TOKEN"

# output key in graph yaml
GRAPH_OUTPUT_KEY = "output_keys"

# Server constants
ERROR_PREFIX = "###SERVER_ERROR###"

# special error message when inference server is down(tgi and vllm only)
# element ai job is down
ELEMAI_JOB_DOWN = "job is not running"
# Connection error in another environment
CONNECTION_ERROR = "Connection error"

# model config file
MODEL_CONFIG_YAML = os.path.join(ROOT_DIR, "config", "models.yaml")

# model failure handling - shutdown the process after some try
HANDLE_SERVER_DOWN = True
# list of error code to handle - service is down or unavailable
SERVER_DOWN_ERROR_CODE = [404, 500, 501, 502, 503]
# validation for last n errors
MAX_FAILED_ERROR = 10
# if last n error occurs within t time, action to take(shut down)
MODEL_FAILURE_WINDOW_IN_SEC = 30

# retry the request for below http errors
RETRYABLE_HTTP_ERROR = [408, 429, 599, 444]

# all metadata variables for resumability
META_TASK_NAME = "task_name"
META_OUTPUT_FILE = "output_file"
META_POSITION = "position"
META_PROCESSED_RECORDS = "processed_records"
META_LAST_POSITION = "last_position"
META_HIGHEST_POSITION = "highest_position"
META_DATASET_TYPE = "dataset_type"
META_TOTAL_PROCESSED = "total_processed"
META_SAMPLER_CACHE = "sampler_cache"
META_COMPLETED = "completed"
META_RECORD_POSITIONS = "record_positions"
META_TIMESTAMP = "timestamp"

# model request default timeout in seconds
DEFAULT_TIMEOUT = 120

# separator for list values in environment variables
LIST_SEPARATOR = "|"

# TODO: Unused variable, to be removed later
COMPLETION_ONLY_MODELS: list[str] = []

# constants for template based payload
PAYLOAD_CFG_FILE = "config/payload_cfg.json"
PAYLOAD_JSON = "payload_json"
TEST_PAYLOAD = "test_payload"
RESPONSE_KEY = "response_key"

# constants for inference server types
INFERENCE_SERVER_TRITON = "triton"
