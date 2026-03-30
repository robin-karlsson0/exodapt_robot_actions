import json
import os
import time
from datetime import datetime
from enum import Enum

import rclpy
from exodapt_robot_interfaces.action import ReplyAction
from exodapt_robot_pt import reply_action_pt
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

RESULT_KEY = 'reply'
THOUGHT_KEY = 'thought'
LANG_CODE_KEY = 'lang_code'


class StreamingState(Enum):
    """States for JSON streaming parser."""
    WAITING_FOR_JSON_START = "waiting_for_json_start"
    PARSING_LANG_CODE_KEY = "parsing_lang_code_key"
    CAPTURING_LANG_CODE = "capturing_lang_code"
    PARSING_REPLY_KEY = "parsing_reply_key"
    PARSING_JSON_HEADER = "parsing_json_header"  # legacy fallback path
    STREAMING_CONTENT = "streaming_content"
    JSON_COMPLETE = "json_complete"
    FALLBACK_RAW = "fallback_raw"


class JSONStreamParser:
    """Parser for extracting clean content from JSON-formatted LLM streams.

    This parser implements a state machine to detect and extract the content
    from JSON responses in the format
    {"lang_code": "code", "reply_text": "actual content"} while streaming,
    allowing clean feedback without JSON artifacts.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset parser state for a new stream."""
        self.state = StreamingState.WAITING_FOR_JSON_START
        self.buffer = ""
        # Legacy single-key header target (fallback path)
        self.header_target = '{' + f'"{RESULT_KEY}": "'
        self.header_pos = 0
        self.escape_next = False
        # Two-field header targets
        self._lang_key_target = f'"{LANG_CODE_KEY}": "'
        self._lang_key_pos = 0
        self._reply_key_target = f', "{RESULT_KEY}": "'
        self._reply_key_pos = 0
        # Detected language code (populated during streaming)
        self.detected_lang_code: str = ''

    def process_chunk(self, chunk: str) -> str:
        """Process a chunk and return clean content to send as feedback.

        Handles two JSON formats:

        Two-field (multilingual):
            {"lang_code": "en", "reply": "Hello world!"}
        Single-field (legacy fallback):
            {"reply": "Hello world!"}

        Args:
            chunk (str): Raw chunk from LLM stream

        Returns:
            str: Clean reply content to send as feedback (may be empty).
                 Side-effect: populates self.detected_lang_code once the
                 lang_code value is fully captured.
        """
        if not chunk:
            return ""

        clean_content = ""

        for char in chunk:
            # ----------------------------------------------------------------
            # State: WAITING_FOR_JSON_START
            # ----------------------------------------------------------------
            if self.state == StreamingState.WAITING_FOR_JSON_START:
                if char == '{':
                    # Try the two-field path first (lang_code before reply)
                    self.state = StreamingState.PARSING_LANG_CODE_KEY
                    self._lang_key_pos = 0
                elif char.strip():  # Non-whitespace outside JSON → raw text
                    self.state = StreamingState.FALLBACK_RAW
                    clean_content += char

            # ----------------------------------------------------------------
            # State: PARSING_LANG_CODE_KEY  — matching '"lang_code": "'
            # ----------------------------------------------------------------
            elif self.state == StreamingState.PARSING_LANG_CODE_KEY:
                expected = self._lang_key_target[self._lang_key_pos]
                if char == expected:
                    self._lang_key_pos += 1
                    if self._lang_key_pos == len(self._lang_key_target):
                        # Full lang_code key matched → start capturing value
                        self.state = StreamingState.CAPTURING_LANG_CODE
                else:
                    # Mismatch — this might be the legacy single-key format.
                    # Replay as if we are in PARSING_JSON_HEADER from pos 1.
                    self.state = StreamingState.PARSING_JSON_HEADER
                    self.header_pos = 1
                    # Feed the already-seen chars plus the failing char into
                    # the legacy path by reconstructing what was consumed.
                    # The buffered prefix was self._lang_key_target[:self._lang_key_pos]  # noqa
                    # followed by `char`.  Delegate to legacy handler.
                    reconstructed = self._lang_key_target[:self.
                                                          _lang_key_pos] + char
                    for rc in reconstructed:
                        if self.state != StreamingState.PARSING_JSON_HEADER:
                            break
                        if self.header_pos < len(self.header_target):
                            if rc == self.header_target[self.header_pos]:
                                self.header_pos += 1
                                if self.header_pos == len(self.header_target):
                                    self.state = StreamingState.STREAMING_CONTENT  # noqa
                            else:
                                self.state = StreamingState.FALLBACK_RAW
                                buffered = self.header_target[:self.header_pos]
                                clean_content += buffered + rc

            # ----------------------------------------------------------------
            # State: CAPTURING_LANG_CODE  — chars until unescaped '"'
            # ----------------------------------------------------------------
            elif self.state == StreamingState.CAPTURING_LANG_CODE:
                if self.escape_next:
                    self.detected_lang_code += char
                    self.escape_next = False
                elif char == '\\':
                    self.escape_next = True
                elif char == '"':
                    # End of lang_code value → now look for the reply key
                    self.state = StreamingState.PARSING_REPLY_KEY
                    self._reply_key_pos = 0
                else:
                    self.detected_lang_code += char

            # ----------------------------------------------------------------
            # State: PARSING_REPLY_KEY  — matching ', "reply": "'
            # ----------------------------------------------------------------
            elif self.state == StreamingState.PARSING_REPLY_KEY:
                expected = self._reply_key_target[self._reply_key_pos]
                if char == expected:
                    self._reply_key_pos += 1
                    if self._reply_key_pos == len(self._reply_key_target):
                        # Full reply key matched → stream content
                        self.state = StreamingState.STREAMING_CONTENT
                        self.escape_next = False
                else:
                    # Unexpected char after lang_code — fall back to raw
                    self.state = StreamingState.FALLBACK_RAW
                    buffered = self._reply_key_target[:self._reply_key_pos]
                    clean_content += buffered + char

            # ----------------------------------------------------------------
            # State: PARSING_JSON_HEADER  — legacy single-key path
            # ----------------------------------------------------------------
            elif self.state == StreamingState.PARSING_JSON_HEADER:
                if self.header_pos < len(self.header_target):
                    if char == self.header_target[self.header_pos]:
                        self.header_pos += 1
                        if self.header_pos == len(self.header_target):
                            self.state = StreamingState.STREAMING_CONTENT
                    else:
                        self.state = StreamingState.FALLBACK_RAW
                        buffered = self.header_target[:self.header_pos]
                        clean_content += buffered + char

            # ----------------------------------------------------------------
            # State: STREAMING_CONTENT  — emit reply chars
            # ----------------------------------------------------------------
            elif self.state == StreamingState.STREAMING_CONTENT:
                if self.escape_next:
                    clean_content += char
                    self.escape_next = False
                elif char == '\\':
                    clean_content += char
                    self.escape_next = True
                elif char == '"':
                    self.state = StreamingState.JSON_COMPLETE
                else:
                    clean_content += char

            # ----------------------------------------------------------------
            # State: FALLBACK_RAW  — pass everything through
            # ----------------------------------------------------------------
            elif self.state == StreamingState.FALLBACK_RAW:
                clean_content += char

            # JSON_COMPLETE: stop processing

        return clean_content


class ReplyActionServer2(Node):
    """
    ROS 2 action server for generating contextual robot replies using LLM call.

    This action server receives robot state information through the ReplyAction
    interface and generates appropriate natural language responses by querying
    an inference server. The server provides real-time streaming
    feedback of the generated response and returns the complete reply as the
    action result.

    The node processes robot state data using a prompt template function and
    interfaces with an inference server running an LLM model to generate contextually
    appropriate responses for human-robot interaction scenarios.

    Parameters:
        action_server_name (str): Name of the action server (default: 'reply_action_server')
        reply_action_topic (str): Topic for publishing ReplyAction results (default: '/reply_action')
        inference_server_type (str): Type of inference server: 'tgi' or 'vllm' (default: 'tgi')
        inference_server_url (str): URL of the inference server (default: 'http://localhost:8000')
        max_tokens (int): Maximum number of tokens to generate (default: 1024)
        llm_temp (float): Temperature parameter for response generation (default: 0.6)
        llm_seed (int): Random seed for reproducible generation (default: 14)

    Action Interface:
        Action Type: ReplyAction
        Goal: Robot state information and optional instruction
        Feedback: Streaming response chunks during generation
        Result: Complete generated reply text

    Attributes:
        action_server_name (str): Name of the action server
        reply_action_topic (str): Topic for publishing ReplyAction results
        log_pred_io_pth (str): Directory path where LLM prediction
            (input, output) will be logged as individual JSON files. If empty,
            no logging will be performed.
            Ex: 'log/action_reply/'
        inference_server_type (str): Type of inference server ('tgi' or 'vllm')
        client: Inference client (InferenceClient for TGI, OpenAI for vLLM)
        inference_server_url (str): Configured inference server URL
        max_tokens (int): Maximum tokens for generation
        llm_temp (float): Temperature for response sampling
        llm_seed (int): Seed for deterministic generation (-1 for random seed)
    """  # noqa: E501

    def __init__(self, **kwargs):
        """
        Initialize the ReplyActionServer2 node.

        Sets up the ROS 2 action server, declares and retrieves parameters for
        inference server configuration, and initializes the inference
        client. The server is configured to handle ReplyAction requests with
        streaming response capabilities.

        Args:
            **kwargs: Additional keyword arguments passed to the parent Node
                constructor

        Raises:
            ConnectionError: If unable to connect to the inference server
            ParameterException: If required parameters are invalid
        """
        super().__init__('reply_action_2', **kwargs)

        self.declare_parameter('action_server_name', 'reply_action_server')
        self.declare_parameter('reply_action_topic', '/reply_action')
        self.declare_parameter('log_pred_io_pth', '')
        self.declare_parameter('inference_server_type', 'vllm')
        self.declare_parameter('inference_server_url', 'http://localhost:8000')
        self.declare_parameter('max_tokens', 1024)
        self.declare_parameter('llm_temp', 0.6)
        self.declare_parameter('llm_seed', -1)
        self.action_server_name = self.get_parameter(
            'action_server_name').value
        self.log_pred_io_pth = self.get_parameter('log_pred_io_pth').value
        self.reply_action_topic = self.get_parameter(
            'reply_action_topic').value
        self.inference_server_type = self.get_parameter(
            'inference_server_type').value
        self.inference_server_url = self.get_parameter(
            'inference_server_url').value
        self.max_tokens = self.get_parameter('max_tokens').value
        self.llm_temp = self.get_parameter('llm_temp').value

        # Handle llm_seed parameter - support multiple ways to specify None
        llm_seed_param = self.get_parameter('llm_seed').value
        if llm_seed_param == -1:
            self.llm_seed = None
        else:
            self.llm_seed = int(llm_seed_param)

        self.get_logger().info(
            'ReplyActionServer2 initializing\n'
            'Parameters:\n'
            f'  action_server_name: {self.action_server_name}\n'
            f'  reply_action_topic: {self.reply_action_topic}\n'
            f'  log_pred_io_pth: {self.log_pred_io_pth}\n'
            f'  inference_server_type: {self.inference_server_type}\n'
            f'  Inference server url: {self.inference_server_url}\n'
            f'  max_tokens={self.max_tokens}\n'
            f'  llm_temp={self.llm_temp}\n'
            f'  llm_seed={self.llm_seed}')

        # Configure inference server type and corresponding client/callback
        if self.inference_server_type.lower() == 'vllm':
            self._setup_vllm_client()
            self.execute_callback = self.execute_callback_vllm
        else:
            raise ValueError("Unsupported inference_server_type: "
                             f"{self.inference_server_type}. "
                             "Supported types: 'vllm'")

        self._action_server = ActionServer(
            self,
            ReplyAction,
            self.action_server_name,
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self._reply_action_pub = self.create_publisher(
            String,
            self.reply_action_topic,
            10,
        )

        self._reply_thought_pub = self.create_publisher(
            String,
            self.reply_action_topic + '_thought',
            10,
        )

        # Create log directory
        if self.log_pred_io_pth:
            if not os.path.exists(self.log_pred_io_pth):
                os.makedirs(self.log_pred_io_pth)

        self.cancellation_msg = '<REPLY_CANCELLED>'

    def _setup_vllm_client(self):
        """Setup OpenAI-compatible client for vLLM server."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI client is required for vLLM support. "
                              "Install with: pip install openai")

        self.client = OpenAI(
            base_url=f"{self.inference_server_url}/v1",
            api_key="dummy-key"  # vLLM doesn't require a real API key
        )
        self._vllm_model = self.client.models.list().data[0].id
        self.get_logger().info(
            f"Configured vLLM client for: {self.inference_server_url}/v1")

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        # This server allows multiple goals in parallel
        self.get_logger().debug('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel and action."""
        self.get_logger().debug('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback_vllm(self, goal_handle):
        """
        Execute callback for processing ReplyAction goals using vLLM inference.

        This async callback processes incoming action goals containing robot
        state information, formats the state data into a prompt using the
        reply_action_pt function, and queries the vLLM server for response
        generation. The method streams response chunks as feedback and returns
        the complete reply.

        Args:
            goal_handle (ServerGoalHandle): ROS 2 action goal handle containing:
                - request.state: Robot state information for context
                - request.instruction: Optional instruction (not currently
                  implemented)

        Returns:
            ReplyAction.Result: Action result containing:
                - reply (str): Complete generated response text

        Raises:
            ConnectionError: If vLLM server is unreachable during inference
            TimeoutError: If inference exceeds expected duration
            ValueError: If robot state data is invalid or malformed
        """
        self.get_logger().debug('Executing ReplyActionServer with vLLM...')

        # Unpack ReplyAction.Goal() msg
        goal = goal_handle.request
        state_json = goal.state

        # Parse JSON state payload and extract ASR metadata
        try:
            state_dict = json.loads(state_json)
            pre = state_dict.get('pre', '')
            chunks = state_dict.get('chunks', '')
            dyn = state_dict.get('dyn', '')
            asr_metadata = state_dict.get('metadata', {})
            full_state_str = pre + chunks + dyn
        except json.JSONDecodeError:
            self.get_logger().warn(
                "ReplyActionServer received invalid JSON state")  # noqa
            # Fallback to assuming the state is purely plain text
            full_state_str = state_json
            asr_metadata = {}

        if len(goal.instruction) > 0:
            self.get_logger().warn('Providing instruction is not implemented')

        llm_input = reply_action_pt(full_state_str)

        t0 = time.time()

        # vLLM uses OpenAI-compatible API
        # Prepare arguments for chat completion
        # Build extra_body to tunnel the ASR metadata to the proxy
        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        }
        if asr_metadata:
            extra_body["asr_metadata"] = asr_metadata

        chat_args = {
            'model': self._vllm_model,
            'messages': [
                {
                    'role': 'user',
                    'content': llm_input
                },
            ],
            'stream': True,
            'stream_options': {
                "include_usage": True
            },
            'max_tokens': self.max_tokens,
            'temperature': self.llm_temp,
            'extra_body': extra_body,
        }

        # Only include seed if it's not None
        if self.llm_seed is not None:
            chat_args['seed'] = self.llm_seed

        output = self.client.chat.completions.create(**chat_args)

        streaming_resp_buffer = []
        feedback_msg = ReplyAction.Feedback()
        was_cancelled = False

        # JSON parser for clean content extraction
        json_parser = JSONStreamParser()

        tokens = -1
        for chunk in output:
            # Check for cancellation request before processing each chunk
            if goal_handle.is_cancel_requested:
                was_cancelled = True
                self.get_logger().info('Reply action cancelled')
                break

            # Check for usage information in the final chunk
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                tokens = int(chunk.usage.total_tokens)
                continue

            # vLLM response format is similar to OpenAI
            raw_content = chunk.choices[0].delta.content
            if raw_content is not None:
                streaming_resp_buffer.append(raw_content)

                # Process chunk through JSON parser for clean feedback
                clean_content = json_parser.process_chunk(raw_content)

                # Send clean feedback (only if there's content to send)
                if clean_content:
                    feedback_msg.streaming_resp = clean_content
                    feedback_msg.lang_code = json_parser.detected_lang_code
                    goal_handle.publish_feedback(feedback_msg)

        # Inference time
        t1 = time.time()
        dt = t1 - t0

        # Concatenate chunks and prepare result
        resp = ''.join(streaming_resp_buffer)

        # Parse reply from output JSON
        try:
            resp_json = json.loads(resp)
            reply = resp_json[RESULT_KEY]
            thought = resp_json[THOUGHT_KEY]

            self.get_logger().debug(
                f'Successfully parsed JSON\n  reply: {reply}\n  thought: {thought}'  # noqa
            )
        except json.JSONDecodeError as e:
            self.get_logger().warn(
                f'Failed to parse JSON response: {e}. Using raw response.')
            reply = resp
            thought = ''
        except KeyError as e:
            self.get_logger().warn(
                f'Missing expected key in JSON response: {e}. '
                f'Using raw response.')
            reply = resp
            thought = ''
        except Exception as e:
            self.get_logger().warn(
                f'Unexpected error parsing JSON response: {e}. '
                f'Using raw response.')
            reply = resp
            thought = ''

        # Handle cancellation vs completion
        if was_cancelled:
            reply += self.cancellation_msg
            thought = ''
            goal_handle.canceled()
            self.get_logger().debug(f'Reply canceled with partial response: '
                                    f'{reply} ({dt:.2f} s)')
        else:
            goal_handle.succeed()
            tokens_str = await self.format_number(tokens)
            self.get_logger().info(
                f'{YELLOW}Reply: {reply} ({dt:.2f} s, {tokens_str} tokens){RESET}'  # noqa
            )
            self.get_logger().info(
                f'{BLUE}Thought: {thought} ({dt:.2f} s, {tokens_str} tokens){RESET}'  # noqa
            )

        result_msg = ReplyAction.Result()
        result_msg.reply = reply

        # Publish result (even for cancelled actions)
        result_msg_str = String()
        result_msg_str.data = reply
        self._reply_action_pub.publish(result_msg_str)

        # Ensure publishing order
        time.sleep(0.01)

        # Publish thoughts
        thought_msg = String()
        thought_msg.data = f"Robot's thoughts about the robot reply: {thought}"
        self._reply_thought_pub.publish(thought_msg)

        # Write prediction IO example to file
        if self.log_pred_io_pth:
            await self.log_pred_io(llm_input, resp, dt)

        return result_msg

    async def log_pred_io(self, llm_input: str, llm_output: str, dt: float):
        """Log LLM prediction input and output to JSON file.

        Creates timestamped JSON files containing the complete prediction
        context for model evaluation, debugging, and dataset creation. Each
        log entry includes the prompt, prediction, timing information, and
        timestamps for comprehensive tracking.

        Args:
            input (str): The formatted prompt sent to the LLM
            output (str): The predicted action token returned by the LLM
            dt (float): Inference duration in seconds

        File Format:
            JSON files named 'pred_io_{timestamp_ms}.json' containing:
            - ts: Unix timestamp in milliseconds
            - iso_ts: ISO format timestamp for human readability
            - llm_input: Complete LLM prompt string
            - llm_output: Predicted reply
            - dt: Inference duration in seconds

        Error Handling:
            Logging failures are caught and logged as errors without
            interrupting the action decision process, ensuring system
            reliability when logging is non-critical.

        Note:
            Only logs when log_pred_io_pth parameter is configured.
            Files use UTF-8 encoding with pretty-printed JSON formatting.
        """
        try:
            ts = int(time.time() * 1000)  # millisecond precision
            file_name = f'pred_io_{ts}.json'
            file_pth = os.path.join(self.log_pred_io_pth, file_name)

            log_entry = {
                'ts': ts,
                'iso_ts': datetime.now().isoformat(),
                'input': llm_input,
                'output': llm_output,
                'dt': dt,
            }

            with open(file_pth, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.get_logger().error(
                f"Failed to log prediction IO example: {e}")
            return

    @staticmethod
    async def format_number(num: int, decimals: int = 2) -> str:
        """
        Format an integer with appropriate quantity suffixes (K, M, B, T).

        Args:
            num (int): The number to format
            decimals (int): Number of decimals to display

        NOTE The underscore `_` in integer literals is a digit separator for
            readability.

        Returns:
            str: Formatted string with 1 decimal place and suffix
        """
        if num < 1000:
            return str(num)

        # Define the suffixes and their corresponding divisors
        suffixes = [
            (1_000_000_000_000, 'T'),  # Trillion
            (1_000_000_000, 'B'),  # Billion
            (1_000_000, 'M'),  # Million
            (1_000, 'K')  # Thousand
        ]

        for divisor, suffix in suffixes:
            if num >= divisor:
                result = num / divisor
                return f"{result:.{decimals}f}{suffix}"

        return str(num)


def main(args=None):
    """
    Main entry point for the ReplyActionServer ROS 2 node.

    Initializes the ROS 2 context, creates and spins the ReplyActionServer node
    using asyncio for handling async inference callbacks, and performs proper
    cleanup on shutdown. The function handles the complete node lifecycle
    including initialization, execution, and teardown.

    Args:
        args (List[str], optional): Command line arguments for ROS 2
            initialization. Defaults to None, which uses sys.argv.

    Raises:
        KeyboardInterrupt: Gracefully handles Ctrl+C shutdown
        RuntimeError: If ROS 2 initialization or node creation fails

    Example:
        $ ros2 run your_package reply_action_server
        $ ros2 run your_package reply_action_server --ros-args -p inference_server_type:=vllm -p inference_server_url:=http://gpu-server:8080
    """  # noqa: E501
    rclpy.init(args=args)

    reply_action_server = ReplyActionServer2()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()
    rclpy.spin(reply_action_server, executor=executor)

    reply_action_server.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
