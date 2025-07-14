import json
import os
import time
from datetime import datetime

import rclpy
from exodapt_robot_interfaces.action import ReplyAction
from exodapt_robot_pt import reply_action_pt
from huggingface_hub import InferenceClient
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String


class ReplyActionServer(Node):
    """
    ROS 2 action server for generating contextual robot replies using LLM call.

    This action server receives robot state information through the ReplyAction
    interface and generates appropriate natural language responses by querying
    a Huggingface TGI inference server. The server provides real-time streaming
    feedback of the generated response and returns the complete reply as the
    action result.

    The node processes robot state data using a prompt template function and
    interfaces with a TGI server running an LLM model to generate contextually
    appropriate responses for human-robot interaction scenarios.

    Parameters:
        action_server_name (str): Name of the action server (default: 'reply_action_server')
        reply_action_topic (str): Topic for publishing ReplyAction results (default: '/reply_action')
        tgi_server_url (str): URL of the TGI inference server (default: 'http://localhost:5000')
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
        client (InferenceClient): Huggingface TGI inference client
        tgi_server_url (str): Configured TGI server URL
        max_tokens (int): Maximum tokens for generation
        llm_temp (float): Temperature for response sampling
        llm_seed (int): Seed for deterministic generation
    """  # noqa: E501

    def __init__(self, **kwargs):
        """
        Initialize the ReplyActionServer node.

        Sets up the ROS 2 action server, declares and retrieves parameters for
        TGI server configuration, and initializes the Huggingface inference
        client. The server is configured to handle ReplyAction requests with
        streaming response capabilities.

        Args:
            **kwargs: Additional keyword arguments passed to the parent Node
                constructor

        Raises:
            ConnectionError: If unable to connect to the TGI server
            ParameterException: If required parameters are invalid
        """
        super().__init__('reply_action', **kwargs)

        self.declare_parameter('action_server_name', 'reply_action_server')
        self.declare_parameter('reply_action_topic', '/reply_action')
        self.declare_parameter('log_pred_io_pth', '')
        self.declare_parameter('tgi_server_url', 'http://localhost:5000')
        self.declare_parameter('max_tokens', 1024)
        self.declare_parameter('llm_temp', 0.6)
        self.declare_parameter('llm_seed', 14)
        self.action_server_name = self.get_parameter(
            'action_server_name').value
        self.log_pred_io_pth = self.get_parameter('log_pred_io_pth').value
        self.reply_action_topic = self.get_parameter(
            'reply_action_topic').value
        self.tgi_server_url = self.get_parameter('tgi_server_url').value
        self.max_tokens = self.get_parameter('max_tokens').value
        self.llm_temp = self.get_parameter('llm_temp').value
        self.llm_seed = self.get_parameter('llm_seed').value

        self._action_server = ActionServer(
            self,
            ReplyAction,
            self.action_server_name,
            execute_callback=self.execute_callback_tgi,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self._reply_action_pub = self.create_publisher(
            String,
            self.reply_action_topic,
            10,
        )

        # TGI inference client
        base_url = f"{self.tgi_server_url}/v1/"
        self.client = InferenceClient(base_url=base_url)

        # Create log directory
        if self.log_pred_io_pth:
            if not os.path.exists(self.log_pred_io_pth):
                os.makedirs(self.log_pred_io_pth)

        self.cancellation_msg = '<REPLY_CANCELLED>'

        self.get_logger().info(
            'ReplyActionServer initialized\n'
            'Parameters:\n'
            f'  action_server_name: {self.action_server_name}\n'
            f'  reply_action_topic: {self.reply_action_topic}\n'
            f'  log_pred_io_pth: {self.log_pred_io_pth}\n'
            f'  TGI server url: {self.tgi_server_url}\n'
            f'  max_tokens={self.max_tokens}\n'
            f'  llm_temp={self.llm_temp}\n'
            f'  llm_seed={self.llm_seed}')

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        # This server allows multiple goals in parallel
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel and action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback_tgi(self, goal_handle):
        """
        Execute callback for processing ReplyAction goals using TGI inference.

        This async callback processes incoming action goals containing robot
        state information, formats the state data into a prompt using the
        reply_action_pt function, and queries the TGI server for response
        generation. The method streams response chunks as feedback and returns
        the complete reply.

        The callback handles the full inference pipeline including:
        - Unpacking and validating the goal message
        - Converting robot state to LLM prompt format
        - Streaming inference with real-time feedback and cancellation support
        - Timing inference duration
        - Assembling and returning the final result

        Args:
            goal_handle (ServerGoalHandle): ROS 2 action goal handle containing:
                - request.state: Robot state information for context
                - request.instruction: Optional instruction (not currently
                  implemented)

        Returns:
            ReplyAction.Result: Action result containing:
                - reply (str): Complete generated response text

        Raises:
            ConnectionError: If TGI server is unreachable during inference
            TimeoutError: If inference exceeds expected duration
            ValueError: If robot state data is invalid or malformed

        Note:
            The instruction field in the goal is currently not implemented and
            will generate a warning if provided. Response streaming provides
            real-time feedback but may introduce latency depending on model
            size and server configuration. The action can be canceled at any
            point during streaming, allowing immediate termination of ongoing
            inference.
        """
        self.get_logger().info('Executing ReplyActionServer...')

        # Unpack ReplyAction.Goal() msg
        goal = goal_handle.request
        state = goal.state

        if len(goal.instruction) > 0:
            self.get_logger().warn('Providing instruction is not implemented')

        llm_input = reply_action_pt(state)

        t0 = time.time()

        output = self.client.chat.completions.create(
            model="tgi",
            messages=[
                {
                    "role": "user",
                    "content": llm_input
                },
            ],
            stream=True,
            max_tokens=self.max_tokens,
            temperature=self.llm_temp,
            seed=self.llm_seed)

        streaming_resp_buffer = []
        feedback_msg = ReplyAction.Feedback()
        was_cancelled = False

        for chunk in output:
            # Check for cancellation request before processing each chunk
            if goal_handle.is_cancel_requested:
                was_cancelled = True
                self.get_logger().info('Reply action cancelled')
                break

            content = chunk.choices[0].delta.content
            streaming_resp_buffer.append(content)
            # Send feedback
            feedback_msg.streaming_resp = content
            goal_handle.publish_feedback(feedback_msg)

        # TGI inference time
        t1 = time.time()
        dt = t1 - t0

        # Concatenate chunks and prepare result
        resp = ''.join(streaming_resp_buffer)

        # Handle cancellation vs completion
        if was_cancelled:
            resp += self.cancellation_msg
            goal_handle.canceled()
            self.get_logger().info(f'Reply canceled with partial response: '
                                   f'{resp} ({dt:.2f} s)')
        else:
            goal_handle.succeed()
            self.get_logger().info(f'Reply: {resp} ({dt:.2f} s)')

        result_msg = ReplyAction.Result()
        result_msg.reply = resp

        # Publish result (even for cancelled actions)
        result_msg_str = String()
        result_msg_str.data = resp
        self._reply_action_pub.publish(result_msg_str)

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
        $ ros2 run your_package reply_action_server --ros-args -p tgi_server_url:=http://gpu-server:8080
    """  # noqa: E501
    rclpy.init(args=args)

    reply_action_server = ReplyActionServer()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()
    rclpy.spin(reply_action_server, executor=executor)

    reply_action_server.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
