import asyncio
import time

import rclpy
from exodapt_robot_interfaces.action import ReplyAction
from exodapt_robot_pt import reply_action_pt
from huggingface_hub import InferenceClient
from rclpy.action import ActionServer
from rclpy.node import Node


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
        self.declare_parameter('tgi_server_url', 'http://localhost:5000')
        self.declare_parameter('max_tokens', 1024)
        self.declare_parameter('llm_temp', 0.6)
        self.declare_parameter('llm_seed', 14)
        self.action_server_name = self.get_parameter(
            'action_server_name').value
        self.tgi_server_url = self.get_parameter('tgi_server_url').value
        self.max_tokens = self.get_parameter('max_tokens').value
        self.llm_temp = self.get_parameter('llm_temp').value
        self.llm_seed = self.get_parameter('llm_seed').value

        self._action_server = ActionServer(
            self,
            ReplyAction,
            self.action_server_name,
            execute_callback=self.execute_callback_tgi,
        )
        # TGI inference client
        base_url = f"{self.tgi_server_url}/v1/"
        self.client = InferenceClient(base_url=base_url)

        self.get_logger().info(
            'ReplyActionServer initialized\n'
            'Parameters:\n'
            f'  action_server_name: {self.action_server_name}\n'
            f'  TGI server url: {self.tgi_server_url}\n'
            f'  max_tokens={self.max_tokens}\n'
            f'  llm_temp={self.llm_temp}\n'
            f'  llm_seed={self.llm_seed}')

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
        - Streaming inference with real-time feedback
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
            size and server configuration.
        """
        self.get_logger().info('Executing ReplyActionServer...')

        # Unpack ReplyAction.Goal() msg
        goal = goal_handle.request
        state = goal.state

        if goal.instruction is not None:
            self.get_logger().warn('Providing instruction is not implemented')

        user_msg = reply_action_pt(state)

        t0 = time.time()

        output = self.client.chat.completions.create(
            model="tgi",
            messages=[
                {
                    "role": "user",
                    "content": user_msg
                },
            ],
            stream=True,
            max_tokens=self.max_tokens,
            temperature=self.llm_temp,
            seed=self.llm_seed)

        streaming_resp_buffer = []
        feedback_msg = ReplyAction.Feedback()

        for chunk in output:
            content = chunk.choices[0].delta.content
            streaming_resp_buffer.append(content)
            # Send feedback
            feedback_msg.streaming_resp = content
            goal_handle.publish_feedback(feedback_msg)

        # TGI inference time
        t1 = time.time()
        dt = t1 - t0

        # Concatenate chunks and send result
        resp = ''.join(streaming_resp_buffer)

        goal_handle.succeed()
        self.get_logger().info(
            f'ReplyActionServer response: {resp} ({dt:.2f} s)')

        result_msg = ReplyAction.Result()
        result_msg.reply = resp
        return result_msg


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

    node = ReplyActionServer()

    asyncio.run(rclpy.spin(node))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
