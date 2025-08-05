import math
import time

import rclpy
from exodapt_robot_interfaces.action import LongDummyAction
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class LongDummyActionServer(Node):
    """
    ROS 2 action server for simulating long-running dummy actions.

    This action server provides a configurable dummy action that runs for a
    specified duration, useful for testing action client implementations,
    timeout handling, and system behavior under long-running operations.
    The server provides periodic feedback about the remaining time until
    completion.

    The node simulates a long-running process by sleeping in 1-second intervals
    while logging the estimated time to completion. This allows testing of
    action cancellation, timeout behavior, and client-server interaction
    patterns in ROS 2 systems.

    Ref: https://github.com/ros2/examples/blob/jazzy/rclpy/actions/minimal_action_server/examples_rclpy_minimal_action_server/server.py

    Parameters:
        action_server_name (str): Name of the action server
            (default: 'long_dummy_action_server')
        action_duration (int): Duration of the dummy action in seconds
            (default: 60)

    Action Interface:
        Action Type: LongDummyAction
        Goal: No goal data required
        Feedback: No feedback messages sent
        Result: Empty result message upon completion

    Attributes:
        action_server_name (str): Name of the action server
        action_duration (int): Duration of the dummy action in seconds
    """  # noqa: E501

    def __init__(self, **kwargs):
        """
        Initialize the LongDummyActionServer node.

        Sets up the ROS 2 action server with configurable parameters for
        dummy action duration and server naming. The server is configured
        to handle LongDummyAction requests that simulate long-running
        operations for testing purposes.

        Args:
            **kwargs: Additional keyword arguments passed to the parent Node
                constructor

        Raises:
            ParameterException: If required parameters are invalid
        """
        super().__init__('long_dummy_action', **kwargs)

        self.declare_parameter('action_server_name',
                               'long_dummy_action_server')
        self.declare_parameter('action_duration', 60)

        self.action_server_name = self.get_parameter(
            'action_server_name').value
        self.action_duration = self.get_parameter('action_duration').value

        self.get_logger().info(
            'LongDummyActionServer initializing\n'
            'Parameters:\n'
            f'  action_server_name: {self.action_server_name}\n'
            f'  action_duration: {self.action_duration}')

        self._action_server = ActionServer(
            self,
            LongDummyAction,
            self.action_server_name,
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        # This server allows multiple goals in parallel
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """
        Execute callback for processing LongDummyAction goals.

        This async callback simulates a long-running operation by sleeping
        for the configured duration while providing periodic logging of the
        estimated time to completion. The method runs in 1-second intervals,
        logging the remaining time until the action completes.

        The callback is useful for testing action client implementations,
        timeout handling, system behavior under long-running operations,
        and action cancellation mechanisms.

        Args:
            goal_handle (ServerGoalHandle): ROS 2 action goal handle containing
                the dummy action request (no goal data required)

        Returns:
            LongDummyAction.Result: Empty action result message indicating
                successful completion

        Note:
            The action runs for the duration specified by the action_duration
            parameter. The method logs completion ETA every second until the
            action finishes.
        """
        self.get_logger().info('Executing LongDummyAction...')

        start_time = time.time()
        end_time = start_time + self.action_duration

        while True:

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Dummy action canceled')
                return LongDummyAction.Result()

            dt = math.ceil(end_time - time.time())

            # Stop action once current time surpasses specified end time
            if dt > 0:
                self.get_logger().info(f'Completion ETA: {dt:d} s')
                time.sleep(1)
            else:
                break

        self.get_logger().info('Dummy action completed')

        goal_handle.succeed()
        result_msg = LongDummyAction.Result()
        return result_msg


def main(args=None):
    """
    Main entry point for the LongDummyActionServer ROS 2 node.

    Initializes the ROS 2 context, creates and spins the LongDummyActionServer
    node using asyncio for handling async action callbacks, and performs
    proper cleanup on shutdown. The function handles the complete node
    lifecycle including initialization, execution, and teardown.

    Args:
        args (List[str], optional): Command line arguments for ROS 2
            initialization. Defaults to None, which uses sys.argv.

    Raises:
        KeyboardInterrupt: Gracefully handles Ctrl+C shutdown
        RuntimeError: If ROS 2 initialization or node creation fails

    Example:
        $ ros2 run your_package long_dummy_action_server
        $ ros2 run your_package long_dummy_action_server --ros-args \\
            -p action_duration:=120
    """
    rclpy.init(args=args)

    long_dummy_action_server = LongDummyActionServer()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()
    rclpy.spin(long_dummy_action_server, executor=executor)

    long_dummy_action_server.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
