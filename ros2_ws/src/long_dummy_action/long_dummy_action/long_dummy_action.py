import asyncio
import math
import time

import rclpy
from exodapt_robot_interfaces.action import LongDummyAction
from rclpy.action import ActionServer
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
    """

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

        self._action_server = ActionServer(
            self,
            LongDummyAction,
            self.action_server_name,
            execute_callback=self.execute_callback,
        )

        self.get_logger().info(
            'LongDummyActionServer initialized\n'
            'Parameters:\n'
            f'  action_server_name: {self.action_server_name}\n'
            f'  action_duration: {self.action_duration}')

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

    node = LongDummyActionServer()

    asyncio.run(rclpy.spin(node))

    node.destroy_node
    rclpy.shutdown()


if __name__ == '__main__':
    main()
