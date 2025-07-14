import time

import rclpy
from action_msgs.msg import GoalStatus
from exodapt_robot_interfaces.action import ReplyAction
from rclpy.action import (ActionClient, ActionServer, CancelResponse,
                          GoalResponse)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class ReplyTTSActionServer(Node):
    """
    Wrapper for ReplyActionServer that generates TTS on intermediate feedback
    messages.
    """

    def __init__(self, **kwargs):
        """"""
        super().__init__('reply_tts_action', **kwargs)

        self.declare_parameter('action_server_name', 'reply_tts_action_server')
        self.declare_parameter(
            'reply_action_server_name',
            'reply_action_server',
        )

        self.action_server_name = self.get_parameter(
            'action_server_name').value
        self.reply_action_server_name = self.get_parameter(
            'reply_action_server_name').value

        self._action_server = ActionServer(
            self,
            ReplyAction,
            self.action_server_name,
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self._action_client = ActionClient(
            self,
            ReplyAction,
            self.reply_action_server_name,
            callback_group=ReentrantCallbackGroup(),
        )

        # Maps goal_handle.goal_id.uuid to reply_action_goal_handles
        self.active_goals = {}

        self.get_logger().info(
            'ReplyTTSActionServer initialized\n'
            'Parameters:\n'
            f'  action_server_name: {self.action_server_name}\n'
            f'  reply_action_server_name: {self.reply_action_server_name}\n')

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

    async def execute_callback(self, goal_handle):
        """Execute a goal by forwarding it to the ReplyActionServer.

        Steps:
            1. Wait for action server
            2. Create and send goal to ReplyActionServer
            3. Wait for goal accept/reject
            4. Check for cancellation requests while waiting for goal completion
            5. Process result
        """
        self.get_logger().info('Executing ReplyTTSActionServer...')

        # Wait for action server
        if not self._action_client.wait_for_server(timeout_sec=5):
            self.get_logger().error('ReplyActionServer not available')
            goal_handle.abort()
            return ReplyAction.Result()

        # Send goal to ReplyActionServer
        reply_action_goal = ReplyAction.Goal()
        # reply_action_goal.state = goal_handle.request.state
        reply_action_goal.state = 'Write a long and exhaustive essay about the history of manking spanning several volumes.'
        reply_action_goal.instruction = goal_handle.request.instruction

        try:
            send_goal_future = self._action_client.send_goal_async(
                reply_action_goal,
                feedback_callback=lambda feedback: self._feedback_callback(
                    goal_handle, feedback))

            # Wait for goal accept/reject
            reply_action_goal_handle = await send_goal_future

            if not reply_action_goal_handle.accepted:
                self.get_logger().error('ReplyActionServer rejected goal')
                goal_handle.abort()
                return ReplyAction.Result()
            self.get_logger().info('ReplyActionServer accepted goal')

            # Store accepted goal for cancellation handling
            goal_uuid = tuple(goal_handle.goal_id.uuid)
            self.active_goals[goal_uuid] = reply_action_goal_handle

            # Check for cancellation request while waiting for goal completion
            result_future = reply_action_goal_handle.get_result_async()

            while not result_future.done():
                if goal_handle.is_cancel_requested:
                    self.get_logger().info(
                        'Canceling ReplyTTSActionServer goal')
                    cancel_future = reply_action_goal_handle.cancel_goal_async(
                    )
                    await cancel_future

                    # Clean up
                    if goal_uuid in self.active_goals:
                        del self.active_goals[goal_uuid]

                    goal_handle.canceled()
                    return ReplyAction.Result()

                time.sleep(0.1)
                # await asyncio.sleep(0.1)

            # Get the ReplyActionServer result
            reply_action_result = await result_future

            # Clean up
            if goal_uuid in self.active_goals:
                del self.active_goals[goal_uuid]

            # Case 1: Success
            if reply_action_result.status == GoalStatus.STATUS_SUCCEEDED:
                reply = reply_action_result.result.reply
                self.get_logger().info(
                    f'ReplyActionServer succeeded with result: {reply}')

                result = ReplyAction.Result()
                result.reply = reply

                goal_handle.succeed()
                return result

            # Case 2: Cancelled
            elif reply_action_result.status == GoalStatus.STATUS_CANCELED:
                self.get_logger().info('ReplyActionServer was canceled')
                goal_handle.canceled()
                return ReplyAction.Result()

            # Case 3: General abort
            else:
                status = reply_action_result.status
                self.get_logger().error(
                    f'ReplyActionServer failed with status: {status}')
                goal_handle.abort()
                return ReplyAction.Result()

        # Case 4: General error
        except Exception as e:
            self.get_logger().error(f'Error during goal execution: {str(e)}')

            # Clean up
            goal_uuid = tuple(goal_handle.goal_id.uuid)
            if goal_uuid in self.active_goals:
                del self.active_goals[goal_uuid]

            goal_handle.abort()
            return ReplyAction.Result()

    def _feedback_callback(self, goal_handle, reply_action_feedback):
        """Perform TTS on intermediate feedback from the ReplyActionServer."""
        chunk = reply_action_feedback.feedback.streaming_resp
        self.get_logger().info(f'Feedback: {chunk}')

        feedback = ReplyAction.Feedback()
        feedback.streaming_resp = chunk

        goal_handle.publish_feedback(feedback)


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

    reply_tts_action_server = ReplyTTSActionServer()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()
    rclpy.spin(reply_tts_action_server, executor=executor)

    reply_tts_action_server.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
