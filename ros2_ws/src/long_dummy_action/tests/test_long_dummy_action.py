import time

import rclpy
from exodapt_robot_interfaces.action import LongDummyAction
from long_dummy_action.long_dummy_action import LongDummyActionServer
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter

LONG_DUMMY_ACTION_SERVER_NAME = 'long_dummy_action_server'


class TestLongDummyActionServer:
    """Unit tests for LongDummyActionServer ROS 2 node."""

    @classmethod
    def setup_class(cls):
        """Initialize ROS2 for all tests."""
        print("Setting up test class...")

        # Initialize ROS2
        rclpy.init()
        cls.executor = SingleThreadedExecutor()

    @classmethod
    def teardown_class(cls):
        """Shutdown ROS2 after all tests."""
        print("Tearing down test class...")

        # Shutdown ROS2
        cls.executor.shutdown()
        rclpy.shutdown()

    def setup_method(self):
        """Setup before each test method."""
        self.long_dummy_action_server = None

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.long_dummy_action_server:
            self.executor.remove_node(self.long_dummy_action_server)
            self.long_dummy_action_server.destroy_node()

    def create_test_parameters(self):
        """Create Parameter objects for testing."""
        return [
            Parameter('action_server_name', Parameter.Type.STRING,
                      'test_long_dummy_action_server'),
            Parameter('action_duration', Parameter.Type.INTEGER, 3),
        ]

    def test_initialization_with_default_parameters(self):
        """Test initialization with default parameters."""
        self.long_dummy_action_server = LongDummyActionServer()
        self.executor.add_node(self.long_dummy_action_server)

        # Check action server is created
        assert isinstance(self.long_dummy_action_server, LongDummyActionServer)
        assert (self.long_dummy_action_server.action_server_name ==
                LONG_DUMMY_ACTION_SERVER_NAME)
        assert self.long_dummy_action_server.action_duration == 60

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        test_params = self.create_test_parameters()
        self.long_dummy_action_server = LongDummyActionServer(
            parameter_overrides=test_params)
        self.executor.add_node(self.long_dummy_action_server)

        # Check custom parameters are set
        expected_name = 'test_long_dummy_action_server'
        assert (
            self.long_dummy_action_server.action_server_name == expected_name)
        assert self.long_dummy_action_server.action_duration == 3

    def test_long_dummy_action_execution(self):
        """Test LongDummyActionServer action execution with short duration."""
        # Use short duration for testing
        test_params = [
            Parameter('action_server_name', Parameter.Type.STRING,
                      'test_long_dummy_action_server'),
            Parameter('action_duration', Parameter.Type.INTEGER, 2),
        ]

        self.long_dummy_action_server = LongDummyActionServer(
            parameter_overrides=test_params)
        self.executor.add_node(self.long_dummy_action_server)

        # Create separate client node
        client_node = rclpy.create_node('long_dummy_action_client_test_node')
        self.executor.add_node(client_node)

        long_dummy_action_client = ActionClient(
            client_node,
            LongDummyAction,
            'test_long_dummy_action_server',
        )

        try:
            # Wait for the action server to be available
            server_available = long_dummy_action_client.wait_for_server(
                timeout_sec=10)
            assert server_available, \
                "Action server 'test_long_dummy_action_server' not available"

            # Create test goal (empty - no attributes)
            goal_msg = LongDummyAction.Goal()

            # Record start time to verify duration
            start_time = time.time()

            # Send the goal
            send_goal_future = long_dummy_action_client.send_goal_async(
                goal_msg)

            # Spin the executor to process the goal request
            execution_start_time = time.time()
            timeout_limit = 5.0
            while (not send_goal_future.done()
                   and (time.time() - execution_start_time) < timeout_limit):
                self.executor.spin_once(timeout_sec=0.1)

            # Check if future completed
            assert send_goal_future.done(), \
                "send_goal_future did not complete within timeout"

            goal_handle = send_goal_future.result()
            assert goal_handle is not None, \
                "Goal handle is None - future may have failed"
            assert goal_handle.accepted, \
                "Goal was not accepted by action server"

            print("Goal accepted, waiting for result...")

            # Wait for result
            get_result_future = goal_handle.get_result_async()

            # Spin executor until result is ready
            result_start_time = time.time()
            result_timeout = 10.0
            while (not get_result_future.done()
                   and (time.time() - result_start_time) < result_timeout):
                self.executor.spin_once(timeout_sec=0.1)

            assert get_result_future.done(), \
                "get_result_future did not complete within timeout"

            # Record end time
            end_time = time.time()
            execution_duration = end_time - start_time

            # Verify result
            result_response = get_result_future.result()
            assert result_response is not None, "No result response received"

            result = result_response.result
            assert result is not None, "No result received"

            # Since the action interface has no attributes, we just verify
            # that we got a valid result object back
            print(f'Received result object: {result}')

            # Verify action took approximately the expected duration
            expected_duration = 2.0  # 2 seconds as set in parameters
            tolerance = 1.0  # 1 second tolerance
            duration_diff = abs(execution_duration - expected_duration)
            assert duration_diff <= tolerance, \
                (f"Action duration {execution_duration:.1f}s not within "
                 f"tolerance of expected {expected_duration}s")

            expected_text = f"expected ~{expected_duration}s"
            print(f'Action completed in {execution_duration:.1f} seconds '
                  f'({expected_text}) with empty result')

        except Exception as e:
            print(f"Test failed with error: {e}")
            future_status = ("N/A" if 'send_goal_future' not in locals() else
                             send_goal_future.done())
            print(f"Future done: {future_status}")
            if 'send_goal_future' in locals() and send_goal_future.done():
                print(f"Future result: {send_goal_future.result()}")
            raise

        finally:
            # Clean up client node
            self.executor.remove_node(client_node)
            client_node.destroy_node()

    def test_action_duration_parameter_effect(self):
        """Test that action_duration parameter affects execution time."""
        # Create server with very short duration
        test_params = [
            Parameter('action_duration', Parameter.Type.INTEGER,
                      1),  # 1 second
        ]

        self.long_dummy_action_server = LongDummyActionServer(
            parameter_overrides=test_params)
        self.executor.add_node(self.long_dummy_action_server)

        # Verify parameter was set correctly
        assert self.long_dummy_action_server.action_duration == 1

        # Create client and test execution time
        client_node = rclpy.create_node('duration_test_client_node')
        self.executor.add_node(client_node)

        long_dummy_action_client = ActionClient(
            client_node,
            LongDummyAction,
            LONG_DUMMY_ACTION_SERVER_NAME,
        )

        try:
            assert long_dummy_action_client.wait_for_server(timeout_sec=5), \
                "Action server not available"

            goal_msg = LongDummyAction.Goal()

            start_time = time.time()

            # Send goal and wait for completion
            send_goal_future = long_dummy_action_client.send_goal_async(
                goal_msg)

            while not send_goal_future.done():
                self.executor.spin_once(timeout_sec=0.1)

            goal_handle = send_goal_future.result()
            get_result_future = goal_handle.get_result_async()

            while not get_result_future.done():
                self.executor.spin_once(timeout_sec=0.1)

            end_time = time.time()
            execution_duration = end_time - start_time

            # Should complete in approximately 1 second (with tolerance)
            min_duration = 0.8
            max_duration = 2.0
            assert execution_duration >= min_duration, \
                f"Action completed too quickly: {execution_duration:.1f}s"
            assert execution_duration <= max_duration, \
                f"Action took too long: {execution_duration:.1f}s"

            print(f'Short duration test completed in '
                  f'{execution_duration:.1f} seconds')

        finally:
            self.executor.remove_node(client_node)
            client_node.destroy_node()
