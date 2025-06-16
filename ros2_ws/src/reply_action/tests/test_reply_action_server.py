import os
import subprocess
import time

import rclpy
import requests
from exodapt_robot_interfaces.action import ReplyAction
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from reply_action.reply_action import ReplyActionServer
from std_msgs.msg import String

REPLY_ACTION_SERVER_NAME = 'reply_action_server'
REPLY_ACTION_TOPIC = '/reply_action'


class TestReplyActionServer:
    """Unit tests for ReplyActionServer ROS 2 node."""

    # Class-level variables for Docker container management
    docker_container_id = None
    tgi_port = 5000
    tgi_url = f"http://localhost:{tgi_port}"

    @classmethod
    def setup_class(cls):
        """Initialize ROS2 and start TGI inference server for all tests."""
        print("Setting up test class...")

        # Initialize ROS2
        rclpy.init()
        cls.executor = SingleThreadedExecutor()

        # Start real TGI inference server
        cls._start_tgi_server()

    @classmethod
    def teardown_class(cls):
        """Shutdown ROS2 and stop TGI inference server after all tests."""
        print("Tearing down test class...")

        # Stop TGI server
        cls._stop_tgi_server()

        # Shutdown ROS2
        cls.executor.shutdown()
        rclpy.shutdown()

    @classmethod
    def _start_tgi_server(cls):
        """Start the TGI inference server using Docker."""
        print("Starting TGI inference server...")

        # Configuration
        model = "Qwen/Qwen2.5-7B-Instruct"
        volume = f"/home/{os.getenv('USER', 'root')}/.cache/huggingface"
        gpu_id = '"device=0"'
        hf_token = os.getenv('HF_TOKEN', '')

        # Docker command
        docker_cmd = [
            "docker",
            "run",
            "-d",  # -d for detached mode
            "--gpus",
            gpu_id,
            "--shm-size",
            "64g",
            "-p",
            f"{cls.tgi_port}:80",
            "-v",
            f"{volume}:/data",
            "-e",
            f"HF_TOKEN={hf_token}",
            "ghcr.io/huggingface/text-generation-inference:3.3.2",
            "--model-id",
            model,
            "--max-total-tokens",
            "8000"
        ]

        try:
            # Start the container
            result = subprocess.run(docker_cmd,
                                    capture_output=True,
                                    text=True,
                                    check=True)
            cls.docker_container_id = result.stdout.strip()
            print(f"Started TGI container: {cls.docker_container_id}")

            # Wait for server to be ready
            cls._wait_for_tgi_server()

        except subprocess.CalledProcessError as e:
            print(f"Failed to start TGI server: {e}")
            print(f"Docker stdout: {e.stdout}")
            print(f"Docker stderr: {e.stderr}")
            raise

    @classmethod
    def _wait_for_tgi_server(cls, timeout=300, check_interval=5):
        """Wait for TGI server to be ready to accept requests."""
        print("Waiting for TGI server to be ready...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to make a health check request
                response = requests.get(f"{cls.tgi_url}/health", timeout=10)
                if response.status_code == 200:
                    print("TGI server is ready!")
                    return
            except requests.exceptions.RequestException:
                pass

            print(f"TGI server not ready yet, waiting {check_interval}s...")
            time.sleep(check_interval)

        # If we get here, the server didn't start in time
        cls._stop_tgi_server()  # Clean up
        raise TimeoutError(
            f"TGI server failed to start within {timeout} seconds")

    @classmethod
    def _stop_tgi_server(cls):
        """Stop and remove the TGI Docker container."""
        if cls.docker_container_id:
            print(f"Stopping TGI container: {cls.docker_container_id}")
            try:
                # Stop the container
                subprocess.run(["docker", "stop", cls.docker_container_id],
                               capture_output=True,
                               check=True)
                # Remove the container
                subprocess.run(["docker", "rm", cls.docker_container_id],
                               capture_output=True,
                               check=True)
                print("TGI container stopped and removed")
            except subprocess.CalledProcessError as e:
                print(f"Error stopping TGI container: {e}")
            finally:
                cls.docker_container_id = None

    def setup_method(self):
        """Setup before each test method."""
        self.action_decisions_action_server = None

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.action_decisions_action_server:
            self.executor.remove_node(self.action_decisions_action_server)
            self.action_decisions_action_server.destroy_node()

    def create_test_parameters(self):
        """Create Parameter objects for testing."""
        return [
            Parameter('tgi_server_url', Parameter.Type.STRING,
                      'http://localhost:5000'),
            Parameter('max_tokens', Parameter.Type.INTEGER, 128),
            Parameter('llm_temp', Parameter.Type.DOUBLE, 0.6),
            Parameter('llm_seed', Parameter.Type.INTEGER, 14),
        ]

    def test_tgi_server_health(self):
        """Test that TGI server is accessible and healthy."""
        response = requests.get(f"{self.tgi_url}/health")
        assert response.status_code == 200

    def test_tgi_inference_call(self):
        """Test making an actual inference call to TGI server."""
        # Example inference request
        payload = {
            "inputs": "Hello, how are you?",
            "parameters": {
                "max_new_tokens": 10,
                "temperature": 0.6,
                "seed": 14
            }
        }

        response = requests.post(f"{self.tgi_url}/generate", json=payload)
        assert response.status_code == 200

        result = response.json()
        assert "generated_text" in result
        print(f"Generated text: {result['generated_text']}")

    def test_initialization_with_default_parameters(self):
        """Test initialization with default parameters."""
        self.action_decision_action_server = ReplyActionServer()
        self.executor.add_node(self.action_decision_action_server)

        # Check action server is created
        assert isinstance(self.action_decision_action_server,
                          ReplyActionServer)

        assert self.action_decision_action_server.action_server_name == REPLY_ACTION_SERVER_NAME  # noqa: E501

    def test_reply_action_with_real_tgi(self):
        """Test ReplyActionServer with real TGI inference server."""
        test_params = self.create_test_parameters()
        self.reply_action_server = ReplyActionServer(
            parameter_overrides=test_params)
        self.executor.add_node(self.reply_action_server)

        # Create separate client node
        client_node = rclpy.create_node('reply_action_client_test_node')
        self.executor.add_node(client_node)

        reply_action_client = ActionClient(
            client_node,
            ReplyAction,
            REPLY_ACTION_SERVER_NAME,
        )

        # Create subscriber to capture published result
        published_result = []

        def result_callback(msg):
            published_result.append(msg.data)

        result_subscriber = client_node.create_subscription(
            String, REPLY_ACTION_TOPIC, result_callback, 10)

        try:
            # Wait for the action server to be available
            assert reply_action_client.wait_for_server(timeout_sec=10), \
                f"Action server '{REPLY_ACTION_SERVER_NAME}' not available"

            # Create test goal
            goal_msg = ReplyAction.Goal()
            goal_msg.state = "DUMMY STATE"
            goal_msg.instruction = "DUMMY INSTRUCTION"

            # Send the goal
            send_goal_future = reply_action_client.send_goal_async(goal_msg)

            # Spin the executor to process the goal request
            # Use the class executor to ensure both nodes are processed
            start_time = time.time()
            while not send_goal_future.done() and (time.time() -
                                                   start_time) < 5.0:
                self.executor.spin_once(timeout_sec=0.1)

            # Check if future completed
            assert send_goal_future.done(
            ), "send_goal_future did not complete within timeout"

            goal_handle = send_goal_future.result()
            assert goal_handle is not None, "Goal handle is None - future may have failed"  # noqa: E501
            assert goal_handle.accepted, "Goal was not accepted by action server"  # noqa: E501

            print("Goal accepted, waiting for result...")

            # Wait for result
            get_result_future = goal_handle.get_result_async()

            # Spin executor until result is ready
            start_time = time.time()
            while not get_result_future.done() and (time.time() -
                                                    start_time) < 30.0:
                self.executor.spin_once(timeout_sec=0.1)

            assert get_result_future.done(
            ), "get_result_future did not complete within timeout"

            # Verify result
            result_response = get_result_future.result()
            assert result_response is not None, "No result response received"

            result = result_response.result
            assert result is not None, "No result received"
            assert hasattr(result, 'reply'), "Result missing reply field"
            reply = result.reply
            assert isinstance(reply, str), "reply should be a string"
            assert len(reply) > 0, "reply should not be empty"

            print(f'Received reply: "{reply}"')

            # Allow some time for the publication to reach the subscriber
            time.sleep(0.5)
            self.executor.spin_once(timeout_sec=0.1)

            # Assert that the result was published to the topic
            assert len(published_result
                       ) > 0, "No result published to /reply_action topic"
            assert published_result[
                0] == reply, f"Published result '{published_result[0]}' does not match action result '{reply}'"  # noqa: E501

            print(f'Confirmed publication to topic: "{published_result[0]}"')

        except Exception as e:
            print(f"Test failed with error: {e}")
            print(
                f"Future done: {send_goal_future.done() if 'send_goal_future' in locals() else 'N/A'}"  # noqa: E501
            )
            if 'send_goal_future' in locals() and send_goal_future.done():
                print(f"Future result: {send_goal_future.result()}")
            raise

        finally:
            # Clean up client node
            self.executor.remove_node(client_node)
            client_node.destroy_node()
