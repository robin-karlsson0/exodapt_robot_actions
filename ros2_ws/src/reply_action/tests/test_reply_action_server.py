"""
Test suite for ReplyActionServer ROS 2 node with multiple LLM inference
backends.

This module provides comprehensive integration tests for the ReplyActionServer,
which integrates ROS 2 action server functionality with Large Language Model
(LLM) inference capabilities. The tests validate both the ROS 2 action interface
and the underlying LLM inference implementations.

Test Structure:
    - BaseReplyActionServerTest: Common test infrastructure and shared test
      logic
    - TestReplyActionServerVLLM: Integration tests using vLLM inference backend
    - TestReplyActionServerTGI: Integration tests using TGI inference backend
      (commented out)

Test Coverage:
    1. Docker container management for LLM inference servers
    2. Server health checks and readiness validation
    3. Direct inference API calls to validate server functionality
    4. ROS 2 action server initialization and parameter handling
    5. End-to-end action execution with real LLM inference
    6. Topic publication verification for action results

Each test class manages its own Docker container for the respective inference
server, ensuring isolated and reproducible test environments. Tests validate
both the technical integration (API calls, ROS 2 messaging) and functional
behavior (text generation).

Requirements:
    - Docker with GPU support
    - HF_TOKEN environment variable for Hugging Face model access
    - Sufficient GPU memory for model loading (configured for 1.5B parameter
      models)
"""

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

# Test Configuration for ReplyActionServer Tests

# Model Configuration
DEFAULT_MODEL = "Qwen/Qwen2-1.5B-Instruct"
TEST_MAX_TOKENS = 128
TEST_TEMPERATURE = 0.6
TEST_SEED = 14

# Server Configuration
TGI_PORT = 8000
VLLM_PORT = 8001  # Use different port to avoid conflicts
TGI_IMAGE = "ghcr.io/huggingface/text-generation-inference:3.3.2"
VLLM_IMAGE = "vllm/vllm-openai:latest"

# Timeouts (in seconds)
SERVER_STARTUP_TIMEOUT = 300  # model loading
ACTION_EXECUTION_TIMEOUT = 30  # inference
GOAL_PROCESSING_TIMEOUT = 5  # goal acceptance

# Test Parameters
TEST_STATE = "DUMMY STATE"
TEST_INSTRUCTION = "DUMMY INSTRUCTION"

# Docker Configuration
DOCKER_SHARED_MEMORY = "64g"
DOCKER_GPU_CONFIG = '"device=0"'
MODEL_MAX_LENGTH = 1024

REPLY_ACTION_SERVER_NAME = 'reply_action_server'
REPLY_ACTION_TOPIC = '/reply_action'


class BaseReplyActionServerTest:
    """
    Base class for ReplyActionServer tests with common functionality.

    This class provides shared infrastructure for testing ReplyActionServer
    with different LLM inference backends. It includes common setup/teardown
    methods and the core test logic that can be reused across different
    inference server implementations.

    Key Features:
        - Common test setup and cleanup
        - Shared action client creation and management
        - Unified test execution flow for different backends
        - Result validation and topic publication verification

    The base class implements the core test pattern:
        1. Initialize ReplyActionServer with backend-specific parameters
        2. Create action client and result subscriber
        3. Send test goal and verify acceptance
        4. Wait for action completion and validate result
        5. Verify result publication to ROS topic
    """

    def setup_method(self):
        """Initialize test state before each test method execution."""
        self.reply_action_server = None

    def teardown_method(self):
        """Clean up resources after each test method execution."""
        if self.reply_action_server:
            self.executor.remove_node(self.reply_action_server)
            self.reply_action_server.destroy_node()

    def _test_reply_action_with_server(self, inference_server_type,
                                       server_url):
        """
        Execute comprehensive end-to-end test of ReplyActionServer.

        This method implements the complete test workflow for validating
        ReplyActionServer functionality with a real LLM inference backend.

        Args:
            inference_server_type (str): Backend type ('tgi' or 'vllm')
            server_url (str): URL of the running inference server

        Test Flow:
            1. Configure ReplyActionServer with backend-specific parameters
            2. Create action client and result topic subscriber
            3. Send test goal with dummy state and instruction
            4. Verify goal acceptance and action execution
            5. Validate generated response content and format
            6. Confirm result publication to designated ROS topic

        Assertions:
            - Action server availability and goal acceptance
            - Non-empty string response from LLM inference
            - Correct result publication to /reply_action topic
        """
        test_params = [
            Parameter('inference_server_type', Parameter.Type.STRING,
                      inference_server_type),
            Parameter('inference_server_url', Parameter.Type.STRING,
                      server_url),
            Parameter('max_tokens', Parameter.Type.INTEGER, TEST_MAX_TOKENS),
            Parameter('llm_temp', Parameter.Type.DOUBLE, TEST_TEMPERATURE),
            Parameter('llm_seed', Parameter.Type.INTEGER, TEST_SEED),
        ]

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
            goal_msg.state = TEST_STATE
            goal_msg.instruction = TEST_INSTRUCTION

            # Send the goal
            send_goal_future = reply_action_client.send_goal_async(goal_msg)

            # Spin the executor to process the goal request
            start_time = time.time()
            while (not send_goal_future.done()
                   and (time.time() - start_time) < GOAL_PROCESSING_TIMEOUT):
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
            start_time = time.time()
            while (not get_result_future.done()
                   and (time.time() - start_time) < ACTION_EXECUTION_TIMEOUT):
                self.executor.spin_once(timeout_sec=0.1)

            assert get_result_future.done(), \
                "get_result_future did not complete within timeout"

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
            assert len(published_result) > 0, \
                "No result published to /reply_action topic"
            assert published_result[0] == reply, \
                f"Published result '{published_result[0]}' does not match " \
                f"action result '{reply}'"

            print(f'Confirmed publication to topic: "{published_result[0]}"')

        except Exception as e:
            print(f"Test failed with error: {e}")
            future_done = (send_goal_future.done()
                           if 'send_goal_future' in locals() else 'N/A')
            print(f"Future done: {future_done}")
            if 'send_goal_future' in locals() and send_goal_future.done():
                print(f"Future result: {send_goal_future.result()}")
            raise

        finally:
            # Clean up client node
            self.executor.remove_node(client_node)
            client_node.destroy_node()


class TestReplyActionServerTGI(BaseReplyActionServerTest):
    """
    Integration tests for ReplyActionServer with TGI inference backend.

    This test class validates the complete integration between ReplyActionServer
    and TGI (Text Generation Inference) serving infrastructure. It manages
    a dedicated Docker container running TGI with HTTP API and executes
    comprehensive tests covering both direct API access and ROS 2 action
    interface functionality.

    Test Infrastructure:
        - Automated Docker container lifecycle management
        - TGI server with HTTP inference API endpoints
        - Health monitoring and readiness validation
        - Proper resource cleanup and error handling

    Test Categories:
        1. Infrastructure Tests:
           - Server health and availability validation
           - Direct API inference calls using TGI HTTP format

        2. Integration Tests:
           - ReplyActionServer initialization with TGI backend
           - End-to-end action execution with real text generation
           - ROS topic publication verification

    Container Configuration:
        - Model: Qwen/Qwen2-1.5B-Instruct (configured for test efficiency)
        - API: TGI HTTP /generate endpoint
        - Port: 8000 (default TGI port mapped to host)
        - GPU: Requires CUDA-capable device for model inference

    Note: Tests require HF_TOKEN environment variable for model access
    and sufficient GPU memory for 1.5B parameter model loading.
    """

    # Class-level variables for Docker container management
    docker_container_id = None
    tgi_port = TGI_PORT
    tgi_url = f"http://localhost:{tgi_port}"

    @classmethod
    def setup_class(cls):
        """
        Initialize test class with ROS 2 context and TGI inference server.

        Sets up the complete test infrastructure including:
        - ROS 2 initialization and executor creation
        - Docker container startup with TGI server
        - Model loading and server readiness validation

        This method blocks until the TGI server is fully ready to accept
        inference requests, ensuring reliable test execution.
        """
        print("Setting up TGI test class...")

        # Initialize ROS2
        rclpy.init()
        cls.executor = SingleThreadedExecutor()

        # Start real TGI inference server
        cls._start_tgi_server()

    @classmethod
    def teardown_class(cls):
        """
        Clean up test class resources and stop TGI inference server.

        Performs complete cleanup including:
        - Docker container termination and removal
        - ROS 2 executor shutdown and context cleanup

        Ensures no resources are leaked between test runs.
        """
        print("Tearing down TGI test class...")

        # Stop TGI server
        cls._stop_tgi_server()

        # Shutdown ROS2
        cls.executor.shutdown()
        rclpy.shutdown()

    @classmethod
    def _start_tgi_server(cls):
        """
        Launch TGI inference server in Docker container.

        Configures and starts a Docker container running TGI with:
        - GPU acceleration for model inference
        - HTTP API on port 80 (mapped to host port 8000)
        - Hugging Face model cache mounting for efficient model loading
        - Shared memory configuration for optimal performance

        Raises:
            subprocess.CalledProcessError: If Docker container fails to start
            TimeoutError: If server doesn't become ready within timeout period
        """
        print("Starting TGI inference server...")

        # Configuration
        model = DEFAULT_MODEL
        volume = f"/home/{os.getenv('USER', 'root')}/.cache/huggingface"
        gpu_id = DOCKER_GPU_CONFIG
        hf_token = os.getenv('HF_TOKEN', '')

        # Docker command
        docker_cmd = [
            "docker",
            "run",
            "-d",  # -d for detached mode
            # "--rm",
            "--gpus",
            gpu_id,
            "--shm-size",
            DOCKER_SHARED_MEMORY,
            "-p",
            f"{cls.tgi_port}:80",
            "-v",
            f"{volume}:/data",
            "-e",
            f"HF_TOKEN={hf_token}",
            TGI_IMAGE,
            "--model-id",
            model,
            "--max-input-length",
            str(MODEL_MAX_LENGTH),
            "--max-total-tokens",
            str(MODEL_MAX_LENGTH + 1),
            "--max-batch-prefill-tokens",
            str(MODEL_MAX_LENGTH),
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
    def _wait_for_tgi_server(cls,
                             timeout=SERVER_STARTUP_TIMEOUT,
                             check_interval=5):
        """
        Wait for TGI server to complete initialization and become ready.

        Performs health checks against the TGI /health endpoint and monitors
        container status to ensure the server is operational. This method
        implements a robust waiting strategy with proper error handling.

        Args:
            timeout (int): Maximum wait time in seconds (default: 300)
            check_interval (int): Time between health checks in seconds
                (default: 5)

        Raises:
            TimeoutError: If server doesn't become ready within timeout
            RuntimeError: If Docker container stops unexpectedly
        """
        print("Waiting for TGI server to be ready...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to make a health check request
                response = requests.get(f"{cls.tgi_url}/health", timeout=10)
                if response.status_code == 200:
                    print("TGI server is ready!")
                    return
                else:
                    print(f"TGI server responded with status "
                          f"{response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"TGI health check failed: {e}")

            # Check if container is still running
            try:
                result = subprocess.run([
                    "docker", "inspect", "--format={{.State.Running}}",
                    cls.docker_container_id
                ],
                                        capture_output=True,
                                        text=True,
                                        check=True)
                is_running = result.stdout.strip() == "true"
                if not is_running:
                    # Container stopped, get logs for debugging
                    logs_result = subprocess.run(
                        ["docker", "logs", cls.docker_container_id],
                        capture_output=True,
                        text=True)
                    print(f"TGI container stopped unexpectedly. Logs:\n"
                          f"{logs_result.stdout}\n{logs_result.stderr}")
                    raise RuntimeError("TGI container stopped unexpectedly")
            except subprocess.CalledProcessError as e:
                print(f"Failed to check container status: {e}")

            print(f"TGI server not ready yet, waiting {check_interval}s...")
            time.sleep(check_interval)

        # If we get here, the server didn't start in time
        cls._stop_tgi_server()  # Clean up
        raise TimeoutError(
            f"TGI server failed to start within {timeout} seconds")

    @classmethod
    def _stop_tgi_server(cls):
        """
        Stop and remove the TGI Docker container.

        Performs graceful shutdown of the Docker container and cleanup:
        - Stops the running container
        - Removes the container to free resources
        - Resets container ID to None

        Handles errors gracefully to ensure cleanup attempts don't fail
        the overall test teardown process.
        """
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

    def test_tgi_server_health(self):
        """
        Validate TGI server accessibility and health status.

        Tests the basic connectivity to the TGI server by making a health
        check request to the /health endpoint. This ensures the server
        is running and responding before conducting more complex tests.
        """
        response = requests.get(f"{self.tgi_url}/health")
        assert response.status_code == 200

    def test_tgi_inference_call(self):
        """
        Test direct inference call to TGI server using HTTP API.

        Validates that the TGI server can successfully process inference
        requests by sending a generation request and verifying the response
        format. This test confirms the server's text generation capabilities
        independently of ROS 2 integration.

        Tests:
            - TGI HTTP /generate endpoint
            - Proper request/response format handling
            - Text generation with controlled parameters
        """
        # Example inference request
        payload = {
            "inputs": "Hello, how are you?",
            "parameters": {
                "max_new_tokens": 10,
                "temperature": TEST_TEMPERATURE,
                "seed": TEST_SEED
            }
        }

        response = requests.post(f"{self.tgi_url}/generate", json=payload)
        assert response.status_code == 200

        result = response.json()
        assert "generated_text" in result
        print(f"Generated text: {result['generated_text']}")

    def test_initialization_with_default_parameters(self):
        """
        Test ReplyActionServer initialization with default configuration.

        Validates that the ReplyActionServer can be successfully instantiated
        with default parameters and properly integrates with the ROS 2
        executor system. This test ensures basic node functionality before
        testing inference capabilities.

        Verifies:
            - Successful node creation and type validation
            - Correct action server name assignment
            - Proper integration with ROS 2 executor
        """
        self.reply_action_server = ReplyActionServer()
        self.executor.add_node(self.reply_action_server)

        # Check action server is created
        assert isinstance(self.reply_action_server, ReplyActionServer)
        assert (self.reply_action_server.action_server_name ==
                REPLY_ACTION_SERVER_NAME)

    def test_reply_action_with_real_tgi(self):
        """
        Test complete ReplyActionServer integration with TGI backend.

        Executes the comprehensive end-to-end test using the shared test
        logic from BaseReplyActionServerTest. This validates the complete
        workflow from ROS 2 action goal submission through TGI inference
        to result publication.

        Tests the full integration chain:
            - ROS 2 action server configuration with TGI parameters
            - Action goal processing and acceptance
            - TGI inference execution with real text generation
            - Result validation and topic publication verification
        """
        self._test_reply_action_with_server('tgi', self.tgi_url)


class TestReplyActionServerVLLM(BaseReplyActionServerTest):
    """
    Integration tests for ReplyActionServer with vLLM inference backend.

    This test class validates the complete integration between ReplyActionServer
    and vLLM (Virtual Large Language Model) serving infrastructure. It manages
    a dedicated Docker container running vLLM with OpenAI-compatible API and
    executes comprehensive tests covering both direct API access and ROS 2
    action interface functionality.

    Test Infrastructure:
        - Automated Docker container lifecycle management
        - vLLM server with OpenAI-compatible API endpoints
        - Health monitoring and readiness validation
        - Proper resource cleanup and error handling

    Test Categories:
        1. Infrastructure Tests:
           - Server health and availability validation
           - Direct API inference calls using OpenAI format

        2. Integration Tests:
           - ReplyActionServer initialization with vLLM backend
           - End-to-end action execution with real text generation
           - ROS topic publication verification

    Container Configuration:
        - Model: Qwen/Qwen2-1.5B-Instruct (configured for test efficiency)
        - API: OpenAI-compatible chat completions endpoint
        - Port: 8001 (isolated from TGI to avoid conflicts)
        - GPU: Requires CUDA-capable device for model inference

    Note: Tests require HF_TOKEN environment variable for model access
    and sufficient GPU memory for 1.5B parameter model loading.
    """

    # Class-level variables for Docker container management
    docker_container_id = None
    vllm_port = VLLM_PORT
    vllm_url = f"http://localhost:{vllm_port}"

    @classmethod
    def setup_class(cls):
        """
        Initialize test class with ROS 2 context and vLLM inference server.

        Sets up the complete test infrastructure including:
        - ROS 2 initialization and executor creation
        - Docker container startup with vLLM server
        - Model loading and server readiness validation

        This method blocks until the vLLM server is fully ready to accept
        inference requests, ensuring reliable test execution.
        """
        print("Setting up vLLM test class...")

        # Initialize ROS2
        rclpy.init()
        cls.executor = SingleThreadedExecutor()

        # Start real vLLM inference server
        cls._start_vllm_server()

    @classmethod
    def teardown_class(cls):
        """
        Clean up test class resources and stop vLLM inference server.

        Performs complete cleanup including:
        - Docker container termination and removal
        - ROS 2 executor shutdown and context cleanup

        Ensures no resources are leaked between test runs.
        """
        print("Tearing down vLLM test class...")

        # Stop vLLM server
        cls._stop_vllm_server()

        # Shutdown ROS2
        cls.executor.shutdown()
        rclpy.shutdown()

    @classmethod
    def _start_vllm_server(cls):
        """
        Launch vLLM inference server in Docker container.

        Configures and starts a Docker container running vLLM with:
        - GPU acceleration for model inference
        - OpenAI-compatible API on port 8000 (mapped to host port 8001)
        - Hugging Face model cache mounting for efficient model loading
        - Proper memory and IPC configuration for optimal performance

        Raises:
            subprocess.CalledProcessError: If Docker container fails to start
            TimeoutError: If server doesn't become ready within timeout period
        """
        print("Starting vLLM inference server...")

        # Configuration
        model = DEFAULT_MODEL
        volume = f"/home/{os.getenv('USER', 'root')}/.cache/huggingface"
        gpu_id = DOCKER_GPU_CONFIG
        hf_token = os.getenv('HF_TOKEN', '')

        # Docker command for vLLM
        docker_cmd = [
            "docker",
            "run",
            "-d",  # -d for detached mode
            "--gpus",
            gpu_id,
            "-p",
            f"{cls.vllm_port}:8000",
            "-v",
            f"{volume}:/root/.cache/huggingface",
            "--env",
            f"HUGGING_FACE_HUB_TOKEN={hf_token}",
            "--ipc=host",
            VLLM_IMAGE,
            "--model",
            model,
            "--max-model-len",
            str(MODEL_MAX_LENGTH),
            "--gpu_memory_utilization",
            str(0.85),
        ]

        try:
            # Start the container
            result = subprocess.run(docker_cmd,
                                    capture_output=True,
                                    text=True,
                                    check=True)
            cls.docker_container_id = result.stdout.strip()
            print(f"Started vLLM container: {cls.docker_container_id}")

            # Wait for server to be ready
            cls._wait_for_vllm_server()

        except subprocess.CalledProcessError as e:
            print(f"Failed to start vLLM server: {e}")
            print(f"Docker stdout: {e.stdout}")
            print(f"Docker stderr: {e.stderr}")
            raise

    @classmethod
    def _wait_for_vllm_server(cls,
                              timeout=SERVER_STARTUP_TIMEOUT,
                              check_interval=5):
        """
        Wait for vLLM server to complete initialization and become ready.

        Performs health checks against the vLLM /health endpoint and monitors
        container status to ensure the server is operational. This method
        implements a robust waiting strategy with proper error handling.

        Args:
            timeout (int): Maximum wait time in seconds (default: 300)
            check_interval (int): Time between health checks in seconds
                (default: 5)

        Raises:
            TimeoutError: If server doesn't become ready within timeout
            RuntimeError: If Docker container stops unexpectedly
        """
        print("Waiting for vLLM server to be ready...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to make a health check request to vLLM
                response = requests.get(f"{cls.vllm_url}/health", timeout=10)
                if response.status_code == 200:
                    print("vLLM server is ready!")
                    return
                else:
                    print(f"vLLM server responded with status "
                          f"{response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"vLLM health check failed: {e}")

            # Check if container is still running
            try:
                result = subprocess.run([
                    "docker", "inspect", "--format={{.State.Running}}",
                    cls.docker_container_id
                ],
                                        capture_output=True,
                                        text=True,
                                        check=True)
                is_running = result.stdout.strip() == "true"
                if not is_running:
                    # Container stopped, get logs for debugging
                    logs_result = subprocess.run(
                        ["docker", "logs", cls.docker_container_id],
                        capture_output=True,
                        text=True)
                    print(f"vLLM container stopped unexpectedly. Logs:\n"
                          f"{logs_result.stdout}\n{logs_result.stderr}")
                    raise RuntimeError("vLLM container stopped unexpectedly")
            except subprocess.CalledProcessError as e:
                print(f"Failed to check container status: {e}")

            print(f"vLLM server not ready yet, waiting {check_interval}s...")
            time.sleep(check_interval)

        # If we get here, the server didn't start in time
        cls._stop_vllm_server()  # Clean up
        raise TimeoutError(
            f"vLLM server failed to start within {timeout} seconds")

    @classmethod
    def _stop_vllm_server(cls):
        """
        Stop and remove the vLLM Docker container.

        Performs graceful shutdown of the Docker container and cleanup:
        - Stops the running container
        - Removes the container to free resources
        - Resets container ID to None

        Handles errors gracefully to ensure cleanup attempts don't fail
        the overall test teardown process.
        """
        if cls.docker_container_id:
            print(f"Stopping vLLM container: {cls.docker_container_id}")
            try:
                # Stop the container
                subprocess.run(["docker", "stop", cls.docker_container_id],
                               capture_output=True,
                               check=True)
                # Remove the container
                subprocess.run(["docker", "rm", cls.docker_container_id],
                               capture_output=True,
                               check=True)
                print("vLLM container stopped and removed")
            except subprocess.CalledProcessError as e:
                print(f"Error stopping vLLM container: {e}")
            finally:
                cls.docker_container_id = None

    def test_vllm_server_health(self):
        """
        Validate vLLM server accessibility and health status.

        Tests the basic connectivity to the vLLM server by making a health
        check request to the /health endpoint. This ensures the server
        is running and responding before conducting more complex tests.
        """
        response = requests.get(f"{self.vllm_url}/health")
        assert response.status_code == 200

    def test_vllm_inference_call(self):
        """
        Test direct inference call to vLLM server using OpenAI-compatible API.

        Validates that the vLLM server can successfully process inference
        requests by sending a chat completion request and verifying the
        response format. This test confirms the server's text generation
        capabilities independently of ROS 2 integration.

        Tests:
            - OpenAI-compatible chat completions endpoint
            - Proper request/response format handling
            - Text generation with controlled parameters
        """
        # Example inference request using OpenAI-compatible API
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{
                "role": "user",
                "content": "Hello, how are you?"
            }],
            "max_tokens": 10,
            "temperature": TEST_TEMPERATURE,
            "seed": TEST_SEED
        }

        response = requests.post(f"{self.vllm_url}/v1/chat/completions",
                                 json=payload)
        assert response.status_code == 200

        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0
        assert "message" in result["choices"][0]
        print(f"Generated text: {result['choices'][0]['message']['content']}")

    def test_initialization_with_default_parameters(self):
        """
        Test ReplyActionServer initialization with default configuration.

        Validates that the ReplyActionServer can be successfully instantiated
        with default parameters and properly integrates with the ROS 2
        executor system. This test ensures basic node functionality before
        testing inference capabilities.

        Verifies:
            - Successful node creation and type validation
            - Correct action server name assignment
            - Proper integration with ROS 2 executor
        """
        self.reply_action_server = ReplyActionServer()
        self.executor.add_node(self.reply_action_server)

        # Check action server is created
        assert isinstance(self.reply_action_server, ReplyActionServer)
        assert (self.reply_action_server.action_server_name ==
                REPLY_ACTION_SERVER_NAME)

    def test_reply_action_with_real_vllm(self):
        """
        Test complete ReplyActionServer integration with vLLM backend.

        Executes the comprehensive end-to-end test using the shared test
        logic from BaseReplyActionServerTest. This validates the complete
        workflow from ROS 2 action goal submission through vLLM inference
        to result publication.

        Tests the full integration chain:
            - ROS 2 action server configuration with vLLM parameters
            - Action goal processing and acceptance
            - vLLM inference execution with real text generation
            - Result validation and topic publication verification
        """
        self._test_reply_action_with_server('vllm', self.vllm_url)
