import re
import threading
import time
from queue import Empty, Queue
from typing import Dict, Optional

import azure.cognitiveservices.speech as speechsdk
import rclpy
from action_msgs.msg import GoalStatus
from exodapt_robot_interfaces.action import ReplyAction
from rclpy.action import (ActionClient, ActionServer, CancelResponse,
                          GoalResponse)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class SentenceBuffer:
    """
    Buffers text chunks and detects sentence boundaries for TTS processing.
    """

    def __init__(self,
                 sentence_timeout: float = 3.0,
                 max_buffer_size: int = 1000):
        self.chunks = []
        self.sentence_timeout = sentence_timeout
        self.max_buffer_size = max_buffer_size
        self.last_chunk_time = time.time()
        self._lock = threading.Lock()

        # Sentence ending patterns
        self.sentence_endings = re.compile(r'[.!?:]')

    def add_chunk(self, chunk: str) -> Optional[str]:
        """
        Add a text chunk and return a complete sentence if detected.

        Returns:
            Complete sentence if boundary detected, None otherwise
        """
        with self._lock:
            if not chunk.strip():
                return None

            self.chunks.append(chunk)
            self.last_chunk_time = time.time()

            # Prevent buffer overflow
            if len(self.chunks) > self.max_buffer_size:
                self.chunks = self.chunks[-self.max_buffer_size:]

            current_text = ''.join(self.chunks)

            # Look for sentence ending followed by whitespace
            # Use regex to find all sentence endings
            matches = list(self.sentence_endings.finditer(current_text))

            if matches:
                # Check if any sentence ending is followed by whitespace
                for match in matches:
                    end_pos = match.end()
                    # Check if there's whitespace after the sentence ending
                    if (end_pos < len(current_text)
                            and current_text[end_pos].isspace()):
                        # Found complete sentence
                        sentence = current_text[:end_pos].strip()
                        remaining = current_text[end_pos:].lstrip()

                        # Update chunks with remaining text
                        self.chunks = [remaining] if remaining else []
                        return sentence

            return None

    def flush_buffer(self) -> Optional[str]:
        """
        Force flush the current buffer as a sentence.

        Returns:
            Current buffer content as sentence, or None if empty
        """
        with self._lock:
            if not self.chunks:
                return None

            sentence = ''.join(self.chunks).strip()
            self.chunks.clear()
            return sentence if sentence else None

    def should_timeout_flush(self) -> bool:
        """Check if buffer should be flushed due to timeout."""
        with self._lock:
            if not self.chunks:
                return False
            elapsed = time.time() - self.last_chunk_time
            return elapsed > self.sentence_timeout


class AzureTTSWorker:
    """
    Manages Azure TTS synthesis for text-to-speech conversion.
    """

    def __init__(
        self,
        speech_key: str,
        speech_endpoint: str,
        logger,
        voice_name: str = 'en-US-AvaMultilingualNeural',
    ):
        self.speech_key = speech_key
        self.speech_endpoint = speech_endpoint
        self.logger = logger
        self.voice_name = voice_name
        self._synthesizer = None
        self._lock = threading.Lock()

        # Initialize Azure TTS
        self._init_azure_tts()

    def _init_azure_tts(self):
        """Initialize Azure TTS synthesizer."""
        try:

            if not self.speech_key or not self.speech_endpoint:
                self.logger.warn(
                    'Azure Speech credentials not found in environment. '
                    'TTS will log sentences without audio synthesis.')
                return

            # Configure speech synthesis
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                endpoint=self.speech_endpoint,
            )
            speech_config.speech_synthesis_voice_name = self.voice_name

            # Use default speaker
            audio_config = speechsdk.audio.AudioOutputConfig(
                use_default_speaker=True)

            self._synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )

            self.logger.info('Azure TTS synthesizer initialized successfully')

        except Exception as e:
            self.logger.error(f'Failed to initialize Azure TTS: {str(e)}')
            self._synthesizer = None

    def synthesize_text(
        self,
        text: str,
        cancellation_event: threading.Event,
    ) -> bool:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            cancellation_event: Event to check for cancellation

        Returns:
            True if synthesis completed successfully, False otherwise
        """
        if cancellation_event.is_set():
            return False

        # If no synthesizer available, just log the text
        if not self._synthesizer:
            self.logger.info(f'TTS (no audio): {text}')
            return True

        try:
            # Start synthesis
            result_future = self._synthesizer.speak_text_async(text)

            # Azure Speech SDK's get() method blocks until completion
            # We can't easily interrupt it, so we check cancellation
            # before starting and let it complete if already started
            if cancellation_event.is_set():
                try:
                    result_future.cancel()
                except Exception:
                    pass
                return False

            # Get the result (this blocks until completion)
            result = result_future.get()

            completed_reason = speechsdk.ResultReason.SynthesizingAudioCompleted
            if result.reason == completed_reason:
                self.logger.info(f'TTS completed: {text[:50]}...')
                return True
            elif result.reason == speechsdk.ResultReason.Canceled:
                if not cancellation_event.is_set():
                    cancellation_details = result.cancellation_details
                    self.logger.warn(
                        f'TTS synthesis canceled: {cancellation_details.reason}'
                    )
                    if (cancellation_details.reason ==
                            speechsdk.CancellationReason.Error):
                        self.logger.error(
                            f'TTS error: {cancellation_details.error_details}')
                return False
            else:
                self.logger.error(f'TTS synthesis failed: {result.reason}')
                return False

        except Exception as e:
            if not cancellation_event.is_set():
                self.logger.error(f'TTS synthesis error: {str(e)}')
            return False


class TTSManager:
    """
    Manages TTS processing for a single goal, including sentence buffering
    and synthesis coordination.
    """

    def __init__(
        self,
        goal_uuid: tuple,
        logger,
        speech_key: str,
        speech_endpoint: str,
    ):
        self.goal_uuid = goal_uuid
        self.logger = logger
        self.speech_key = speech_key
        self.speech_endpoint = speech_endpoint

        # Components
        self.sentence_buffer = SentenceBuffer()
        self.tts_worker = AzureTTSWorker(
            speech_key,
            speech_endpoint,
            logger,
        )

        # Threading
        self.cancellation_event = threading.Event()
        self.synthesis_queue = Queue()
        self.synthesis_thread = None

        # Start synthesis thread
        self._start_synthesis_thread()

    def _start_synthesis_thread(self):
        """Start the background synthesis thread."""
        self.synthesis_thread = threading.Thread(target=self._synthesis_worker,
                                                 name=f'TTS-{self.goal_uuid}',
                                                 daemon=True)
        self.synthesis_thread.start()

    def _synthesis_worker(self):
        """Background worker for TTS synthesis."""
        while not self.cancellation_event.is_set():
            try:
                # Get next sentence with timeout
                sentence = self.synthesis_queue.get(timeout=1.0)

                if sentence is None:  # Shutdown signal
                    break

                # Synthesize the sentence
                self.tts_worker.synthesize_text(sentence,
                                                self.cancellation_event)

                self.synthesis_queue.task_done()

            except Empty:
                # Check for timeout flush
                if self.sentence_buffer.should_timeout_flush():
                    sentence = self.sentence_buffer.flush_buffer()
                    if sentence:
                        self.synthesis_queue.put(sentence)
                continue
            except Exception as e:
                self.logger.error(f'TTS synthesis worker error: {str(e)}')

        # After cancellation, process any remaining items in the queue
        # This ensures final sentences are still synthesized
        while not self.synthesis_queue.empty():
            try:
                sentence = self.synthesis_queue.get_nowait()
                if sentence is not None:  # Skip shutdown signal
                    msg = f'Processing final sentence: {sentence[:50]}...'
                    self.logger.info(msg)
                    # Create a new event that's not set for final synthesis
                    final_event = threading.Event()
                    self.tts_worker.synthesize_text(sentence, final_event)
                self.synthesis_queue.task_done()
            except Empty:
                break
            except Exception as e:
                self.logger.error(f'Error processing final sentence: {str(e)}')

    def process_chunk(self, chunk: str):
        """
        Process a text chunk, potentially triggering TTS synthesis.

        Args:
            chunk: Text chunk to process
        """
        if self.cancellation_event.is_set():
            return

        # Add chunk to buffer and check for complete sentence
        sentence = self.sentence_buffer.add_chunk(chunk)

        if sentence:
            # Queue sentence for synthesis
            try:
                self.synthesis_queue.put(sentence, timeout=1.0)
            except Exception as e:
                self.logger.warn(f'Failed to queue sentence for TTS: {str(e)}')

    def cancel(self):
        """Cancel TTS processing and cleanup resources."""
        self.logger.info(f'Canceling TTS manager for goal {self.goal_uuid}')

        # Flush any remaining buffer content and queue it for synthesis
        remaining_text = self.sentence_buffer.flush_buffer()
        if remaining_text:
            self.logger.info(f'Queueing final text: {remaining_text}')
            try:
                self.synthesis_queue.put(remaining_text, timeout=0.5)
            except Exception as e:
                self.logger.warn(f'Failed to queue final text: {str(e)}')

        # Signal cancellation - synthesis worker will process remaining queue
        self.cancellation_event.set()

        # Add shutdown signal
        self.synthesis_queue.put(None)

        # Wait for synthesis thread to finish
        if self.synthesis_thread and self.synthesis_thread.is_alive():
            self.synthesis_thread.join(timeout=3.0)


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
        self.declare_parameter('azure_speech_key', '')
        self.declare_parameter('azure_speech_endpoint', '')

        self.action_server_name = self.get_parameter(
            'action_server_name').value
        self.reply_action_server_name = self.get_parameter(
            'reply_action_server_name').value
        self.azure_speech_key = self.get_parameter('azure_speech_key').value
        self.azure_speech_endpoint = self.get_parameter(
            'azure_speech_endpoint').value

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

        # Maps goal_handle.goal_id.uuid to TTSManager instances
        self.tts_managers: Dict[tuple, TTSManager] = {}

        self.get_logger().info(
            'ReplyTTSActionServer initialized\n'
            'Parameters:\n'
            f'  action_server_name: {self.action_server_name}\n'
            f'  reply_action_server_name: {self.reply_action_server_name}\n'
            f'  azure_speech_key: {self.azure_speech_key[:8]}...\n'
            f'  azure_speech_endpoint: {self.azure_speech_endpoint}')

    def destroy(self):
        """Clean up resources including active TTS managers."""
        # Cancel all active TTS managers
        for goal_uuid, tts_manager in self.tts_managers.items():
            self.get_logger().info(
                f'Cleaning up TTS manager for goal {goal_uuid}')
            tts_manager.cancel()

        self.tts_managers.clear()
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
        reply_action_goal.state = goal_handle.request.state
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

            # Create or get TTS manager for this goal
            tts_manager = self.tts_managers.get(goal_uuid)
            if not tts_manager:
                tts_manager = TTSManager(
                    goal_uuid,
                    self.get_logger(),
                    self.azure_speech_key,
                    self.azure_speech_endpoint,
                )
                self.tts_managers[goal_uuid] = tts_manager

            # Check for cancellation request while waiting for goal completion
            result_future = reply_action_goal_handle.get_result_async()

            while not result_future.done():
                if goal_handle.is_cancel_requested:
                    self.get_logger().info(
                        'Canceling ReplyTTSActionServer goal')
                    cancel_future = reply_action_goal_handle.cancel_goal_async(
                    )
                    await cancel_future

                    # Clean up TTS manager
                    if goal_uuid in self.tts_managers:
                        self.tts_managers[goal_uuid].cancel()
                        del self.tts_managers[goal_uuid]

                    # Clean up goal tracking
                    if goal_uuid in self.active_goals:
                        del self.active_goals[goal_uuid]

                    goal_handle.canceled()
                    return ReplyAction.Result()

                time.sleep(0.1)

            # Get the ReplyActionServer result
            reply_action_result = await result_future

            # Clean up TTS manager
            if goal_uuid in self.tts_managers:
                self.tts_managers[goal_uuid].cancel()
                del self.tts_managers[goal_uuid]

            # Clean up goal tracking
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

            # Clean up TTS manager
            if goal_uuid in self.tts_managers:
                self.tts_managers[goal_uuid].cancel()
                del self.tts_managers[goal_uuid]

            # Clean up goal tracking
            if goal_uuid in self.active_goals:
                del self.active_goals[goal_uuid]

            goal_handle.abort()
            return ReplyAction.Result()

    def _feedback_callback(self, goal_handle, reply_action_feedback):
        """Perform TTS on intermediate feedback from the ReplyActionServer."""
        chunk = reply_action_feedback.feedback.streaming_resp
        self.get_logger().info(f'Feedback: {chunk}')

        # Get the TTS manager for this goal
        goal_uuid = tuple(goal_handle.goal_id.uuid)
        tts_manager = self.tts_managers.get(goal_uuid)

        if tts_manager:
            # Process the chunk through TTS
            tts_manager.process_chunk(chunk)
        else:
            self.get_logger().warn(
                f'No TTS manager found for goal {goal_uuid}')

        # Forward the feedback to clients
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
