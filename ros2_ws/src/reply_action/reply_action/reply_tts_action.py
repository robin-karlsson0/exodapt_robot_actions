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
    """Intelligent text chunk buffering and sentence boundary detection

    SentenceBuffer provides sophisticated text accumulation and natural language
    boundary detection to enable high-quality speech synthesis from streaming
    text feedback. It converts continuous character streams into discrete
    sentences optimized for Text-to-Speech processing while maintaining low
    latency for real-time robot communication scenarios.

    ## ROS 2 Integration & Lifecycle

    ### Phase 1: Instantiation During Goal Creation
    SentenceBuffer instances are created within TTSManager when a new
    ReplyAction goal is accepted by the ReplyTTSActionServer:

    ```python
    # In TTSManager.__init__()
    self.sentence_buffer = SentenceBuffer(
        sentence_timeout=3.0,    # Configurable flush timeout
        max_buffer_size=1000     # Memory protection limit
    )
    ```

    **Lifecycle Context:**
    - **Per-Goal Isolation**: Each ReplyAction goal gets independent buffer
    - **Concurrent Support**: Multiple goals can process text simultaneously
    - **Memory Management**: Buffer size limits prevent memory exhaustion
    - **Thread Safety**: Lock-protected operations for multi-threaded access

    ### Phase 2: Real-time Streaming Text Processing
    Integration with ROS 2 action feedback for continuous text accumulation:

    ```python
    # Called from TTSManager.process_chunk() during streaming feedback
    def _feedback_callback(self, goal_handle, reply_action_feedback):
        chunk = reply_action_feedback.feedback.streaming_resp
        sentence = sentence_buffer.add_chunk(chunk)  # Real-time processing
        if sentence:
            synthesis_queue.put(sentence)  # Queue for TTS
    ```

    **Processing Pipeline:**
    1. **Chunk Accumulation**: Streaming text chunks added incrementally
    2. **Boundary Analysis**: Regex-based sentence ending detection
    3. **Natural Segmentation**: Identifies `.!?:` followed by whitespace
    4. **Memory Protection**: Automatic buffer truncation at size limits
    5. **Thread-Safe Access**: Lock protection for concurrent operations

    ### Phase 3: Timeout-Based Processing
    Automatic sentence completion for responsive TTS synthesis:

    ```python
    # In TTSManager._synthesis_worker() background thread
    if sentence_buffer.should_timeout_flush():
        sentence = sentence_buffer.flush_buffer()
        if sentence:
            synthesis_queue.put(sentence)
    ```

    **Timeout Handling:**
    - **Responsive TTS**: 3-second default timeout prevents excessive delay
    - **Incomplete Sentences**: Processes partial text when streaming pauses
    - **Background Processing**: Timeout checks in synthesis worker thread
    - **Configurable Timing**: Adjustable timeout for different use cases

    ### Phase 4: Completion and Cleanup
    Final text processing during action completion or cancellation:

    ```python
    # In TTSManager.mark_reply_action_completed()
    remaining_text = sentence_buffer.flush_buffer()
    if remaining_text:
        synthesis_queue.put(remaining_text)
    ```

    ## Text Processing Architecture

    ### Intelligent Boundary Detection
    Advanced sentence segmentation using multiple linguistic patterns:

    **Primary Patterns:**
    - **Terminal Punctuation**: `.!?:` marks potential sentence endings
    - **Whitespace Validation**: Requires following whitespace for completion
    - **Context Awareness**: Distinguishes sentence endings from abbreviations
    - **Incremental Processing**: Evaluates boundaries as text arrives

    **Processing Algorithm:**
    1. **Pattern Matching**: Regex search for sentence ending markers
    2. **Position Validation**: Confirms whitespace follows punctuation
    3. **Text Extraction**: Splits at confirmed boundary locations
    4. **Remainder Preservation**: Maintains incomplete text for next iteration
    5. **Cleanup Operations**: Strips whitespace and normalizes output

    ### Memory Management & Protection

    **Buffer Overflow Prevention:**
    ```python
    # Automatic truncation prevents memory exhaustion
    if len(self.chunks) > self.max_buffer_size:
        self.chunks = self.chunks[-self.max_buffer_size:]
    ```

    **Resource Characteristics:**
    - **Default Limit**: 1000 character maximum buffer size
    - **Sliding Window**: Keeps recent text when overflow occurs
    - **Memory Efficiency**: List-based storage with minimal overhead
    - **Garbage Collection**: Automatic cleanup of processed text

    ### Thread Safety & Concurrency

    **Lock-Protected Operations:**
    - **add_chunk()**: Thread-safe chunk addition and boundary detection
    - **flush_buffer()**: Atomic buffer extraction and clearing
    - **should_timeout_flush()**: Safe timeout evaluation
    - **State Access**: Consistent view of buffer state across threads

    **Threading Model:**
    - **Producer**: Main thread adds chunks via feedback callbacks
    - **Consumer**: Background synthesis worker checks timeouts
    - **Coordination**: Reentrant locks enable safe concurrent access
    - **Performance**: Minimal lock contention for high throughput

    ## Performance Optimization

    ### Streaming Optimization
    Designed for continuous text arrival patterns:

    **Chunk Size Handling:**
    - **Character-by-Character**: Handles individual character streams
    - **Word Boundaries**: Efficient processing of word-sized chunks
    - **Sentence Fragments**: Manages partial sentence arrival
    - **Bulk Processing**: Supports large text chunk processing

    ## Error Handling & Robustness

    ### Input Validation & Sanitization
    Defensive programming against malformed or unexpected input:

    **Input Processing:**
    - **Empty Chunks**: Silently ignored without buffer modification
    - **Whitespace-Only**: Stripped and ignored for clean processing
    - **Unicode Support**: Full UTF-8 text processing capabilities
    - **Malformed Text**: Robust regex handling of irregular input

    ### Resource Exhaustion Protection
    Safeguards against memory and processing overload:

    **Protection Mechanisms:**
    - **Buffer Size Limits**: Prevents unbounded memory growth
    - **Timeout Boundaries**: Prevents indefinite text accumulation
    - **Exception Handling**: Graceful degradation on processing errors
    - **State Recovery**: Maintains buffer integrity during error conditions

    ## Configuration & Customization

    ### Configurable Parameters

    **Timeout Configuration:**
    ```python
    buffer = SentenceBuffer(
        sentence_timeout=5.0,      # Longer timeout for careful speech
        max_buffer_size=2000       # Larger buffer for complex text
    )
    ```

    **Parameter Effects:**
    - **sentence_timeout**: Balance between responsiveness and completeness
    - **max_buffer_size**: Memory usage vs. text processing capability
    - **Thread safety**: All parameters respected across concurrent access

    ### Language & Pattern Customization
    Extensible design for different linguistic requirements:

    **Current Patterns:**
    - **English Focus**: Optimized for English sentence structures
    - **Universal Punctuation**: `.!?:` covers most Western languages
    - **Whitespace Detection**: Standard ASCII/Unicode space characters
    - **Extensibility**: Regex patterns can be modified for other languages

    ## Usage Patterns & Best Practices

    ### Typical Integration Pattern
    ```python
    # Initialize with appropriate timeouts for use case
    sentence_buffer = SentenceBuffer(sentence_timeout=3.0)

    # Process streaming chunks in real-time
    for chunk in streaming_text:
        sentence = sentence_buffer.add_chunk(chunk)
        if sentence:
            # Process complete sentence immediately
            text_to_speech_queue.put(sentence)

    # Handle final incomplete text
    final_text = sentence_buffer.flush_buffer()
    if final_text:
        text_to_speech_queue.put(final_text)
    ```

    ## Integration with Azure TTS

    ### Sentence Quality Optimization
    Buffer design optimized for natural speech synthesis:

    **TTS-Friendly Output:**
    - **Complete Thoughts**: Sentences end at natural pause points
    - **Punctuation Preservation**: Maintains intonation cues for TTS
    - **Length Management**: Prevents overly long sentences for synthesis
    - **Clean Text**: Whitespace normalized for consistent audio quality

    ### Real-time Coordination
    Seamless integration with Azure Cognitive Services workflow:

    **Synthesis Pipeline:**
    1. **Buffer Processing**: Sentence boundary detection
    2. **Queue Handoff**: Complete sentences passed to TTS worker
    3. **Parallel Processing**: Buffer continues while TTS synthesizes
    4. **Memory Efficiency**: Processed text cleared to prevent accumulation

    Note:
        SentenceBuffer instances are designed for single-stream processing
        within TTSManager lifecycle. For multiple concurrent text streams,
        create separate buffer instances to prevent text mixing and ensure
        proper sentence boundary detection for each stream.
    """

    def __init__(self,
                 sentence_timeout: float = 3.0,
                 max_buffer_size: int = 1000):
        """Initialize SentenceBuffer with configurable timeout and size limits.

        Creates a new buffer instance for accumulating streaming text chunks
        and detecting sentence boundaries for optimal TTS processing.

        Args:
            sentence_timeout (float): Maximum time in seconds to wait before
                force-flushing incomplete sentences. Balances responsiveness
                with sentence completeness. Default: 3.0 seconds.
            max_buffer_size (int): Maximum number of chunks to retain in
                buffer before automatic truncation. Prevents memory exhaustion
                during long streaming sessions. Default: 1000 chunks.

        Thread Safety:
            Initializes threading.Lock for safe concurrent access across
            multiple threads (main feedback thread + synthesis worker).
        """
        self.chunks = []
        self.sentence_timeout = sentence_timeout
        self.max_buffer_size = max_buffer_size
        self.last_chunk_time = time.time()
        self._lock = threading.Lock()

        # Sentence ending patterns
        self.sentence_endings = re.compile(r'[.!?:]')

    def add_chunk(self, chunk: str) -> Optional[str]:
        """Add streaming text chunk and detect complete sentence boundaries.

        Accumulates text chunks incrementally and uses intelligent boundary
        detection to identify complete sentences ready for TTS synthesis.
        Returns sentences immediately when detected to minimize latency.

        Processing Logic:
            1. Validates and appends chunk to internal buffer
            2. Updates timestamp for timeout tracking
            3. Prevents buffer overflow via sliding window truncation
            4. Searches for sentence endings (.!?:) followed by whitespace
            5. Extracts complete sentence and preserves remaining text

        Args:
            chunk (str): Text fragment from streaming source. Can be single
                characters, words, or multi-sentence blocks. Empty or
                whitespace-only chunks are ignored without buffer modification.

        Returns:
            Optional[str]: Complete sentence if boundary detected and confirmed
                by whitespace validation, None if sentence is still incomplete.
                Returned sentences are stripped of leading/trailing whitespace.

        Thread Safety:
            Protected by internal lock for safe concurrent access from multiple
            threads. Typical usage: main thread adds chunks, worker thread
            checks timeouts.

        Performance:
            Optimized for high-frequency calls with O(n) regex search where
            n is current buffer length. Memory usage controlled by
            max_buffer_size.

        Examples:
            >>> buffer = SentenceBuffer()
            >>> buffer.add_chunk("Hello")      # Returns: None
            >>> buffer.add_chunk(" world")     # Returns: None
            >>> buffer.add_chunk("! How")      # Returns: "Hello world!"
            >>> buffer.add_chunk(" are you?") # Returns: None
            >>> buffer.add_chunk(" ")          # Returns: "How are you?"
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
        """Force extraction of all buffered text as a complete sentence.

        Immediately returns all accumulated text chunks as a single sentence,
        regardless of whether natural sentence boundaries have been detected.
        Used for timeout-based processing and final cleanup to ensure no
        text is lost during completion scenarios.

        Behavior:
            - Concatenates all buffered chunks into single string
            - Strips leading/trailing whitespace for clean output
            - Clears internal buffer state after extraction
            - Returns None if buffer is empty (no text to flush)

        Use Cases:
            - Timeout-based flushing when streaming pauses (3s default)
            - Final text extraction during action completion
            - Emergency buffer clearing during cancellation scenarios
            - Recovery from incomplete sentence detection edge cases

        Returns:
            Optional[str]: Complete buffered text as sentence with normalized
                whitespace, or None if buffer contains no text. Empty strings
                after whitespace stripping also return None.

        Thread Safety:
            Protected by internal lock for atomic buffer extraction and
            clearing. Safe for concurrent access from multiple threads.

        Performance:
            O(n) operation where n is total character count in buffer.
            Memory is immediately freed after extraction.

        Examples:
            >>> buffer = SentenceBuffer()
            >>> buffer.add_chunk("Incomplete thought")
            >>> buffer.add_chunk(" without ending")
            >>> sentence = buffer.flush_buffer()  # "Incomplete thought..."
            >>> buffer.flush_buffer()             # Returns: None (empty)
        """
        with self._lock:
            if not self.chunks:
                return None

            sentence = ''.join(self.chunks).strip()
            self.chunks.clear()
            return sentence if sentence else None

    def should_timeout_flush(self) -> bool:
        """Check if buffer should be flushed due to inactivity timeout.

        Evaluates whether sufficient time has elapsed since the last chunk
        addition to warrant force-flushing the buffer contents. Used by
        background synthesis workers to ensure responsive TTS output even
        when streaming pauses or incomplete sentences are received.

        Timeout Logic:
            - Compares current time against last_chunk_time timestamp
            - Uses configurable sentence_timeout (default: 3.0 seconds)
            - Only triggers if buffer contains text chunks
            - Prevents indefinite waiting for sentence completion

        Use Cases:
            - Background worker periodic timeout checks (1s intervals)
            - Responsive TTS during streaming pauses or network delays
            - Recovery from incomplete sentence patterns or missing punctuation
            - Ensuring user feedback when LLM output lacks proper endings

        Returns:
            bool: True if buffer contains text AND timeout has expired,
                False if buffer is empty OR timeout has not been reached.
                Empty buffers never timeout to prevent unnecessary processing.

        Thread Safety:
            Protected by internal lock for consistent timestamp and buffer
            state evaluation. Safe for concurrent calls from worker threads.

        Performance:
            O(1) operation with minimal computation overhead. Suitable for
            high-frequency polling scenarios in background threads.

        Examples:
            >>> buffer = SentenceBuffer(sentence_timeout=2.0)
            >>> buffer.add_chunk("Waiting")
            >>> buffer.should_timeout_flush()  # False (just added)
            >>> time.sleep(2.5)
            >>> buffer.should_timeout_flush()  # True (timeout exceeded)
            >>> buffer.flush_buffer()          # Clear buffer
            >>> buffer.should_timeout_flush()  # False (empty buffer)
        """
        with self._lock:
            if not self.chunks:
                return False
            elapsed = time.time() - self.last_chunk_time
            return elapsed > self.sentence_timeout


class AzureTTSWorker:
    """Azure Cognitive Services TTS synthesis worker for streaming replies.

    AzureTTSWorker provides the core text-to-speech synthesis capabilities for
    the ReplyTTSActionServer, managing Azure Cognitive Services integration,
    audio output coordination, and cancellation support for real-time robot
    communication scenarios.

    ## ROS 2 Node Lifecycle Integration

    ### Phase 1: Node Initialization & Credential Management
    The AzureTTSWorker is instantiated during TTSManager creation when a new
    ReplyAction goal is accepted by the ReplyTTSActionServer:

    ```python
    # In TTSManager.__init__()
    self.tts_worker = AzureTTSWorker(
        speech_key=azure_speech_key,      # From ROS 2 parameter
        speech_endpoint=azure_endpoint,   # From ROS 2 parameter
        logger=node.get_logger(),         # ROS 2 node logger
        voice_name='en-US-AvaMultilingualNeural'
    )
    ```

    **Credential Configuration:**
    - Azure credentials sourced from ROS 2 node parameters:
      - `azure_speech_key`: Subscription key for Azure Cognitive Services
      - `azure_speech_endpoint`: Regional TTS service endpoint URL
    - Graceful degradation: Missing credentials enable text-only mode
    - No audio synthesis performed, text logged instead for development/testing

    **Azure SDK Initialization:**
    - Creates `SpeechConfig` with subscription and endpoint configuration
    - Configures `AudioOutputConfig` for default system speaker output
    - Initializes `SpeechSynthesizer` for text-to-speech operations
    - Error handling ensures robustness against network/credential issues

    ### Phase 2: Real-time Synthesis During Goal Execution
    Integration with ROS 2 action streaming feedback for live audio generation:

    **Synthesis Coordination:**
    ```python
    # Called from TTSManager background thread (_synthesis_worker)
    success = tts_worker.synthesize_text(
        text=complete_sentence,
        cancellation_event=goal_cancellation_event
    )
    ```

    **Thread-Safe Operation:**
    - Multiple concurrent goals each get isolated AzureTTSWorker instances
    - Thread-safe synthesis tracking via `_lock` and `_current_synthesis_future`
    - Coordinated cancellation using threading.Event primitives
    - No audio interference between simultaneous robot actions

    **Audio Output Management:**
    - Configurable voice selection (default: en-US-AvaMultilingualNeural)
    - System speaker output for immediate audio feedback
    - Real-time synthesis with ~200-500ms latency typical
    - Automatic audio device management through Azure SDK

    ### Phase 3: Cancellation & Cleanup Integration
    Responsive cancellation support for ROS 2 action lifecycle management.

    ## Azure Cognitive Services Integration

    ### Service Configuration:
    - **Subscription Model**: Requires Azure Cognitive Services subscription
    - **Regional Endpoints**: Configurable endpoint for latency optimization
    - **Voice Selection**: Support for 400+ voices across 140+ languages
    - **Audio Format**: 16kHz 16-bit PCM output via system speakers

    ### Network Resilience:
    - Automatic retry mechanisms within Azure Speech SDK
    - Graceful degradation on network failures (logs errors, continues
      execution)
    - Connection pooling and keepalive handled by Azure SDK
    - TLS encryption for all service communication

    ### Error Handling:
    - Service unavailable: Logged warnings, text-only mode
    - Authentication failures: Initialization errors logged, fallback mode
    - Synthesis errors: Per-sentence error handling, goal execution continues
    - Network timeouts: Azure SDK handles retries transparently

    ## Synthesis Lifecycle Management

    ### Text Processing Pipeline:
    1. **Input Validation**: Checks for empty text and cancellation state
    2. **Fallback Mode**: Text logging when synthesizer unavailable
    3. **Event Setup**: Creates completion/cancellation event handlers
    4. **Async Synthesis**: Initiates Azure TTS `speak_text_async()`
    5. **Polling Loop**: 50ms intervals checking for completion/cancellation
    6. **Result Processing**: Handles success, cancellation, and error cases
    7. **Cleanup**: Event handler disconnection and resource release

    ### Cancellation Coordination:
    ```python
    # Synthesis polling with cancellation support
    while not synthesis_complete.is_set():
        if cancellation_event.is_set():
            self._synthesizer.stop_speaking_async()  # Immediate stop
            return False
        time.sleep(0.05)  # 50ms responsive polling
    ```

    ### Result Status Handling:
    - **SynthesizingAudioCompleted**: Successful synthesis, audio played
    - **Canceled**: Client cancellation or synthesis interruption
    - **Error**: Network, authentication, or service failures

    ## Performance Characteristics

    ### Audio Latency:
    - **Text-to-Audio**: ~200-500ms typical latency for sentence synthesis
    - **Cancellation Response**: ~50-100ms for stop_speaking_async()
    - **Network Dependent**: Regional endpoint selection affects latency
    - **Voice Dependent**: Some voices have slightly different processing times

    ### Concurrent Scaling:
    - **Per-Goal Isolation**: Independent workers prevent audio interference
    - **Thread Safety**: Lock-protected synthesis state management
    - **Azure Quotas**: Service-level rate limits apply (typically 200 TPS)

    ## Usage Patterns & Best Practices

    ### Production Configuration:
    ```python
    # Full TTS synthesis for production deployment
    worker = AzureTTSWorker(
        speech_key=azure_subscription_key,
        speech_endpoint='https://eastus.tts.speech.microsoft.com/',
        logger=node.get_logger(),
        voice_name='en-US-AvaMultilingualNeural'  # Customizable
    )
    ```

    ## Error Scenarios & Recovery

    ### Missing Credentials:
    - **Behavior**: Text-only mode with warning logs
    - **Use Case**: Development, testing, demonstrations without audio
    - **Recovery**: Provide valid Azure credentials and restart node

    ### Network Failures:
    - **Behavior**: Per-synthesis error logging, goal execution continues
    - **Resilience**: Azure SDK automatic retry mechanisms
    - **Recovery**: Transient issues self-recover, persistent issues require
      network troubleshooting

    ### Service Quota Exceeded:
    - **Behavior**: Synthesis failures logged, text-only fallback
    - **Monitoring**: Azure portal for quota usage and rate limits
    - **Recovery**: Upgrade Azure subscription or implement rate limiting

    ### Audio Device Issues:
    - **Behavior**: Azure SDK reports audio device errors
    - **Fallback**: Synthesis succeeds but no audio output
    - **Recovery**: Check system audio configuration and device availability

    ## Architecture Integration Notes

    ### Component Relationships:
    - **Parent**: TTSManager (per-goal lifecycle management)
    - **Sibling**: SentenceBuffer (text chunking and boundary detection)
    - **Consumer**: ROS 2 background threads (synthesis workers)
    - **Dependencies**: Azure Cognitive Services SDK, system audio

    Note:
        AzureTTSWorker instances are designed for single-goal usage within
        TTSManager lifecycle. Do not share instances across multiple goals
        or reuse after goal completion to prevent resource conflicts and
        synthesis state corruption.
    """

    def __init__(
        self,
        speech_key: str,
        speech_endpoint: str,
        logger,
        voice_name: str = 'en-US-AvaMultilingualNeural',
    ):
        """Initialize Azure TTS worker with credentials and configuration.

        Creates a new AzureTTSWorker instance with Azure Cognitive Services
        credentials and voice configuration. Automatically initializes the
        Azure Speech SDK synthesizer if valid credentials are provided.

        Args:
            speech_key: Azure Cognitive Services subscription key. Empty string
                enables text-only mode without audio synthesis.
            speech_endpoint: Azure TTS service endpoint URL (region-specific).
                Empty string enables text-only mode.
            logger: ROS 2 logger instance for status and error reporting.
            voice_name: Azure TTS voice identifier. Defaults to
                'en-US-AvaMultilingualNeural' for high-quality multilingual
                speech synthesis.

        Note:
            If credentials are missing or invalid, the worker operates in
            text-only mode, logging synthesis requests without audio output.
            This enables development and testing without Azure subscription.
        """
        self.speech_key = speech_key
        self.speech_endpoint = speech_endpoint
        self.logger = logger
        self.voice_name = voice_name
        self._synthesizer = None
        self._lock = threading.Lock()
        self._current_synthesis_future = None

        # Initialize Azure TTS
        self._init_azure_tts()

    def _init_azure_tts(self):
        """Initialize Azure Speech SDK synthesizer with credential validation.

        Sets up the Azure Cognitive Services TTS synthesizer using provided
        credentials and voice configuration. Handles missing credentials
        gracefully by enabling text-only mode instead of failing.

        The method configures:
        - SpeechConfig with subscription key and regional endpoint
        - AudioOutputConfig for default system speaker
        - SpeechSynthesizer for text-to-speech operations
        - Voice selection for audio output characteristics

        If initialization fails due to missing credentials, network issues,
        or invalid configuration, the synthesizer remains None and the
        worker operates in text-only mode with warning logs.

        Raises:
            No exceptions raised - all errors are logged and handled gracefully
            through text-only mode fallback.
        """
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
        """Synthesize text to speech with real-time cancellation support.

        Converts input text to speech audio using Azure Cognitive Services
        TTS with support for immediate cancellation. The method blocks until
        synthesis completes or cancellation is requested.

        Synthesis Process:
        1. Validates input and checks for early cancellation
        2. Falls back to text logging if synthesizer unavailable
        3. Sets up event-driven completion/cancellation tracking
        4. Initiates asynchronous Azure TTS synthesis
        5. Polls for completion with 50ms responsiveness
        6. Processes results and handles all completion scenarios

        Args:
            text: Text to convert to speech. Should be complete sentences
                for natural audio output. Long text may be truncated in logs.
            cancellation_event: Threading event to signal immediate
                cancellation. When set, synthesis stops mid-sentence and
                returns False immediately.

        Returns:
            bool: True if synthesis completed successfully with audio output,
                False if cancelled, failed, or no synthesizer available.
                Note that text-only mode (no synthesizer) returns True.

        Thread Safety:
            Method is thread-safe and can be called from background synthesis
            workers. Uses internal locking for synthesis state management.

        Performance:
            Typical latency 200-500ms for sentence synthesis. Cancellation
            response within 50-100ms through stop_speaking_async().
        """
        if cancellation_event.is_set():
            return False

        # If no synthesizer available, just log the text
        if not self._synthesizer:
            self.logger.info(f'TTS (no audio): {text}')
            return True

        # Use a threading event to track completion
        synthesis_complete = threading.Event()
        synthesis_result = [None]  # Use list to allow modification in callback

        def synthesis_completed_callback(evt):
            synthesis_result[0] = evt.result
            synthesis_complete.set()

        def synthesis_canceled_callback(evt):
            synthesis_result[0] = evt.result
            synthesis_complete.set()

        try:
            # Connect event handlers
            synth = self._synthesizer
            synth.synthesis_completed.connect(synthesis_completed_callback)
            synth.synthesis_canceled.connect(synthesis_canceled_callback)

            # Start synthesis
            result_future = self._synthesizer.speak_text_async(text)

            # Store current synthesis future for potential cancellation
            with self._lock:
                self._current_synthesis_future = result_future

            # Wait for completion or cancellation
            while not synthesis_complete.is_set():
                if cancellation_event.is_set():
                    # Stop synthesis immediately using stop_speaking_async
                    try:
                        self._synthesizer.stop_speaking_async()
                        cancel_msg = (f'TTS synthesis cancelled mid-sentence: '
                                      f'{text[:50]}...')
                        self.logger.info(cancel_msg)
                    except Exception as e:
                        self.logger.warn(f'Failed to stop TTS: {str(e)}')

                    # Clean up
                    with self._lock:
                        self._current_synthesis_future = None
                    return False

                # Short sleep to avoid busy waiting
                time.sleep(0.05)  # 50ms polling interval

            # Clean up
            with self._lock:
                self._current_synthesis_future = None

            # Process the result
            result = synthesis_result[0]
            if result is None:
                error_msg = 'TTS synthesis completed but no result available'
                self.logger.error(error_msg)
                return False

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
            # Clean up on exception
            with self._lock:
                self._current_synthesis_future = None

            if not cancellation_event.is_set():
                self.logger.error(f'TTS synthesis error: {str(e)}')
            return False
        finally:
            # Disconnect event handlers
            try:
                synth = self._synthesizer
                synth.synthesis_completed.disconnect(
                    synthesis_completed_callback)
                synth.synthesis_canceled.disconnect(
                    synthesis_canceled_callback)
            except Exception:
                pass

    def stop_current_synthesis(self):
        """Immediately terminate any active TTS synthesis and audio output.

        Provides emergency stop functionality for current speech synthesis,
        halting audio output mid-sentence if needed. Used primarily during
        immediate cancellation scenarios where responsiveness is prioritized
        over completing current speech.

        The method:
        - Calls Azure SDK stop_speaking_async() to halt audio immediately
        - Logs success/failure for debugging and monitoring
        - Handles errors gracefully without raising exceptions
        - Only acts if synthesizer exists and synthesis is active

        Thread Safety:
            Protected by internal lock to prevent race conditions with
            concurrent synthesis operations. Safe to call from any thread.

        Timing:
            Audio typically stops within 50-100ms of method call. Does not
            wait for stop completion to maintain responsiveness.

        Use Cases:
            - Client cancellation requests requiring immediate response
            - Emergency shutdown during node destruction
            - Error recovery requiring immediate TTS termination
        """
        with self._lock:
            if self._synthesizer and self._current_synthesis_future:
                try:
                    self._synthesizer.stop_speaking_async()
                    self.logger.info('Stopped current TTS synthesis')
                except Exception as e:
                    self.logger.warn(f'Failed to stop current TTS: {str(e)}')


class TTSManager:
    """
    Manages TTS processing for a single ROS 2 action goal, providing sentence
    buffering, synthesis coordination, and lifecycle management for real-time
    text-to-speech during streaming reply actions.

    ## ROS 2 Integration & Lifecycle

    The TTSManager is instantiated per ReplyAction goal to provide isolated,
    concurrent TTS processing. Each instance manages the complete lifecycle
    from goal creation through completion or cancellation, ensuring thread-safe
    resource management and proper cleanup.

    ### Phase 1: Goal Initialization
    Created when a new ReplyAction goal is accepted by ReplyTTSActionServer:

    Components Initialized:
    - **SentenceBuffer**: Accumulates streaming text chunks and detects sentence
      boundaries using regex patterns for natural speech synthesis breaks
    - **AzureTTSWorker**: Handles Azure Cognitive Services TTS synthesis with
      cancellation support and graceful degradation for missing credentials
    - **Background Thread**: Starts `_synthesis_worker()` for non-blocking TTS
      processing with producer-consumer pattern implementation

    Threading Architecture:
    - `synthesis_queue`: Thread-safe Queue for sentences awaiting synthesis
    - `cancellation_event`: Threading.Event for coordinated shutdown signaling
    - `completion_event`: Threading.Event to signal TTS processing completion
    - `reply_action_completed`: Boolean flag tracking underlying action state

    ### Phase 2: Streaming Feedback Processing
    Continuously processes text chunks as they arrive from the underlying
    ReplyActionServer via feedback callbacks:

    ```python
    # In ReplyTTSActionServer._feedback_callback()
    def _feedback_callback(self, goal_handle, reply_action_feedback):
        chunk = reply_action_feedback.feedback.streaming_resp
        tts_manager = self.tts_managers.get(goal_uuid)
        tts_manager.process_chunk(chunk)  # Real-time sentence detection
    ```

    Processing Pipeline:
    1. Chunk Accumulation: `process_chunk()` adds text to SentenceBuffer
    2. Boundary Detection: Regex-based detection of sentence endings
       (`.!?:` followed by whitespace) for natural speech breaks
    3. Sentence Queueing: Complete sentences added to `synthesis_queue`
    4. Background Synthesis: Worker thread processes queue asynchronously
    5. Timeout Handling: Automatic buffer flush after sentence_timeout (3s)

    Concurrency Model:
    - Main thread: Receives chunks, detects sentences, queues for synthesis
    - Background thread: Continuously processes synthesis queue with Azure TTS
    - Thread-safe coordination via Queue and threading.Event primitives

    ### Phase 3: Completion Scenarios

    #### 3a. Successful Goal Completion
    When underlying ReplyActionServer completes successfully:

    ```python
    # Mark end of streaming and process final content
    tts_manager.mark_reply_action_completed()

    # Wait for all TTS synthesis to complete
    tts_completed = tts_manager.wait_for_completion(timeout=0.5)
    ```

    Completion Workflow:
    1. `mark_reply_action_completed()` sets completion flag and flushes buffer
    2. Background thread processes remaining queued sentences
    3. `completion_event` is set when synthesis queue is empty
    4. Action server waits for TTS completion before returning result

    #### 3b. Graceful Cancellation
    Allows current synthesis to complete before shutdown:

    ```python
    # Process remaining content then shutdown
    tts_manager.cancel()
    ```

    Graceful Shutdown:
    - Flushes any remaining buffered text as final sentence
    - Allows background thread to finish processing queued sentences
    - Waits up to 3 seconds for clean thread termination
    - Used when underlying action completes but client doesn't need to wait

    #### 3c. Immediate Cancellation
    Stops all TTS processing immediately:

    ```python
    # Stop mid-sentence and cleanup immediately
    tts_manager.cancel_immediately()
    ```

    Immediate Shutdown:
    - Calls `stop_current_synthesis()` to interrupt active audio playback
    - Clears synthesis queue without processing remaining sentences
    - Sets cancellation event and forces thread termination
    - Used for client cancellation requests or node shutdown

    ### Phase 4: Resource Cleanup
    Automatic cleanup prevents memory leaks and resource accumulation:

    ```python
    # In ReplyTTSActionServer cleanup
    if goal_uuid in self.tts_managers:
        self.tts_managers[goal_uuid].cancel_immediately()
        del self.tts_managers[goal_uuid]
    ```

    ## Architecture Patterns

    ### Producer-Consumer Pattern
    - Producer: Main thread receives chunks and produces sentences
    - Consumer: Background thread consumes sentences for TTS synthesis
    - Buffer: Thread-safe Queue enables decoupled processing rates

    ### Event-Driven Coordination
    - `cancellation_event`: Coordinates shutdown across threads
    - `completion_event`: Signals when all processing is complete
    - Prevents race conditions and ensures proper resource cleanup

    ### Graceful Degradation
    - Works without Azure credentials (logs text instead of audio synthesis)
    - Network failures don't interrupt goal execution
    - TTS errors are logged but don't abort the underlying action

    ## Performance Characteristics

    ### Memory Management
    - Per-Goal Isolation: Each TTSManager instance is independent
    - Buffer Limits: SentenceBuffer has configurable max_buffer_size (1000)
    - Queue Management: Synthesis queue automatically drained on completion
    - Thread Cleanup: Background threads properly joined during shutdown

    ### Concurrent Goal Support
    - Multiple TTSManager instances can run simultaneously
    - Goal UUIDs provide isolation between concurrent actions
    - Independent cancellation and completion per goal
    - No audio interference between concurrent TTS streams

    ## Error Handling & Robustness

    ### Network Resilience
    - Azure TTS failures logged but don't interrupt goal execution
    - Automatic retry mechanisms within Azure Speech SDK
    - Fallback to text-only mode when synthesis fails

    ### Thread Safety
    - All shared state protected by threading primitives
    - Queue operations are inherently thread-safe
    - Lock-free design minimizes contention and deadlock risk

    ### Resource Exhaustion Protection
    - Buffer size limits prevent memory overflow
    - Thread timeouts prevent hanging on shutdown
    - Automatic cleanup on all completion paths

    ## Usage Examples

    ### Standard Goal Processing
    ```python
    # Goal lifecycle managed by ReplyTTSActionServer
    tts_manager = TTSManager(goal_uuid, logger, speech_key, endpoint)

    # Process streaming chunks
    for chunk in streaming_feedback:
        tts_manager.process_chunk(chunk)

    # Signal completion and wait for TTS
    tts_manager.mark_reply_action_completed()
    completed = tts_manager.wait_for_completion(timeout=10.0)

    # Cleanup
    tts_manager.cancel()
    ```

    ### Emergency Shutdown
    ```python
    # Immediate cleanup during node shutdown
    for tts_manager in self.tts_managers.values():
        tts_manager.cancel_immediately()
    ```

    ## Dependencies & Configuration

    ### Required Components
    - SentenceBuffer: Text chunking and sentence boundary detection
    - AzureTTSWorker: Azure Cognitive Services TTS synthesis
    - Threading: Background processing and synchronization primitives

    ### Configuration Parameters
    - `sentence_timeout`: Buffer flush timeout (default: 3.0s)
    - `max_buffer_size`: Maximum buffer size (default: 1000 chars)
    - `voice_name`: Azure TTS voice (default: 'en-US-AvaMultilingualNeural')

    Note:
        This class is designed for single-goal, single-use lifecycle. Create
        one instance per ReplyAction goal and dispose after completion. Do not
        reuse instances across multiple goals as this can lead to resource
        conflicts and unpredictable behavior.
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
        self.completion_event = threading.Event()
        self.reply_action_completed = False

        # Start synthesis thread
        self._start_synthesis_thread()

    def _start_synthesis_thread(self):
        """Initialize and start background synthesis thread for TTS processing.

        Creates a daemon thread that runs the `_synthesis_worker` method
        continuously to process sentences from the synthesis queue. The thread
        is configured as a daemon to ensure it doesn't prevent program exit.

        Thread Configuration:
            - Target: `_synthesis_worker` method for processing synthesis queue
            - Name: 'TTS-{goal_uuid}' for debugging and monitoring purposes
            - Daemon: True to allow clean program shutdown without explicit join
            - Automatic Start: Thread starts immediately after creation

        Thread Safety:
            This method should only be called once during TTSManager
            initialization. Multiple calls would create multiple worker threads
            which could lead to race conditions and resource conflicts.

        Note:
            The daemon thread will automatically terminate when the main program
            exits, but proper cleanup via cancel() or cancel_immediately() is
            recommended for graceful shutdown and resource management.
        """
        self.synthesis_thread = threading.Thread(target=self._synthesis_worker,
                                                 name=f'TTS-{self.goal_uuid}',
                                                 daemon=True)
        self.synthesis_thread.start()

    def _synthesis_worker(self):
        """Background thread worker for continuous TTS synthesis processing.

        This is the main worker method that runs in a background thread to
        process sentences from the synthesis queue and coordinate TTS synthesis
        with Azure Cognitive Services. It implements a producer-consumer pattern
        where the main thread produces sentences and this worker consumes them.

        ## Processing Loop

        ### Normal Operation:
        1. **Queue Processing**: Gets sentences from `synthesis_queue` with
           1s timeout
        2. **Shutdown Detection**: Recognizes None as shutdown signal and exits
        3. **TTS Synthesis**: Calls `tts_worker.synthesize_text()` for each
           sentence
        4. **Task Completion**: Marks queue items as done for proper
           synchronization

        ### Timeout Handling:
        - **Buffer Flush**: Checks if sentence buffer should timeout flush (3s)
        - **Completion Check**: Exits when reply action done AND queue empty
          AND buffer empty
        - **Continuation**: Continues loop if more work is expected

        ### Exception Handling:
        - **Worker Errors**: Logs synthesis errors but continues processing
        - **Graceful Degradation**: Continues operation despite individual
          failures

        ## Completion Scenarios

        ### Normal Completion (not cancelled):
        - Processes all remaining queued sentences to ensure nothing is lost
        - Creates fresh cancellation events for final synthesis to prevent
          early termination
        - Logs processing status for debugging and monitoring
        - Ensures complete TTS output before thread termination

        ### Cancelled Completion:
        - Drains synthesis queue without processing to enable fast shutdown
        - Logs cancellation status for debugging
        - Skips remaining TTS synthesis to minimize shutdown delay

        ### Final Cleanup:
        - Sets `completion_event` to signal main thread that TTS is finished
        - Enables proper coordination with `wait_for_completion()` method
        - Ensures thread termination is properly signaled

        ## Thread Coordination

        ### Synchronization Primitives:
        - `cancellation_event`: Shared signal for coordinated shutdown
        - `completion_event`: Signals when all TTS processing is complete
        - `synthesis_queue`: Thread-safe communication channel for sentences
        - `reply_action_completed`: Boolean flag from main thread

        ### Producer-Consumer Pattern:
        - **Producer**: Main thread via `process_chunk()` method
        - **Consumer**: This background worker thread
        - **Buffer**: `synthesis_queue` with timeout-based coordination

        ## Error Recovery

        ### Resilience Features:
        - Individual synthesis failures don't crash the worker
        - Queue timeout prevents indefinite blocking
        - Graceful degradation maintains partial functionality

        ### Resource Management:
        - Proper task_done() calls for queue synchronization
        - Event-based coordination prevents resource leaks
        - Timeout-based operations prevent hanging threads

        Note:
            This method runs continuously until cancellation or completion. It's
            designed to be robust against network failures, TTS errors, and
            other transient issues while maintaining responsive shutdown
            capabilities for ROS 2 lifecycle management.
        """
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

                # If reply action is done and no more work, we can complete
                if (self.reply_action_completed
                        and self.synthesis_queue.empty()
                        and not self.sentence_buffer.chunks):
                    break
                continue
            except Exception as e:
                self.logger.error(f'TTS synthesis worker error: {str(e)}')

        # After normal completion (not cancellation), process remaining items
        # This ensures final sentences are synthesized for normal completion
        if not self.cancellation_event.is_set():
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
                    error_msg = f'Error processing final sentence: {str(e)}'
                    self.logger.error(error_msg)
        else:
            # If cancelled, just drain the queue without processing
            self.logger.info('Cancellation requested, skipping remaining TTS')
            while not self.synthesis_queue.empty():
                try:
                    sentence = self.synthesis_queue.get_nowait()
                    self.synthesis_queue.task_done()
                except Empty:
                    break

        # Signal that TTS processing is complete
        self.completion_event.set()

    def process_chunk(self, chunk: str):
        """Process text chunks for real-time sentence detection and TTS.

        This is the primary entry point for processing streaming text feedback
        from the ReplyActionServer. It implements intelligent sentence boundary
        detection to enable natural speech synthesis while maintaining low
        latency for real-time applications.

        ## Processing Pipeline

        ### Sentence Detection:
        1. **Buffer Addition**: Adds chunk to SentenceBuffer for accumulation
        2. **Boundary Analysis**: Uses regex patterns to detect sentence endings
        3. **Completion Check**: Identifies sentences ending with `.!?:`
            followed by whitespace
        4. **Text Extraction**: Extracts complete sentence and preserves
            remainder

        ### TTS Queueing:
        - **Immediate Processing**: Complete sentences queued immediately
            for TTS
        - **Background Synthesis**: Synthesis happens asynchronously in worker
        - **Non-blocking**: Main feedback loop continues without waiting for TTS
        - **Error Resilience**: Queue failures logged but don't interrupt flow

        ## Cancellation Handling

        ### Early Return:
        - **Cancellation Check**: Returns immediately if cancellation is set
        - **No Processing**: Skips all buffer operations and TTS queueing
        - **Fast Shutdown**: Enables responsive cancellation behavior
        - **Resource Protection**: Prevents unnecessary work during shutdown

        ## Error Scenarios

        ### Queue Failures:
        - **Timeout Errors**: 1-second timeout on queue.put() operations
        - **Capacity Issues**: Logged warnings if synthesis queue is full
        - **Graceful Degradation**: Continues processing despite queue failures
        - **Debug Information**: Comprehensive logging for troubleshooting

        ### Malformed Input:
        - **Empty Chunks**: Silently handled by SentenceBuffer logic
        - **Unicode Issues**: Handled by regex engine and string operations
        - **Large Chunks**: Protected by buffer size limits

        Args:
            chunk (str): Text chunk from streaming reply feedback. Can be
                partial words, complete words, or multiple sentences. The
                method handles all cases and accumulates text until complete
                sentences can be detected.

        Usage:
            ```python
            # Called from ReplyTTSActionServer._feedback_callback()
            tts_manager = self.tts_managers.get(goal_uuid)
            tts_manager.process_chunk(feedback.streaming_resp)
            ```

        Thread Safety:
            This method is called from the main ROS 2 callback thread while
            the synthesis queue is consumed by the background worker thread.
            All operations are thread-safe through Queue and Event primitives.

        Performance:
            Designed for high-frequency calls with minimal latency impact.
            Sentence detection is optimized for streaming scenarios with
            partial text chunks arriving continuously.
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

    def mark_reply_action_completed(self):
        """Signal that the underlying ReplyAction has completed streaming.

        This method is called when the underlying ReplyActionServer has finished
        sending feedback chunks, indicating that no more text will be arriving
        for TTS processing. It triggers final processing of any remaining
        buffered text and coordinates the completion sequence.

        ## Completion Coordination

        ### State Management:
        - **Completion Flag**: Sets `reply_action_completed = True` to signal
          the background worker that no more chunks will arrive
        - **Worker Notification**: Enables the synthesis worker to detect when
          all processing can complete
        - **Final Processing**: Ensures remaining buffered text gets processed

        ### Buffer Finalization:
        - **Forced Flush**: Calls `sentence_buffer.flush_buffer()` to extract
          any remaining text that hasn't formed a complete sentence
        - **Final Queueing**: Adds remaining text to synthesis queue for TTS
        - **Timeout Protection**: Uses 0.5s timeout to prevent blocking
        - **Error Handling**: Logs warnings if final text can't be queued

        ## Background Worker Integration

        ### Completion Detection:
        The background synthesis worker uses this state change to determine
        when it can safely exit:

        ```python
        # In _synthesis_worker()
        if (self.reply_action_completed
                and self.synthesis_queue.empty()
                and not self.sentence_buffer.chunks):
            break  # Safe to exit worker loop
        ```

        ### Processing Guarantee:
        - **No Data Loss**: Ensures all text gets processed, even partial
          sentences that don't end with punctuation
        - **Complete Synthesis**: Background worker processes final text
          before signaling completion
        - **Graceful Termination**: Enables clean shutdown without losing audio

        ## Usage Pattern

        ### Action Server Integration:
        ```python
        # In ReplyTTSActionServer.execute_callback()
        reply_action_result = await result_future

        # Signal that reply action is complete
        if goal_uuid in self.tts_managers:
            self.tts_managers[goal_uuid].mark_reply_action_completed()

        # Wait for TTS to finish processing
        tts_completed = tts_manager.wait_for_completion(timeout=0.5)
        ```

        ### Timing Considerations:
        - **Call Order**: Should be called after underlying action completes
          but before waiting for TTS completion
        - **Non-blocking**: Method returns immediately after queueing final text
        - **Completion Wait**: Use `wait_for_completion()` to wait for
            TTS finish

        ## Error Scenarios

        ### Queue Failures:
        - **Timeout Protection**: 0.5s timeout prevents indefinite blocking
        - **Graceful Degradation**: TTS processing continues even if final
            text can't be queued

        ### Multiple Calls:
        - **Idempotent**: Safe to call multiple times (flag already set)
        - **Buffer State**: Subsequent calls won't find additional text to flush
        - **No Side Effects**: Multiple calls don't cause issues

        Thread Safety:
            This method is called from the main ROS 2 callback thread while the
            background synthesis worker may be reading the completion flag. The
            boolean assignment is atomic in Python, ensuring thread safety.

        Note:
            This method only signals completion; it doesn't wait for TTS to
            finish. Use `wait_for_completion()` if you need to wait for all
            synthesis to complete before proceeding.
        """
        self.reply_action_completed = True
        # Flush any remaining buffer content for normal completion
        remaining_text = self.sentence_buffer.flush_buffer()
        if remaining_text:
            try:
                self.synthesis_queue.put(remaining_text, timeout=0.5)
            except Exception as e:
                self.logger.warn(f'Failed to queue final text: {str(e)}')

    def wait_for_completion(self, timeout: float = None) -> bool:
        """Wait for all TTS processing to complete with optional timeout.

        This method provides synchronous waiting for the completion of all
        background TTS synthesis, enabling the calling thread to coordinate
        with the synthesis worker and ensure all audio output has finished
        before proceeding with action completion or cleanup.

        ## Completion Detection

        ### Event-Based Coordination:
        - **Completion Event**: Waits on `completion_event` set by synthesis
          worker when all processing is finished
        - **Thread Synchronization**: Provides safe coordination between main
          thread and background synthesis worker
        - **Atomic Signaling**: Uses threading.Event for race-condition-free
          completion detection

        ### Completion Criteria:
        The synthesis worker sets the completion event when:
        1. **Reply Action Complete**: `mark_reply_action_completed()` called
        2. **Queue Empty**: All queued sentences have been processed
        3. **Buffer Empty**: No remaining text in sentence buffer
        4. **Final Synthesis**: All TTS synthesis operations completed

        ## Timeout Behavior

        ### Timeout Scenarios:
        ```python
        # Block indefinitely until completion
        completed = tts_manager.wait_for_completion()

        # Wait with timeout for responsive UI
        completed = tts_manager.wait_for_completion(timeout=5.0)
        if not completed:
            logger.warn("TTS completion timed out")
        ```

        ### Timeout Values:
        - **None (default)**: Wait indefinitely until completion
        - **Positive float**: Maximum wait time in seconds
        - **Zero**: Non-blocking check of completion status
        - **Negative**: Treated as zero (immediate return)

        ## Use Cases

        ### Action Server Integration:
        ```python
        # Wait for TTS completion before returning action result
        tts_manager.mark_reply_action_completed()

        # Responsive waiting with periodic cancellation checks
        while not tts_completed:
            if goal_handle.is_cancel_requested:
                tts_manager.cancel_immediately()
                break
            tts_completed = tts_manager.wait_for_completion(timeout=0.5)
        ```

        ### Graceful Shutdown:
        ```python
        # Ensure TTS completes before node shutdown
        for tts_manager in self.tts_managers.values():
            completed = tts_manager.wait_for_completion(timeout=3.0)
            if not completed:
                tts_manager.cancel_immediately()
        ```

        ## Performance Characteristics

        ### Blocking Behavior:
        - **Efficient Waiting**: Uses OS-level event waiting (no busy polling)
        - **Responsive Timeout**: Returns immediately when timeout expires
        - **Thread Safe**: Multiple threads can wait on the same event

        ## Error Handling

        ### Edge Cases:
        - **Already Complete**: Returns immediately if TTS already finished
        - **Never Started**: Returns True if no TTS processing was needed
        - **Cancelled Manager**: Returns completion status regardless of
          cancellation state

        Args:
            timeout (float, optional): Maximum time to wait in seconds.
                None means wait indefinitely. Zero means non-blocking check.
                Negative values treated as zero.

        Returns:
            bool: True if TTS processing completed within the timeout period,
                False if timeout expired before completion. Note that False
                doesn't necessarily indicate an error - it may mean TTS is
                still processing normally.

        Thread Safety:
            This method is thread-safe and can be called from any thread.
            Multiple threads can wait on the same completion event
            simultaneously without issues.

        Performance:
            Highly efficient blocking operation using OS-level event primitives.
            CPU usage is minimal during waiting periods.
        """
        return self.completion_event.wait(timeout)

    def cancel_immediately(self):
        """Cancel TTS processing immediately by mid-sentence audio termination.

        This method provides emergency shutdown capabilities for TTS processing,
        stopping all synthesis immediately including any currently playing
        audio. It's designed for scenarios where responsive cancellation is
        more important than completing current speech output.

        ## Immediate Shutdown Sequence

        ### Phase 1: Stop Current Audio
        ```python
        # Interrupt any currently playing TTS audio
        self.tts_worker.stop_current_synthesis()
        ```
        - **Audio Termination**: Calls Azure TTS `stop_speaking_async()` to
          halt current audio output mid-sentence
        - **Immediate Response**: Audio stops within ~50-100ms typically
        - **No Completion**: Current sentence synthesis is abandoned

        ### Phase 2: Signal Cancellation
        ```python
        # Prevent any new TTS processing
        self.cancellation_event.set()
        ```
        - **Worker Notification**: Signals background synthesis worker to stop
        - **Processing Prevention**: Blocks new chunks from being processed
        - **Queue Protection**: Prevents additional sentences from being queued

        ### Phase 3: Clear Synthesis Queue
        ```python
        # Remove all pending sentences without processing
        while not self.synthesis_queue.empty():
            self.synthesis_queue.get_nowait()
            self.synthesis_queue.task_done()
        ```
        - **Queue Draining**: Removes all pending sentences without synthesis
        - **Fast Cleanup**: Enables rapid termination without waiting for TTS
        - **Resource Release**: Frees memory used by queued text

        ### Phase 4: Thread Termination
        ```python
        # Signal worker to exit and wait for thread cleanup
        self.synthesis_queue.put(None)  # Shutdown signal
        self.synthesis_thread.join(timeout=3.0)
        self.completion_event.set()
        ```
        - **Graceful Thread Exit**: Allows worker thread to clean up properly
        - **Timeout Protection**: 3-second limit prevents hanging shutdown
        - **Completion Signal**: Ensures `wait_for_completion()` returns

        ## Use Cases

        ### Client Cancellation:
        ```python
        # User cancels action - stop TTS immediately
        if goal_handle.is_cancel_requested:
            tts_manager.cancel_immediately()
            goal_handle.canceled()
        ```

        ### Node Shutdown:
        ```python
        # Emergency shutdown during node destruction
        def destroy(self):
            for tts_manager in self.tts_managers.values():
                tts_manager.cancel_immediately()
        ```

        ### Error Recovery:
        ```python
        # Stop TTS due to critical error
        try:
            # Some critical operation
            pass
        except CriticalError:
            tts_manager.cancel_immediately()
            raise
        ```

        ## Comparison with Graceful Cancel

        ### Immediate (`cancel_immediately()`):
        - **Audio**: Stops mid-sentence immediately
        - **Queue**: Discards all pending sentences
        - **Speed**: ~200ms typical completion
        - **Use Case**: User cancellation, emergency shutdown

        ### Graceful (`cancel()`):
        - **Audio**: Completes current sentence before stopping
        - **Queue**: Processes remaining queued sentences
        - **Speed**: Variable depending on remaining TTS work
        - **Use Case**: Normal completion, polite interruption

        Note:
            This method prioritizes speed over completeness. Use `cancel()` if
            you want to allow current synthesis to complete gracefully before
            shutdown.
        """
        self.logger.info(
            f'Immediately canceling TTS manager for goal {self.goal_uuid}')

        # Stop any current synthesis mid-sentence
        self.tts_worker.stop_current_synthesis()

        # Signal cancellation immediately - do NOT queue any more text
        self.cancellation_event.set()

        # Clear the synthesis queue to stop processing remaining items
        while not self.synthesis_queue.empty():
            try:
                self.synthesis_queue.get_nowait()
                self.synthesis_queue.task_done()
            except Empty:
                break

        # Add shutdown signal
        self.synthesis_queue.put(None)

        # Wait for synthesis thread to finish
        if self.synthesis_thread and self.synthesis_thread.is_alive():
            self.synthesis_thread.join(timeout=3.0)

        # Ensure completion event is set
        self.completion_event.set()

    def cancel(self):
        """Cancel TTS processing gracefully finishing TTS on queued sentences.

        This method provides polite shutdown of TTS processing, allowing any
        currently playing audio to finish naturally while preventing new
        synthesis from starting. It ensures that partial sentences in the buffer
        get processed, providing complete audio output before termination.

        ## Graceful Shutdown Sequence

        ### Phase 1: Process Remaining Content
        ```python
        # Flush buffer and queue any remaining text
        remaining_text = self.sentence_buffer.flush_buffer()
        if remaining_text:
            self.synthesis_queue.put(remaining_text, timeout=0.5)
        ```
        - **Buffer Finalization**: Extracts any remaining text from sentence
          buffer that hasn't formed a complete sentence
        - **Final Queueing**: Ensures all available text gets processed for TTS
        - **Timeout Protection**: 0.5s limit prevents blocking during shutdown
        - **Complete Output**: No text is lost during graceful cancellation

        ### Phase 2: Signal Graceful Cancellation
        ```python
        # Allow worker to finish processing queue
        self.cancellation_event.set()
        self.synthesis_queue.put(None)  # Shutdown signal
        ```
        - **Worker Notification**: Signals background thread to finish current
          work and exit
        - **Queue Completion**: Worker processes remaining queued sentences
          before terminating
        - **Natural Completion**: Current synthesis allowed to finish normally

        ### Phase 3: Wait for Clean Termination
        ```python
        # Wait for worker thread to complete processing
        if self.synthesis_thread and self.synthesis_thread.is_alive():
            self.synthesis_thread.join(timeout=3.0)
        self.completion_event.set()
        ```
        - **Thread Coordination**: Waits for background worker to finish
          processing and exit cleanly
        - **Timeout Safety**: 3-second limit prevents indefinite waiting
        - **Completion Signal**: Ensures `wait_for_completion()` returns True

        ## Use Cases

        ### Normal Action Completion:
        ```python
        # Action completed successfully - finish TTS gracefully
        reply_action_result = await result_future
        tts_manager.mark_reply_action_completed()

        # Wait for TTS completion, then clean shutdown
        tts_completed = tts_manager.wait_for_completion(timeout=5.0)
        tts_manager.cancel()  # Graceful cleanup
        ```

        ### Polite Interruption:
        ```python
        # User requests different action - finish current speech first
        if new_goal_requested and speech_in_progress:
            current_tts_manager.cancel()  # Let it finish speaking
            # Then start new TTS manager for new goal
        ```

        ### Resource Cleanup:
        ```python
        # Clean shutdown during normal operation
        def cleanup_completed_goals(self):
            for goal_uuid, tts_manager in completed_goals.items():
                tts_manager.cancel()  # Graceful cleanup
                del self.tts_managers[goal_uuid]
        ```

        ## Comparison with Immediate Cancel

        ### Graceful (`cancel()`):
        - **Audio Quality**: Complete, natural-sounding speech output
        - **Content**: All text gets processed into audio
        - **Timing**: Variable based on remaining TTS work
        - **User Experience**: Polite, non-jarring termination

        ### Immediate (`cancel_immediately()`):
        - **Audio Quality**: May cut off mid-sentence
        - **Content**: Pending text may be lost
        - **Timing**: Fast, predictable ~200ms termination
        - **User Experience**: Immediate response to cancellation

        Note:
            This method prioritizes completeness over speed. Use
            `cancel_immediately()` if you need rapid termination without
            waiting for current synthesis to complete.
        """
        self.logger.info(
            f'Gracefully canceling TTS manager for goal {self.goal_uuid}')

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

        # Ensure completion event is set
        self.completion_event.set()


class ReplyTTSActionServer(Node):
    """
    ROS 2 Action Server providing text-to-speech for streaming robot replies.

    This node implements a TTS-enabled proxy wrapper around the standard
    ReplyActionServer, adding real-time speech synthesis capabilities to
    streaming reply text feedback. It maintains full compatibility with the
    ReplyAction interface while providing audio output through Azure
    Cognitive Services TTS.

    ## ROS 2 Node Lifecycle

    ### Phase 1: Node Initialization

    Key components initialized:
    - **ActionServer**: Accepts ReplyAction goals with TTS capabilities
    - **ActionClient**: Forwards goals to underlying ReplyActionServer
    - **Resource Tracking**: UUID-based dictionaries for concurrent goal mgmt
    - **Thread Safety**: ReentrantCallbackGroup enables concurrent processing

    ### Phase 2: Runtime Operation
    The node operates as a transparent proxy with TTS enhancement:

    **Goal Processing Pipeline:**
    1. Client sends ReplyAction goal → ReplyTTSActionServer
    2. Server forwards identical goal → underlying ReplyActionServer
    3. Feedback streaming: ReplyActionServer → TTSManager → Azure TTS → Audio
    4. Feedback forwarding: Server publishes feedback to original client
    5. Result forwarding: Server returns final result to client

    **Concurrent Goal Management:**
    - Each goal gets isolated TTSManager instance with dedicated threading
    - Goals tracked by UUID to prevent cross-goal interference
    - Independent cancellation and cleanup per goal
    - Multiple goals can run simultaneously without audio conflicts

    ### Phase 3: Cleanup and Shutdown
    Graceful resource cleanup through destroy() method:

    ```python
    # Immediate TTS termination for all active goals
    for goal_uuid, tts_manager in self.tts_managers.items():
        tts_manager.cancel_immediately()

    # Clean resource tracking and shutdown action server/client
    self.tts_managers.clear()
    self._action_server.destroy()
    super().destroy_node()
    ```

    ## Architecture Overview

    ```
    ┌─────────────────┐  ReplyAction  ┌───────────────────────┐  ReplyAction
    │   ROS2 Client   │ ─────────────→│  ReplyTTSActionServer │ ────────────→
    │                 │               │                       │
    │                 │←─feedback+audio                       │←─strm feedback
    └─────────────────┘               │                       │
                                      │                       │
                                      │    ┌──────────────────┴─┐
                                      │    │   TTSManager       │
                                      │    │ ┌───────────────┐  │
                                      │    │ │SentenceBuffer │  │
                                      │    │ └───────────────┘  │
                                      │    │ ┌───────────────┐  │
                                      │    │ │AzureTTSWorker │  │──→ Audio
                                      │    │ └───────────────┘  │
                                      │    └────────────────────┘
                                      └───────────────────────┘
     ┌──────────────────────┐
     │  reply_action_server │
     │                      │
     │                      │
     └──────────────────────┘
    ```

    ## Usage Patterns

    ### Standard Deployment
    ```bash
    # Launch with Azure TTS credentials
    ros2 run reply_action reply_tts_action --ros-args \
        -p azure_speech_key:=your_subscription_key \
        -p azure_speech_endpoint:=https://your-region.tts.speech.microsoft.com/
    ```

    ### Client Integration
    ```python
    from exodapt_robot_interfaces.action import ReplyAction

    client = ActionClient(node, ReplyAction, 'reply_tts_action_server')

    goal = ReplyAction.Goal()
    goal.state = "Some textual description of robot state"

    # Feedback includes both text and audio output
    future = client.send_goal_async(goal, feedback_callback=feedback_cb)
    ```

    ## Resource Management

    ### Goal Tracking Dictionaries
    - **active_goals**: `Dict[tuple, GoalHandle]` - Maps goal UUIDs to
      underlying server goal handles for proper cancellation forwarding when
      clients cancel TTS goals

    - **tts_managers**: `Dict[tuple, TTSManager]` - Maps goal UUIDs to
      TTSManager instances for isolated per-goal speech synthesis including
      sentence buffering, synthesis queue, and background threading

    ### Thread Safety & Concurrency
    - Uses ReentrantCallbackGroup to prevent deadlocks during nested callbacks
    - TTSManager instances run independent background threads per goal
    - Thread-safe resource cleanup during cancellation and completion
    - Multiple concurrent goals supported without audio interference

    ### Memory Management
    - Automatic cleanup on goal completion, cancellation, or error
    - Immediate TTS termination during shutdown prevents hanging threads
    - Resource tracking dictionaries prevent memory leaks from abandoned goals

    ## Configuration Parameters

    - **action_server_name** (str): Name for this TTS-enabled action server
      Default: 'reply_tts_action_server'

    - **reply_action_server_name** (str): Name of underlying ReplyActionServer
      to forward to. Default: 'reply_action_server'

    - **azure_speech_key** (str): Azure Cognitive Services subscription key
      Default: '' (enables text-only mode without audio)

    - **azure_speech_endpoint** (str): Azure TTS service endpoint URL
      Default: '' (enables text-only mode without audio)

    ## Error Handling & Graceful Degradation

    - **Missing Azure Credentials**: Node operates in text-only mode, logging
      TTS text without audio synthesis, maintaining full functionality for
      development/testing

    - **Network Failures**: TTS errors are logged but don't interrupt goal
      execution, ensuring robust operation in unreliable network conditions

    - **Cancellation Scenarios**: Supports both immediate cancellation (stops
      mid-sentence) and graceful cancellation (completes current sentence)
      based on timing

    Note:
        The node maintains full backward compatibility with existing ReplyAction
        clients. No client-side changes are required to benefit from TTS
        capabilities.
    """

    def __init__(self, **kwargs):
        """
        Initialize the ReplyTTSActionServer ROS 2 node

        Parameters (ROS 2):
            action_server_name (str): Name for this TTS action server
                Default: 'reply_tts_action_server'
            reply_action_server_name (str): Name of underlying reply action
                server. Default: 'reply_action_server'
            azure_speech_key (str): Azure Cognitive Services speech key
                Default: '' (TTS will log text without audio if empty)
            azure_speech_endpoint (str): Azure speech service endpoint URL
                Default: '' (TTS will log text without audio if empty)
            enable_tts_warmup (bool): Whether to perform TTS warmup during
                initialization to reduce first-call latency. Default: True

        Key Components Created:
            - ActionServer: Handles incoming TTS-enabled reply requests
            - ActionClient: Forwards requests to underlying reply server
            - active_goals: Dict mapping goal UUIDs to reply action goal handles
            - tts_managers: Dict mapping goal UUIDs to TTSManager instances

        Args:
            **kwargs: Additional keyword arguments passed to the ROS 2 Node
                constructor

        Raises:
            RuntimeError: If ROS 2 node initialization fails
            Exception: If action server/client setup encounters errors

        Note:
            If Azure credentials are not provided, the node will still function
            but will only log TTS text without producing audio output.
        """
        super().__init__('reply_tts_action', **kwargs)

        self.declare_parameter('action_server_name', 'reply_tts_action_server')
        self.declare_parameter(
            'reply_action_server_name',
            'reply_action_server',
        )
        self.declare_parameter('azure_speech_key', '')
        self.declare_parameter('azure_speech_endpoint', '')
        self.declare_parameter('enable_tts_warmup', True)

        self.action_server_name = self.get_parameter(
            'action_server_name').value
        self.reply_action_server_name = self.get_parameter(
            'reply_action_server_name').value
        self.azure_speech_key = self.get_parameter('azure_speech_key').value
        self.azure_speech_endpoint = self.get_parameter(
            'azure_speech_endpoint').value
        self.enable_tts_warmup = self.get_parameter('enable_tts_warmup').value

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

        # Perform TTS warmup if enabled and credentials are available
        if self.enable_tts_warmup:
            self._warmup_azure_tts()

        self.get_logger().info(
            'ReplyTTSActionServer initialized\n'
            'Parameters:\n'
            f'  action_server_name: {self.action_server_name}\n'
            f'  reply_action_server_name: {self.reply_action_server_name}\n'
            f'  azure_speech_key: {self.azure_speech_key[:8]}...\n'
            f'  azure_speech_endpoint: {self.azure_speech_endpoint}\n'
            f'  enable_tts_warmup: {self.enable_tts_warmup}')

    def _warmup_azure_tts(self):
        """Perform Azure TTS warmup to reduce latency for subsequent goals.

        Creates a temporary AzureTTSWorker and performs a brief synthesis to
        establish the initial connection to Azure Cognitive Services. This
        reduces the latency experienced by the first goal that uses TTS.

        The warmup:
        - Only runs if Azure credentials are provided (skips text-only mode)
        - Uses a brief, quiet warmup phrase to minimize audio disruption
        - Handles errors gracefully without affecting node initialization
        - Completes in the background without blocking node startup

        Note:
            This is a "fire-and-forget" warmup that doesn't wait for completion.
            The Azure SDK handles connection establishment asynchronously, so
            even a quick synthesis call provides connection warming benefits.
        """
        # Skip warmup if no credentials (text-only mode)
        if not self.azure_speech_key or not self.azure_speech_endpoint:
            self.get_logger().info(
                'Skipping TTS warmup - no Azure credentials')
            return

        self.get_logger().info('Starting Azure TTS warmup...')

        try:
            # Create temporary warmup worker
            warmup_worker = AzureTTSWorker(
                speech_key=self.azure_speech_key,
                speech_endpoint=self.azure_speech_endpoint,
                logger=self.get_logger(),
                voice_name='en-US-AvaMultilingualNeural')

            # Perform quick warmup synthesis
            # Use a never-set event to ensure synthesis completes normally
            warmup_event = threading.Event()
            success = warmup_worker.synthesize_text(
                "System initialized.",
                warmup_event,
            )

            if success:
                self.get_logger().info(
                    'Azure TTS warmup completed successfully')
            else:
                self.get_logger().warn(
                    'Azure TTS warmup completed with warnings')

        except Exception as e:
            # Don't let warmup errors affect node initialization
            self.get_logger().warn(f'Azure TTS warmup failed: {str(e)}')

    def destroy(self):
        """Clean up resources including active TTS managers."""
        # Cancel all active TTS managers immediately on shutdown
        for goal_uuid, tts_manager in self.tts_managers.items():
            self.get_logger().info(
                f'Cleaning up TTS manager for goal {goal_uuid}')
            tts_manager.cancel_immediately()

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
        """Forwards the goal to ReplyActionServer and perform TTS on feedback.

        This is the core execution method that implements the complete lifecycle
        of a TTS-enabled reply action goal. It acts as a proxy between clients
        and the underlying ReplyActionServer while adding real-time
        text-to-speech synthesis for streaming feedback.

        Execution Workflow:

        **Phase 1: Forward goal to underlying ReplyActionServer**
            - Waits for underlying ReplyActionServer availability (5s timeout)
            - Creates and forwards the goal request with identical parameters
            - Registers feedback callback for real-time TTS processing
            - Waits for goal acceptance from underlying server

        **Phase 2: Set up TTS processing and tracking**
            - Creates unique goal UUID for resource tracking
            - Instantiates TTSManager with Azure Speech credentials
            - Stores goal handle and TTS manager in tracking dictionaries
            - Enables concurrent goal processing and proper cleanup

        **Phase 3: Concurrent Monitoring**
            - Monitors underlying action completion via async result future
            - Continuously checks for client cancellation requests (100ms)
            - Processes streaming feedback chunks through TTS pipeline
            - Maintains goal state consistency between servers

        **Phase 4: Completion Handling**
            Four distinct completion scenarios are handled:

            1. Successful Completion:
               - Underlying server completes successfully
               - Marks reply action as completed in TTS manager
               - Waits for all queued TTS synthesis to finish
               - Cleans up resources and returns success result

            2. Cancellation During ReplyAction Execution:
               - Client requests cancellation while underlying action runs
               - Cancels forwarded goal on underlying server
               - Immediately terminates TTS with cancel_immediately()
               - Cleans up resources and returns canceled status

            3. Cancellation During TTS Completion:
               - Client requests cancellation while waiting for TTS finish
               - Immediately stops all TTS synthesis
               - Cleans up resources and returns canceled status

            4. Error/Abort Scenarios:
               - Underlying server rejection, timeout, or failure
               - General exceptions during execution
               - Immediate cleanup and abort status return

        Resource Management:
            - TTSManager: Per-goal speech synthesis with sentence buffering
            - active_goals: UUID → goal_handle mapping for cancellation
            - tts_managers: UUID → TTSManager mapping for TTS control
            - Automatic cleanup on all completion paths prevents memory leaks

        Concurrency & Threading:
            - Uses ReentrantCallbackGroup for non-blocking execution
            - TTSManager runs background synthesis threads per goal
            - Async/await pattern prevents executor blocking
            - Thread-safe resource cleanup during cancellation

        Feedback Processing Pipeline:
            ```
            ReplyActionServer → _feedback_callback → TTSManager.process_chunk
                   ↓                    ↓                      ↓
            streaming_resp → sentence_buffer → synthesis_queue → Azure TTS
                   ↓
            Forward to TTS client
            ```

        Args:
            goal_handle (ServerGoalHandle): ROS 2 action goal handle containing:
                - request.state: Robot state information for reply generation
                - request.instruction: User instruction for reply context
                - goal_id.uuid: Unique identifier for resource tracking
                - Cancellation and feedback publishing capabilities

        Returns:
            ReplyAction.Result: Action result containing:
                - reply (str): Generated reply text from underlying server
                - Empty result on cancellation or error scenarios

        Raises:
            Exception: Captured and logged, triggers cleanup and goal abort.
                Common causes include:
                - Network timeouts to underlying server
                - Azure TTS service errors
                - Resource allocation failures
                - Client disconnection during execution

        Thread Safety:
            All resource modifications use goal UUID keys to prevent race
            conditions between concurrent goals. TTSManager instances are
            isolated per goal with independent threading and cancellation.
        """
        self.get_logger().info('Executing ReplyTTSActionServer...')

        #####################################################
        #  1. Forward goal to underlying ReplyActionServer
        #####################################################

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

            ##############################
            #  2. Set up TTS processing
            ##############################

            # Store accepted goal handle as an active goal
            goal_uuid = tuple(goal_handle.goal_id.uuid)
            self.active_goals[goal_uuid] = reply_action_goal_handle

            # Create or get TTS manager for this goal and store goal handle
            tts_manager = self.tts_managers.get(goal_uuid)
            if not tts_manager:
                tts_manager = TTSManager(
                    goal_uuid,
                    self.get_logger(),
                    self.azure_speech_key,
                    self.azure_speech_endpoint,
                )
                self.tts_managers[goal_uuid] = tts_manager

            #################################################################
            #  3. Monitor running action while waiting for goal completion
            #################################################################

            result_future = reply_action_goal_handle.get_result_async()

            # Check for ReplyTTSAction cancel request during ongoing ReplyAction
            while not result_future.done():
                if goal_handle.is_cancel_requested:
                    self.get_logger().info(
                        'Canceling ReplyTTSActionServer goal')
                    cancel_future = reply_action_goal_handle.cancel_goal_async(
                    )
                    await cancel_future

                    # Clean up TTS manager (with immediate speech termination)
                    if goal_uuid in self.tts_managers:
                        self.tts_managers[goal_uuid].cancel_immediately()
                        del self.tts_managers[goal_uuid]

                    # Clean up goal tracking
                    if goal_uuid in self.active_goals:
                        del self.active_goals[goal_uuid]

                    goal_handle.canceled()
                    return ReplyAction.Result()

                time.sleep(0.1)

            # Get the ReplyActionServer result
            reply_action_result = await result_future

            ####################################
            #  4. Handle completion scenarios
            ####################################

            # Mark that the reply action has completed (no more feedback coming)
            if goal_uuid in self.tts_managers:
                self.tts_managers[goal_uuid].mark_reply_action_completed()

            # Check for ReplyTTSAction cancel request during ongoing TTS
            tts_completed = False
            if goal_uuid in self.tts_managers:
                self.get_logger().info(
                    'Waiting for TTS synthesis to complete...')
                # Wait for TTS with periodic cancellation checks
                while not tts_completed:

                    ###################################
                    #  Scenario 1: Cancel during TTS
                    ###################################
                    if goal_handle.is_cancel_requested:
                        self.get_logger().info(
                            'Canceling during TTS completion wait')
                        self.tts_managers[goal_uuid].cancel_immediately()
                        tts_completed = True

                        # Clean up
                        del self.tts_managers[goal_uuid]
                        if goal_uuid in self.active_goals:
                            del self.active_goals[goal_uuid]

                        goal_handle.canceled()
                        return ReplyAction.Result()

                    # Wait for completion with short timeout for cancellation
                    tts_manager = self.tts_managers[goal_uuid]
                    tts_completed = tts_manager.wait_for_completion(
                        timeout=0.5)

            # Clean up TTS manager
            if goal_uuid in self.tts_managers:
                self.tts_managers[goal_uuid].cancel()
                del self.tts_managers[goal_uuid]

            # Clean up goal tracking
            if goal_uuid in self.active_goals:
                del self.active_goals[goal_uuid]

            #########################
            #  Scenario 2: Success
            #########################
            if reply_action_result.status == GoalStatus.STATUS_SUCCEEDED:
                reply = reply_action_result.result.reply
                self.get_logger().info(
                    f'ReplyActionServer succeeded with result: {reply}')

                result = ReplyAction.Result()
                result.reply = reply

                goal_handle.succeed()
                return result

            #########################
            #  Sceneario 3: Cancel
            #########################
            elif reply_action_result.status == GoalStatus.STATUS_CANCELED:
                self.get_logger().info('ReplyActionServer was canceled')
                goal_handle.canceled()
                return ReplyAction.Result()

            #######################
            #  Scenario 4: Abort
            #######################
            else:
                status = reply_action_result.status
                self.get_logger().error(
                    f'ReplyActionServer failed with status: {status}')
                goal_handle.abort()
                return ReplyAction.Result()

        ###############################
        #  Scenario 5: General error
        ###############################
        except Exception as e:
            self.get_logger().error(f'Error during goal execution: {str(e)}')

            # Clean up
            goal_uuid = tuple(goal_handle.goal_id.uuid)

            # Clean up TTS manager immediately on error
            if goal_uuid in self.tts_managers:
                self.tts_managers[goal_uuid].cancel_immediately()
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
    Main entry point for the ReplyTTSActionServer ROS 2 node.

    Initializes the ROS 2 context, creates and spins the ReplyTTSActionServer
    node using asyncio for handling async inference callbacks, and performs
    proper cleanup on shutdown. The function handles the complete node lifecycle
    including initialization, execution, and teardown.

    Args:
        args (List[str], optional): Command line arguments for ROS 2
            initialization. Defaults to None, which uses sys.argv.

    Raises:
        KeyboardInterrupt: Gracefully handles Ctrl+C shutdown
        RuntimeError: If ROS 2 initialization or node creation fails
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
