import asyncio
import time

import rclpy
from exodapt_robot_interfaces.action import ReplyAction
from exodapt_robot_pt import reply_action_pt
from huggingface_hub import InferenceClient
from rclpy.action import ActionServer
from rclpy.node import Node


class ReplyActionServer(Node):

    def __init__(self):
        super().__init__('reply_action')

        self._action_server = ActionServer(
            self,
            ReplyAction,
            'reply_action_server',
            execute_callback=self.execute_callback_tgi,
        )

        self.declare_parameter('tgi_server_url', 'http://localhost:5000')
        self.declare_parameter('max_tokens', 1024)
        self.declare_parameter('llm_temp', 0.6)
        self.declare_parameter('llm_seed', 14)
        self.tgi_server_url = self.get_parameter('tgi_server_url').value
        self.max_tokens = self.get_parameter('max_tokens').value
        self.llm_temp = self.get_parameter('llm_temp').value
        self.llm_seed = self.get_parameter('llm_seed').value

        # TGI inference client
        self.client = InferenceClient(base_url=self.tgi_server_url)

        self.get_logger().info('ReplyActionServer initialized\n'
                               'Parameters:\n'
                               f'  TGI server url: {self.tgi_server_url}\n'
                               f'  max_tokens={self.max_tokens}\n'
                               f'  llm_temp={self.llm_temp}\n'
                               f'  llm_seed={self.llm_seed}')

    async def execute_callback_tgi(self, goal_handle):
        '''
        '''
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
    rclpy.init(args=args)

    node = ReplyActionServer()

    asyncio.run(rclpy.spin(node))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
