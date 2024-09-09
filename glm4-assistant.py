import base64
import os
from datetime import datetime
from threading import Lock, Thread
from time import sleep
from PIL import Image
import io
import cv2
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

import numpy as np

import logging

 
# 设置日志级别为WARNING，INFO级别的日志将不会显示
logging.basicConfig(level=logging.WARNING)


load_dotenv()
 
GLM_API_BASE = os.getenv("GLM_API_BASE")
GLM_API_KEY = os.getenv("GLM_API_KEY")

# 调试：打印环境变量以确认它们是否正确加载
print(f"GLM_API_BASE: {GLM_API_BASE}")
print(f"GLM_API_KEY: {GLM_API_KEY}")

class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True

        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self):
        self.lock.acquire()
        _, frame = self.stream.read()
        #frame = self.frame.copy()
        self.lock.release()
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()
 
class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)
 
    def answer(self, prompt, image):
        if not prompt:
            return
        image,file_name = self._save_image(image)
        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image},
            config={"configurable": {"session_id": "unused"}},
        ).strip()
        print('======'+file_name)
        print("Response:", response)
 
 
    # 保存图片到本地
    def _save_image(self, image):
        if not os.path.exists("./images"):
            os.makedirs("./images")
        file_name = f"./images/image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        # 将图像转换为内存中的字节流
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()
        img_decoded = cv2.imdecode(np.frombuffer(byte_arr, np.uint8), cv2.IMREAD_COLOR)

        params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        cv2.imwrite(file_name, img_decoded, params)

        _, buffer = imencode(".jpeg", image)
        image = base64.b64encode(buffer).decode()
        return image,file_name

    @staticmethod
    def _create_inference_chain(model):
        SYSTEM_PROMPT = """
        你是一个有眼睛的助手，我会发送图片给你，让你看到周围的景象，将使用用户提供的聊天历史和图片来回答其问题。
        不要提到“图片”这个单词，直接描述图片的内容，不要使用emojis，不要问用户问题。
        保持友好的态度。展示一些个性。不要太正式。
        用中文回复
        """
 
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )
        chain = prompt_template | model | StrOutputParser()
 
        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )



webcam_stream = WebcamStream().start()
model = ChatOpenAI(model="glm-4v", base_url=GLM_API_BASE, api_key=GLM_API_KEY)
assistant = Assistant(model)

while True:
      cv2.imshow("webcam", webcam_stream.read())
      key = cv2.waitKey(1)
      if key & 0xFF == ord('q'):
         break
      elif(key & 0xFF == ord(' ')):
         assistant.answer("你看到了什么?",webcam_stream.read()) 

webcam_stream.stop()
cv2.destroyAllWindows()      