from openai import OpenAI
import os
import base64
from datetime import datetime
from threading import Lock, Thread
from time import sleep
from PIL import Image
import io
import cv2
from cv2 import VideoCapture, imencode
import numpy as np

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
        #_, frame = self.stream.read()
        frame = self.frame.copy()
        self.lock.release()
        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
    
    # 保存图片到本地
    def save(self, image):
        if not os.path.exists("./images"):
            os.makedirs("./images")
        image_path = f"./images/image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        # 将图像转换为内存中的字节流
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()
        img_decoded = cv2.imdecode(np.frombuffer(byte_arr, np.uint8), cv2.IMREAD_COLOR)

        params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        cv2.imwrite(image_path, img_decoded, params)
        return image_path
    
    

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
           return base64.b64encode(image_file.read()).decode("utf-8")

def get_response(image_path):
    # Getting the base64 string
    print("====="+image_path)
    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-max-0809",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "你看到了什么?"},
                    
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        top_p=0.8,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content is not None:
           print(chunk.choices[0].delta.content, end="")


if __name__ == "__main__":
    # Path to your image
    #image_path = "./images/dog_and_girl.jpeg"
    image_path = "./images/image_20240907_171557.png"

    webcam_stream = WebcamStream().start()
    while True:
      cv2.imshow("webcam", webcam_stream.read())
      key = cv2.waitKey(1)
      if key & 0xFF == ord('q'):
         break
      elif(key & 0xFF == ord(' ')):
         image = webcam_stream.read()
         image_path = webcam_stream.save(image)
         get_response(image_path)

    webcam_stream.stop()
    cv2.destroyAllWindows()      


