import os
import re
from PIL import Image
import base64
import requests
import io
from LMMs.utils import consistency_check_and_summary
from PIL import Image
import io
import base64



def encode_image(image_path):
    # if image_path.lower().endswith('.bmp'):
    with Image.open(image_path) as img:
        rgb_img = img.convert('RGB')
        with io.BytesIO() as byte_stream:
            rgb_img.save(byte_stream, format='JPEG')
            byte_stream.seek(0)
            encoded_string = base64.b64encode(byte_stream.read()).decode('utf-8')
            return encoded_string
    # else:
    #   with open(image_path, "rb") as image_file:
    #       return base64.b64encode(image_file.read()).decode('utf-8')


class ChatGPT():
    def __init__(self, model:str="gpt-4-vision-preview", api_key:str='sk-rwixHZImi3GQHJRJbrigT3BlbkFJ5K6'):
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"}
    
    def chat(self, imgPath1, imgPath2):
      base64_image1 = encode_image(imgPath1)
      base64_image2 = encode_image(imgPath2)

      
      payload = {
          "model": self.model,
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image1}"
                  }
                },
                {
                  "type": "text",
                  "text": "This is the first image"
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image2}"
                  }
                },
                {
                  "type": "text",
                  "text": "This is the second image"
                },
                {
                  "type": "text",
                  "text": "Which image has a better visual quality? A. The first image. B. The second image. Answer with the option's letter from the given choices directly."
                },
              ]
            }
          ],
          "max_tokens": 250
      } 

      response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)

      response_json = response.json()

      messages = response_json.get('choices', [])[0].get('message', {}).get('content', '')

      return messages


    def run(self,
            imgPath1, # image 1 path
            imgPath2, # image 2 path
            consistency_check=True # image 2 path
            ):
      
      round = 1
      if consistency_check:
          round = 2

      answer_consistency_check = []
      answer_bias_check = []

      
      for round_i in range(round):
            if (round_i + 1) % 2 != 0: # first round
                ans1 = self.chat(imgPath1, imgPath2)
                ans1 = [ans]
                # read the answer
                if 'A' in ans1:
                    answer_consistency_check.append('img1')
                    answer_bias_check.append('first')
                    result1 = 'first'
                elif 'second' in ans1:
                    answer_consistency_check.append('img2')
                    answer_bias_check.append('second')
                    result1 = 'second'
                else:
                    answer_consistency_check.append('draw')
                    answer_bias_check.append('draw')
                    result1 = 'draw'
            
            else:
                ans2 = self.chat(imgPath2, imgPath1)
                
                # read the answer
                if 'A' in ans2:
                    answer_consistency_check.append('img2')
                    answer_bias_check.append('first')
                    result2 = 'first'
                elif 'B' in ans2:
                    answer_consistency_check.append('img1')
                    answer_bias_check.append('second')
                    result2 = 'second'
                else:
                    answer_consistency_check.append('draw')
                    answer_bias_check.append('draw')
                    result2 = 'draw'
                ans = [result1, result2]
                
      summary = consistency_check_and_summary(
          consistency_check, 
          answer_consistency_check,
          answer_bias_check,
          ans)
      
      return summary

def make_model():
    return ChatGPT()  

if __name__ == '__main__':
  dataset = 'MM21'
  root = '/home/zhw/IQA/code/IQA-PyTorch/datasets/' 
  model = ChatGPT()    
  image1_path = os.path.join(root, dataset, "source/00316.png") 
  image2_path = os.path.join(root, dataset, "FRICwRNN/00316_4.png")  
  print(model.run(image1_path, image2_path, consistency_check=False))

