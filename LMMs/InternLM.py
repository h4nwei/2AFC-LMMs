# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import AutoModel, AutoTokenizer

from accelerate import dispatch_model
from LMMs.prompt import get_prompts
from LMMs.utils import auto_configure_device_map, consistency_check_and_summary
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
torch.set_grad_enabled(False)

class InternLM():
    def __init__(self, checkpoint= "internlm/internlm-xcomposer-7b", device=None):
        torch.set_grad_enabled(False)
        model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-xcomposer-7b')
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()

        # self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device).eval()
        device_map = auto_configure_device_map(2)
        self.model = dispatch_model(self.model, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        # tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.model.tokenizer = tokenizer


    def chat(self, imgPath1, imgPath2, prompts):
        response, history = self.model.chat(text=prompts[0], image=imgPath1, history=None)
        response, history = self.model.chat(text=prompts[1], image=imgPath2, history=history)
        response1, history = self.model.chat(text=prompts[2], image=None, history=history)
        # response2, history = self.model.chat(text=prompts[3], image=None, history=history)

        return response1

    def run(self, 
            imgPath1, # image 1 path
            imgPath2, # image 2 path
            consistency_check=True # image 2 path
            ):
        round = 1
        if consistency_check:
            round = 2

        prompts = get_prompts('InternLM')

        answer_consistency_check = []
        answer_bias_check = []

        for round_i in range(round):
            if (round_i + 1) % 2 != 0: # first round
                ans1 = self.chat(imgPath1, imgPath2, prompts)

                if 'A' in ans1:
                    answer_consistency_check.append('img1')
                    answer_bias_check.append('first')
                    result1 = 'first'
                elif 'B' in ans1:
                    answer_consistency_check.append('img2')
                    answer_bias_check.append('second')
                    result1 = 'second'
                else:
                    answer_consistency_check.append('draw')
                    answer_bias_check.append('draw')
                    result1 = 'draw'

            else:
                ans2 = self.chat(imgPath2, imgPath1,prompts)
                # ans = [ans1, ans2]
                
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
    return InternLM()  

if __name__ == '__main__':
  import os
  dataset = 'LIVEC'
  root = '/home/zhw/IQA/code/IQA-PyTorch/datasets/' 
  model = InternLM(device='cuda')    
  image1_path = os.path.join(root, dataset, "7.bmp") 
  image2_path = os.path.join(root, dataset, "10.bmp")  
  print(model.run(image1_path, image2_path, consistency_check=True))
 