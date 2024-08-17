from transformers import IdeficsForVisionText2Text, AutoProcessor
import torch
from PIL import Image
from LMMs.prompt import get_prompts
from LMMs.utils import consistency_check_and_summary


class IDEFICS():
    def __init__(self, checkpoint= "HuggingFaceM4/idefics-9b-instruct", max_length=1500, device=None):
        self.model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.device = device
        self.max_length = max_length
    
    def chat(self, prompt):
        inputs = self.processor(prompt, add_end_of_utterance_token=False, return_tensors="pt").to(self.device)
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        generated_ids = self.model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=self.max_length)
        ans = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        filter_ans = ans[0].split('Assistant: ')[1]

        return filter_ans

    def run(self, 
            imgPath1, # image 1 path
            imgPath2, # image 2 path
            consistency_check=True # image 2 path
            ):
        
        image1 = Image.open(imgPath1).convert("RGB")
        image2 = Image.open(imgPath2).convert("RGB")
        prompts = get_prompts('IDEFICS', image1, image2)

        round = 1
        if consistency_check:
            round = 2

        answer_consistency_check = []
        answer_bias_check = []

        for round_i in range(round):
            if (round_i + 1) % 2 != 0: # first round
                ans1 = self.chat(prompts[0])
                ans = [ans1]
                
                # read the answer
                _filter_ans1 = ans1.split('.')[0]
                _filter_ans1 = _filter_ans1.split(',')[0]
                if 'A' in _filter_ans1:
                    answer_consistency_check.append('img1')
                    answer_bias_check.append('first')
                    result1 = 'first'
                elif 'B' in _filter_ans1:
                    answer_consistency_check.append('img2')
                    answer_bias_check.append('second')
                    result1 = 'second'
                else:
                    answer_consistency_check.append('draw')
                    answer_bias_check.append('draw')
                    result1 = 'draw'
            
            else:
                ans2 = self.chat(prompts[1])
                # ans = [ans1, ans2]
                
                # read the answer
                _filter_ans2 = ans2.split('.')[0]
                _filter_ans2 = _filter_ans2.split(',')[0]
                if 'A' in _filter_ans2:
                    answer_consistency_check.append('img2')
                    answer_bias_check.append('first')
                    result2 = 'first'
                elif 'B' in _filter_ans2:
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
    return IDEFICS()  

if __name__ == '__main__':
  import os
  dataset = 'MM21'
  root = '/home/zhw/IQA/code/IQA-PyTorch/datasets/' 
  model = IDEFICS(device='cuda')    
  image1_path = os.path.join(root, dataset, "source/00316.png") 
  image2_path = os.path.join(root, dataset, "FRICwRNN/00316_4.png")  
  print(model.run(image1_path, image2_path, consistency_check=True))
