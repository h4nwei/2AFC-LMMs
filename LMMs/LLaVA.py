
# import argparse
import torch

# from LMMs_zoo.mPLUG_Owl.mPLUG_Owl2.mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# from LMMs_zoo.mPLUG_Owl.mPLUG_Owl2.mplug_owl2.conversation import conv_templates, SeparatorStyle
# from LMMs_zoo.mPLUG_Owl.mPLUG_Owl2.mplug_owl2.model.builder import load_pretrained_model
# from LMMs_zoo.mPLUG_Owl.mPLUG_Owl2.mplug_owl2.utils import disable_torch_init
# from LMMs_zoo.mPLUG_Owl.mPLUG_Owl2.mplug_owl2.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
# from transformers import TextStreamer
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor, LlavaForConditionalGeneration
import sys
sys.path.append('zhw/IQA/code/NeurIPS24/Q-Align/2AFC')
from LMMs.prompt import get_prompts
from LMMs.utils import consistency_check_and_summary

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image



class LLaVA():
    def __init__(self, checkpoint= "llava-hf/llava-1.5-13b-hf", device=None):
        
        # self.model = AutoModelForCausalLM.from_pretrained(checkpoint, 
        #                                      trust_remote_code=True, 
        #                                      torch_dtype=torch.float16,
        #                                      attn_implementation="eager", 
        #                                      device_map={"":"cuda:0"})
        self.model = LlavaForConditionalGeneration.from_pretrained(
    checkpoint, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_4bit=True
)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def chat(self, image_list):
        images = [Image.open(_) for _ in image_list]
        # inputs = self.processor(self.prompts[0], [images[0], images[1]], return_tensors='pt').to(0, torch.float16)
        inputs = self.processor(self.prompts[0], [images[0], images[1]], return_tensors='pt').to(0,torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        sentence = self.processor.decode(output[0], skip_special_tokens=True)
        # sentence = self.model.tokenizer.batch_decode(self.model.chat(self.prompts[0], [images[0], images[1]], max_new_tokens=200).clamp(0, 100000))[0].split("ASSISTANT:")[-1]
        # sentence = self.model.chat(self.prompts[0], [images[0], images[1]], max_new_tokens=200)

        return sentence

    def run(self, 
        imgPath1, # image 1 path
        imgPath2, # image 2 path
        consistency_check=True # image 2 path
        ):

        image_list = [imgPath1, imgPath2]
        self.prompts = get_prompts('LLaVA')#Co_instruct

        round = 1
        if consistency_check:
            round = 2

        answer_consistency_check = []
        answer_bias_check = []

        for round_i in range(round):
            if (round_i + 1) % 2 != 0: # first round
                ans1 = self.chat(image_list)
                

                # read the answer
                _filter_ans1 = ans1.split('ASSISTANT:')[-1]
                # _filter_ans1 = _filter_ans1.split(',')[0]
                
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
                image_list = [imgPath2, imgPath1]
                ans2 = self.chat(image_list)
                ans = [ans1, ans2]
                
                # read the answer
                _filter_ans2 = ans2.split('ASSISTANT:')[-1]
                # _filter_ans2 = ans2.split('.')[0]
                # _filter_ans2 = _filter_ans2.split(',')[0]
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
    return LLaVA()  

if __name__ == '__main__':
  import os
  dataset = 'LIVEC'
  root = '/home/zhw/IQA/code/IQA-PyTorch/datasets/' 
  model = LLaVA(device='cuda')    
  image1_path = os.path.join(root, dataset, "10.bmp") 
  image2_path = os.path.join(root, dataset, "12.bmp")  
  print(model.run(image1_path, image2_path, consistency_check=True))    

 



