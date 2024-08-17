# Load via Huggingface Style
import torch
from transformers import AutoTokenizer
from LMMs.mPLUG_Owl.mPLUG_Owl.mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from LMMs.mPLUG_Owl.mPLUG_Owl.mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from LMMs.mPLUG_Owl.mPLUG_Owl.pipeline.interface import do_generate
from LMMs.prompt import get_prompts
from LMMs.utils import consistency_check_and_summary


class mPLUG():
    def __init__(self, checkpoint= "MAGAer13/mplug-owl-llama-7b", device=None):
        self.model =  MplugOwlForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
        image_processor = MplugOwlImageProcessor.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.processor = MplugOwlProcessor(image_processor, self.tokenizer)
    
    def chat(self, image_list):
        sentence = do_generate(self.prompts, image_list, self.model, self.tokenizer, self.processor, 
                       use_bf16=True, max_length=512, top_k=5, do_sample=True)

        return sentence
    
    def run(self, 
        imgPath1, # image 1 path
        imgPath2, # image 2 path
        consistency_check=True # image 2 path
        ):
        
        image_list = [imgPath1, imgPath2]
        self.prompts = get_prompts('mPLUG')

        round = 1
        if consistency_check:
            round = 2

        answer_consistency_check = []
        answer_bias_check = []

        for round_i in range(round):
            if (round_i + 1) % 2 != 0: # first round
                ans1 = self.chat(image_list)
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
                image_list = [imgPath2, imgPath1]
                ans2 = self.chat(image_list)
                # ans = [ans1, ans2]
                
                # read the answer
                _filter_ans2 = ans2.split('.')[0]
                _filter_ans2 = _filter_ans2.split(',')[0]
                if 'first' in _filter_ans2 or '1' in _filter_ans2:
                    answer_consistency_check.append('img2')
                    answer_bias_check.append('first')
                    result2 = 'first'
                elif 'second' in _filter_ans2 or '2' in _filter_ans2:
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
    return mPLUG()  


if __name__ == '__main__':
  import os
  dataset = 'LIVEC'
  root = '/home/zhw/IQA/code/IQA-PyTorch/datasets/' 
  model = mPLUG(device='cuda')    
  image1_path = os.path.join(root, dataset, "10.bmp") 
  image2_path = os.path.join(root, dataset, "12.bmp")  
  print(model.run(image1_path, image2_path, consistency_check=True))    

 

