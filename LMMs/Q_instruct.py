import torch
from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model, image_parser, load_images
from LMMs_zoo.utils import consistency_check_and_summary
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
import re

class Q_instruct():
    def __init__(self, model_path="teowu/llava_v1.5_7b_qinstruct_preview_v0.1"):
       self.model_path = model_path
       self.prompt = "The first image is <image-placeholder>, the second image is <image-placeholder>, which image has better visual quality? the first one or the second one?"
    
       disable_torch_init()
       self.model_name = get_model_name_from_path(self.model_path)
       self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
       self.model_path, None, self.model_name)

    def chat(self, imgPath_combine):
        args = type('Args', (), {
            "model_path": self.model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(self.model_path),
            "query": self.prompt,
            "conv_mode": None,
            "image_file": imgPath_combine,
            "sep": ",",
            "temperature": 0.2, # default
            "top_p": None, # default
            "num_beams": 1, # default
            "max_new_tokens": 512 # default
        })()

        qs = args.query
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_files = image_parser(args)
        images = load_images(image_files)
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)
        return outputs
        # return eval_model(args)
    
    def run(self, 
        imgPath1, # image 1 path
        imgPath2, # image 2 path
        consistency_check=True # image 2 path
        ):
        
        image_list = imgPath1 + ',' + imgPath2

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
                    result = 'draw'
            
            else:
                image_list = imgPath2 + ',' + imgPath1
                ans2 = self.chat(image_list)
                ans = [ans1, ans2]
                
                # read the answer
                if 'first' in ans2:
                    answer_consistency_check.append('img2')
                    answer_bias_check.append('first')
                    result2 = 'first'
                elif 'second' in ans2:
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
    return Q_instruct()  

if __name__ == '__main__':
  import os
  
  dataset = 'LIVEC'
  root = '/home/zhw/IQA/code/IQA-PyTorch/datasets/' 
  model = Q_instruct()    
  image1_path = os.path.join(root, dataset, "12.bmp") 
  image2_path = os.path.join(root, dataset, "10.bmp")

  print(model.run(image1_path, image2_path))    