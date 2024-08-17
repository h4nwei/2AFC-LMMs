import json
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Dict


def img_pair_generation(save_root, path1, path2, text1, text2, save_name):

    image1 = Image.open(path1).convert("RGB")
    image2 = Image.open(path2).convert("RGB")

    width1, height1 = image1.size
    width2, height2 = image2.size

    new_width = width1 + width2 + 70
    new_height = max(height1, height2)

    new_image = Image.new('RGB', (new_width, new_height + 50))
    new_image.paste(image1, (0, 70))
    new_image.paste(image2, (width1 + 60, 70))

    draw = ImageDraw.Draw(new_image)

    font = ImageFont.truetype("arial.ttf", 16)

    text_position = (0, 0)

    text_color = (255, 255, 255)

    draw.text(text_position, text1, font=font, fill=text_color)

    text_position = (0, 35)

    draw.text(text_position, text2, font=font, fill=text_color)

    new_image.save(os.path.join(save_root, save_name + '.png'))

    return 


def output_as_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def auto_configure_device_map(num_gpus):
    num_trans_layers = 32
    per_gpu_layers = 38 / num_gpus

    device_map = {
        'visual_encoder': 0,
        'ln_vision': 0,
        'Qformer': 0,
        'internlm_model.model.embed_tokens': 0,
        'internlm_model.model.norm': 0,
        'internlm_model.lm_head': 0,
        'query_tokens': 0,
        'flag_image_start': 0,
        'flag_image_end': 0,
        'internlm_proj.weight': 0,
        'internlm_proj.bias': 0,
    }

    used = 6
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'internlm_model.model.layers.{i}'] = gpu_target
        used += 1

    return device_map


def consistency_check_and_summary(
        if_consistency_check:bool, 
        consistency_list, 
        answer_bias_check, 
        ans,
        pred_score=[0, 0]):
    
    consistency = False
    # winner = 'draw'

    if if_consistency_check:
        if len(consistency_list) == 2:
            if consistency_list[0] == consistency_list[1]:
                consistency = True
                winner = consistency_list[0]
            else:
                consistency = False
                winner = 'N.A.'
        return {
        'consistency': consistency,
        'winner': winner,
        'answer1': ans[0],
        'answer2': ans[1],
        'pred_score1': pred_score[0],
        'pred_score2': pred_score[1],
        'answer_bias_check': answer_bias_check
        } 
    
    else:
        consistency = True
        if len(consistency_list) != 0:
            winner = consistency_list[0]

        return {
            'consistency': consistency,
            'winner': winner,
            'answer1': ans[0],
            'pred_score1': pred_score[0],
            'pred_score2': pred_score[1],
            'answer_bias_check': answer_bias_check
        } 


def downsample_image(image_path, max_short_side=1024):
    # 打开图像
    image = Image.open(image_path)

    # 获取图像的宽度和高度
    width, height = image.size

    # 确定下采样的比例
    if width < height:
        # 如果宽度小于高度，则以宽度为基准进行下采样
        if width <= max_short_side:
            return  # 图片已经符合要求，无需下采样

        new_width = max_short_side
        new_height = int(height * (max_short_side / width))
    else:
        # 如果高度小于等于宽度，则以高度为基准进行下采样
        if height <= max_short_side:
            return  # 图片已经符合要求，无需下采样

        new_width = int(width * (max_short_side / height))
        new_height = max_short_side

    # 执行下采样
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    
    return resized_image
 