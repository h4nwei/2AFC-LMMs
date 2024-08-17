import argparse
from tkinter import NONE
from unittest import defaultTestLoader
from torch import NoneType

def parse_config():
    parser = argparse.ArgumentParser(description='2AFC Prompting of LMMs for IQA')

    # Data specifications
    parser.add_argument('--data_dir', type=str, default='/data/zhw/datasets/', help='dataset directory')
    parser.add_argument('--dataset_name', type=str, default='Coarse_grain_mixed', help='dataset name')
    parser.add_argument('--model_name', type=str, default='ChatGPT', help='name of the model')
    parser.add_argument('--json_dir', type=str, default='./data', help='root to the metadata of each dataset')
    parser.add_argument('--result_dir', type=str, default='./results', help='dataset directory')
    parser.add_argument('--resume', type=bool, default=False, help='resume competing')
    parser.add_argument('--save_image', type=bool, default=False, help='whether to save the image')
    parser.add_argument('--round', type=int, default=12, help='number of rounds for each image pair')
    parser.add_argument('--resize', type=int, default=225, help='resize input image')
    parser.add_argument("--epochs_per_eval", type=int, default=1, help="number of epochs per evaluation")
    return parser.parse_args()


