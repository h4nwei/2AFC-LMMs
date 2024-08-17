import argparse
import TwoAFC
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_config():
    parser = argparse.ArgumentParser(description='2AFC Prompting of LMMs for IQA')

    # Data specifications
    parser.add_argument('--data_dir', type=str, default='/data/zhw/datasets/', help='dataset directory')
    parser.add_argument('--stage_name', type=str, default='Fine_grained_SPAQ', help='dataset name (default: Coarse_grained_mixed,  Fine_grained_CSIQ_levels, Fine_grained_CSIQ_types, Fine_grained_SPAQ')
    parser.add_argument('--model_name', type=str, default='ChatGPT', help='name of the model')
    parser.add_argument('--json_dir', type=str, default='data', help='root to the metadata of each dataset')
    parser.add_argument('--result_dir', type=str, default='/home/zhw/IQA/code/NeurIPS24/Q-Align/2AFC/results', help='dataset directory')
    parser.add_argument('--save_image', type=bool, default=False, help='whether to save the image')
    parser.add_argument('--round', type=int, default=12, help='number of rounds for each image pair')
    parser.add_argument("--epochs_per_eval", type=int, default=1, help="number of epochs per evaluation")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    t = TwoAFC.TwoAFC_LMM(args)
    t.run_by_dataset()
    
    