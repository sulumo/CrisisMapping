import argparse
from html import parser

def get_args():
    parser = argparse.ArgumentParser()

    # ------------------ General Model Config ------------------ #
    parser.add_argument('--mode', choices=['both', 'image_only', 'text_only'], default='both',
                        help="Which part of the model to use.")
    parser.add_argument('--task', choices=['task1', 'task2', 'task2_merged', 'task3', 'multitask'], default='multitask')

    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--save_dir', default='./output', type=str)
    parser.add_argument('--model_name', default='', type=str)

    # ------------------ Model Loading ------------------ #
    parser.add_argument('--model_to_load', default='', type=str,
                        help='Path to a pretrained multitask model.')
    parser.add_argument('--image_model_to_load', default='', type=str)
    parser.add_argument('--text_model_to_load', default='', type=str)


    parser.add_argument('--num_epochs', default=24, type=int)
    parser.add_argument('--max_iter', default=5700, type=int)

    # ------------------ Run Mode Flags ------------------ #
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', action='store_true', help="Only evaluate, do not train.")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true')

    # ------------------ System ------------------ #
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=8, type=int)

    # ------------------ Dataset & Preprocessing ------------------ #
    parser.add_argument('--merged_dataset_path', type=str, default='./merged_dataset.tsv',
                        help='Path to the multitask merged TSV file.')

    parser.add_argument('--load_size', default=228, type=int)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--max_dataset_size', default=2147483648, type=int)

    parser.add_argument('--text_from', choices=['wiki', 'llava', 'tweet'], default='llava')

    # ------------------ Reproducibility ------------------ #
    parser.add_argument('--seed', default=42, type=int)


    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)


    return parser.parse_args()
