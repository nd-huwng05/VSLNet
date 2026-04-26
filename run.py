import argparse
import logging
import os
from types import SimpleNamespace
import yaml

from dataset.prepare_dataset import prepare
from mode import *


def load_yaml(mode, path):
    assert mode in ['train', 'test', 'inference', 'prepare']
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        configuration = yaml.safe_load(f)
        basic_yaml = configuration.get('basic'.upper(), {})
        mode_yaml = configuration.get(mode.upper(), {})

        return {**basic_yaml, **mode_yaml}

def get_parser_basics(parser):
    parser.add_argument('--path', type=str, default='config/vsl_net_configuration.yaml', help='Path to config file')
    parser.add_argument('--gpu', type=int, nargs='+', help="Array id's gpu to use")
    parser.add_argument('--data', type=str, help='Path to dataset')
    parser.add_argument('--vocab-size', type=int, help='Vocabulary size')
    parser.add_argument('--embedding_size', type=int, help='Embedding size')
    parser.add_argument('--frames', type=int, help='Num of frames')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description='VSLNet cli terminal')

    subparsers = parser.add_subparsers(dest='command', help='Mode program (train/test/inference)')
    subparsers.required = True

    # Train
    parser_train = subparsers.add_parser('train', help='Train model configuration')
    get_parser_basics(parser_train)
    parser_train.add_argument('--epochs', type=int, help='Number of epochs')
    parser_train.add_argument('--batch-size', type=int, help='Batch size')
    parser_train.add_argument('--lr', type=float, help='Learning rate')
    parser_train.add_argument('--resume', type=bool,  help='Continue training from checkpoint')
    parser_train.set_defaults(func=train)

    # Test
    parser_test = subparsers.add_parser('test', help='Test model configuration')
    get_parser_basics(parser_test)
    parser_test.add_argument('--weights', type=str, required=True, help='Path to model weights (.pt, .h5)')
    parser_test.set_defaults(func=train)

    # Inference
    parser_infer = subparsers.add_parser('inference', help='Inference model configuration')
    get_parser_basics(parser_infer)
    parser_infer.set_defaults(func=train)

    # Prepare Dataset
    parser_pre = subparsers.add_parser('prepare', help='Prepare dataset')
    get_parser_basics(parser_pre)
    parser_pre.set_defaults(func=prepare)
    parser_pre.add_argument('--overwrite', type=bool, help='Overwrite existing dataset')
    parser_pre.add_argument('--workers', type=int, help='Number of data loading workers')

    args = parser.parse_args()
    yaml = load_yaml(args.command, args.path)
    args_dict = vars(args)
    for key, value in args_dict.items():
        if key in ['command', 'path']:
            continue

        if value is not None:
            yaml[key.upper()] = value

    args.params = SimpleNamespace(**yaml)

    args.func(args.params)