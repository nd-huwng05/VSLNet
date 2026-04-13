import os.path

import numpy
import yaml
from sklearn.model_selection import train_test_split
from sympy.strategies.core import switch
from torch.utils.data import Subset

from dataset import SignLanguageDataset
from inference import inference
from prepare import prepare_dataset
from argparse import ArgumentParser
from train import train
from test import test

if __name__ == '__main__':
    parse = ArgumentParser()

    # prepare data
    parse.add_argument('--pre-data', type=bool, default=False)
    parse.add_argument('--frames','-f', type=int, default=64, help='Frame to extract')
    parse.add_argument('--dataset', type=str, default='./dataset', help='Path to dataset raw')
    parse.add_argument('--data', type=str, default='./data', help='Path to data processed')
    parse.add_argument('--num-data-label', type=int, default=10, help='Generate num fake video in vocabulary')
    parse.add_argument('--mode', type=str, default='train', help='train, test, inference')
    parse.add_argument('--config', type=str, default='./VSLNet.yaml')

    args = parse.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        setattr(args, key, value)

    if not os.path.exists(os.path.join(args.data, 'numpy')) or args.pre_data:
        print(f'Extracting {args.frames} for data...')
        prepare_dataset(args)

    print("Loading data...")
    dataset = SignLanguageDataset(args.data)
    indices = numpy.arange(len(dataset))
    labels = dataset.labels.numpy()
    data_idx, test_idx = train_test_split(indices,stratify=labels, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(data_idx,stratify=labels[data_idx], test_size=0.2, random_state=42)

    test_dataset = Subset(dataset, test_idx)
    if args.mode == 'test':
        test(test_dataset, args)
    elif args.mode in ['train', 'test']:
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        train(train_dataset, val_dataset, labels[train_idx], labels[val_idx], args)
        test(test_dataset, args)
    elif args.mode == 'inference':
        inference(args)