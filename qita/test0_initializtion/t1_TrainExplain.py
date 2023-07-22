import os
import sys
import torch
import numpy as np

import datetime
import logging

import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm


def parse_args():
    parse = argparse.ArgumentParser(description="用于分类训练的参数解析")
    parse.add_argument('--epoch', type=int, default=10, help='训练的轮数')
    args = parse.parse_args()
    return args


def main(args):
    print(args.epoch, type(args.epoch))


if __name__ == '__main__':
    args = parse_args()
    main(args)
