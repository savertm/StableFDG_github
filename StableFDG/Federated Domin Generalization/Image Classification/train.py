import argparse
import copy
import torch
import random
import numpy as np
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import time

# custom
from yacs.config import CfgNode as CN
import trainers.vanilla2

#
torch.set_num_threads(2)

def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('************')
    print('** Config **')
    print('************')
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.before_relu:
        cfg.MODEL.BACKBONE.BEFORE_RELU = True

    cfg.TRAINER.VANILLA2.SORTING = args.sorting
    cfg.TRAINER.VANILLA2.STAGE = args.stage
    cfg.MODEL.BACKBONE.WEIGHT_LIST = args.weight_list
    cfg.MODEL.BACKBONE.ADV_WEIGHT = args.adv_weight
    cfg.MODEL.BACKBONE.MIX_WEIGHT = args.mix_weight

    cfg.TRAINER.VANILLA3 = CN()  # actually, it is not used.
    cfg.TRAINER.VANILLA3.mix_or_swap = args.mix_or_swap
    cfg.TRAINER.VANILLA3.mix_alpha = args.mix_alpha
    cfg.TRAINER.VANILLA3.statistic_weight = args.statistic_weight

    cfg.DATALOADER.DATA_DISTRIBUTION = args.data_distribution
    cfg.TRAINER.STABLEFDG.EXPLORATION_LEVEL = args.exploration_level
    cfg.TRAINER.STABLEFDG.OVERSAMPLING_SIZE = args.oversampling_size

def extend_cfg(cfg):
    # Here you can extend the existing cfg variables by adding new ones
    cfg.TRAINER.VANILLA2 = CN()
    cfg.TRAINER.VANILLA2.MIX = 'random'  # random or crossdomain
    cfg.TRAINER.VANILLA2.SORTING = 'quicksort'  # quicksort | index | random | neighbor
    cfg.TRAINER.VANILLA2.STAGE = 'one'  # one | two | three | four
    cfg.MODEL.BACKBONE.ADV_WEIGHT = 1.0
    cfg.MODEL.BACKBONE.MIX_WEIGHT = 1.0
    cfg.MODEL.BACKBONE.WEIGHT_LIST = '0-0-0-0'

    cfg.TRAINER.SEMIMIXSTYLE = CN()
    cfg.TRAINER.SEMIMIXSTYLE.WEIGHT_U = 1. # weight on the unlabeled loss
    cfg.TRAINER.SEMIMIXSTYLE.CONF_THRE = 0.95 # confidence threshold
    cfg.TRAINER.SEMIMIXSTYLE.STRONG_TRANSFORMS = ()
    cfg.TRAINER.SEMIMIXSTYLE.MS_LABELED = False # apply mixstyle to labeled data
    cfg.TRAINER.SEMIMIXSTYLE.MIX = 'random' # random or crossdomain

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    reset_cfg(cfg, args)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    start_time = time.time()
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed_more(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.vis:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.vis()
        return

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()
    training_time = time.time() - start_time
    print('start time is:', time.ctime(start_time))
    print('end time is:', time.ctime(time.time()))
    print('training time is:', training_time)


def set_random_seed_more(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument(
        '--output-dir', type=str, default='', help='output directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='checkpoint directory (from which the training resumes)'
    )
    parser.add_argument(
        '--mix_or_swap',
        type=str,
        default='swap',
        help=''
    )
    parser.add_argument(
        '--weight_list',
        type=str,
        default='0-0-0-0',
        help=''
    )
    parser.add_argument(
        '--stage',
        type=str,
        default='one',
        help=''
    )
    parser.add_argument(
        '--mix_alpha',
        type=float,
        default=0.1,
        help='alpha for mean/var mix'
    )
    parser.add_argument(
        '--adv_weight',
        type=float,
        default=1.0,
        help='weight for adversarial training'
    )
    parser.add_argument(
        '--mix_weight',
        type=float,
        default=1.0,
        help='new statistics = mix_weight * adv statistics + (1-mix_weight) * batch statistics'
    )
    parser.add_argument(
        '--statistic_weight',
        type=float,
        default=0.2,
        help='weight for statistic'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=-1,
        help='only positive value enables a fixed seed'
    )
    parser.add_argument(
        '--source-domains',
        type=str,
        nargs='+',
        help='source domains for DA/DG'
    )
    parser.add_argument(
        '--target-domains',
        type=str,
        nargs='+',
        help='target domains for DA/DG'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation methods'
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '--dataset-config-file',
        type=str,
        default='',
        help='path to config file for dataset setup'
    )
    parser.add_argument(
        '--trainer', type=str, default='', help='name of trainer'
    )
    parser.add_argument(
        '--backbone', type=str, default='', help='name of CNN backbone'
    )
    parser.add_argument('--head', type=str, default='', help='name of head')
    parser.add_argument(
        '--eval-only', action='store_true', help='evaluation only'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='',
        help='load model from this directory for eval-only mode'
    )
    parser.add_argument(
        '--load-epoch',
        type=int,
        help='load model weights at this epoch for evaluation'
    )
    parser.add_argument(
        '--no-train', action='store_true', help='do not call trainer.train()'
    )
    parser.add_argument(
        '--before_relu', action='store_true', help='by default, apply style mix after relu; we propose to conduct style mix before relu'
    )
    parser.add_argument('--sorting', type=str, default='quicksort', help='quicksort | index | random | neighbor')
    parser.add_argument('--vis', action='store_true', help='visualization')
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='modify config options using the command-line'
    )

    parser.add_argument('--data_distribution', type=str, default='Single', help='Single | Multi')
    parser.add_argument('--exploration_level', type=float, default='3.0', help='Exploration level (alpha)')
    parser.add_argument('--oversampling_size', type=int, default='32', help='Oversampling Size')

    args = parser.parse_args()
    main(args)