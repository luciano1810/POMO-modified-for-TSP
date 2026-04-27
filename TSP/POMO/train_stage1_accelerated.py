##########################################################################################
# Machine Environment Config (default values; can be overridden by CLI)

DEFAULT_DEBUG_MODE = False
DEFAULT_USE_CUDA = not DEFAULT_DEBUG_MODE
DEFAULT_CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import argparse
import logging
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

from utils.utils import create_logger, copy_all_src

from TSPAcceleratedTrainer import TSPAcceleratedTrainer as Trainer


##########################################################################################
# defaults

DEFAULT_PROBLEM_SIZE = 100
DEFAULT_POMO_SIZE = None
DEFAULT_STAGE_NAME = 'stage1_accelerated'
DEFAULT_EPOCHS = 1200
DEFAULT_TRAIN_EPISODES = 100 * 1000
DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-6
DEFAULT_MILESTONES = [800, 1000]
DEFAULT_SCHEDULER_GAMMA = 0.2
DEFAULT_SCST_LOSS_WEIGHT = 1.0
DEFAULT_ELITE_LOSS_WEIGHT = 0.25
DEFAULT_ELITE_TOPK = 8
DEFAULT_TEACHER_LOSS_WEIGHT = 0.5
DEFAULT_TEACHER_USE_2OPT = True
DEFAULT_TWO_OPT_TEACHER_MAX_ITERATIONS = 8
DEFAULT_MAX_GRAD_NORM = 1.0


##########################################################################################
# CLI helpers

def str2bool(value):
    if isinstance(value, bool):
        return value

    lowered = value.lower()
    if lowered in {'true', '1', 'yes', 'y'}:
        return True
    if lowered in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Accelerated stage1 training for TSP. Uses POMO multi-start rollouts with '
            'a self-critical greedy baseline, elite self-imitation, and optional 2-opt teacher distillation.'
        )
    )
    parser.add_argument('--problem_size', type=int, default=DEFAULT_PROBLEM_SIZE,
                        help='Problem size for stage1 random TSP training.')
    parser.add_argument('--pomo_size', type=int, default=DEFAULT_POMO_SIZE,
                        help='POMO rollout width. Defaults to problem_size.')
    parser.add_argument('--stage_name', default=DEFAULT_STAGE_NAME,
                        help='Short name included in result folder metadata.')
    parser.add_argument('--result_dir', default=None,
                        help='Optional explicit result directory for this run.')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of training epochs.')
    parser.add_argument('--train_episodes', type=int, default=DEFAULT_TRAIN_EPISODES,
                        help='Number of training episodes per epoch.')
    parser.add_argument('--train_batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE,
                        help='Training batch size.')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY,
                        help='Weight decay.')
    parser.add_argument('--milestones', type=int, nargs='*', default=DEFAULT_MILESTONES,
                        help='LR decay milestones for MultiStepLR.')
    parser.add_argument('--scheduler_gamma', type=float, default=DEFAULT_SCHEDULER_GAMMA,
                        help='LR decay factor for MultiStepLR.')
    parser.add_argument('--scst_loss_weight', type=float, default=DEFAULT_SCST_LOSS_WEIGHT,
                        help='Weight applied to the self-critical policy-gradient loss.')
    parser.add_argument('--elite_loss_weight', type=float, default=DEFAULT_ELITE_LOSS_WEIGHT,
                        help='Weight applied to the elite self-imitation loss.')
    parser.add_argument('--elite_topk', type=int, default=DEFAULT_ELITE_TOPK,
                        help='Top-k sampled tours per instance used by the elite imitation loss.')
    parser.add_argument('--teacher_loss_weight', type=float, default=DEFAULT_TEACHER_LOSS_WEIGHT,
                        help='Weight applied to the teacher imitation loss.')
    parser.add_argument('--teacher_use_2opt', type=str2bool, default=DEFAULT_TEACHER_USE_2OPT,
                        help='Improve the best sampled/greedy tour with first-improvement 2-opt before imitation.')
    parser.add_argument('--two_opt_teacher_max_iterations', type=int,
                        default=DEFAULT_TWO_OPT_TEACHER_MAX_ITERATIONS,
                        help='Maximum first-improvement 2-opt passes used to refine each teacher tour.')
    parser.add_argument('--max_grad_norm', type=float, default=DEFAULT_MAX_GRAD_NORM,
                        help='Gradient clipping norm. Set <= 0 to disable clipping.')
    parser.add_argument('--init_checkpoint', default=None,
                        help='Optional checkpoint used to initialize the model weights.')
    parser.add_argument('--resume_checkpoint', default=None,
                        help='Resume an interrupted accelerated stage1 run from a saved checkpoint.')
    parser.add_argument('--use_cuda', type=str2bool, default=DEFAULT_USE_CUDA,
                        help='Whether to use CUDA.')
    parser.add_argument('--cuda_device_num', type=int, default=DEFAULT_CUDA_DEVICE_NUM,
                        help='CUDA device id when --use_cuda=true.')
    parser.add_argument('--debug', type=str2bool, default=DEFAULT_DEBUG_MODE,
                        help='Use a short debug run.')
    return parser


##########################################################################################
# parameter builders

def build_env_params(args):
    pomo_size = args.problem_size if args.pomo_size is None else args.pomo_size
    return {
        'problem_size': args.problem_size,
        'pomo_size': pomo_size,
    }


def build_model_params():
    return {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** (1 / 2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
    }


def build_optimizer_params(args):
    return {
        'optimizer': {
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        },
        'scheduler': {
            'milestones': args.milestones,
            'gamma': args.scheduler_gamma,
        },
    }


def build_trainer_params(args):
    init_checkpoint = None if args.init_checkpoint is None else os.path.abspath(args.init_checkpoint)
    resume_checkpoint = None if args.resume_checkpoint is None else os.path.abspath(args.resume_checkpoint)
    return {
        'use_cuda': args.use_cuda,
        'cuda_device_num': args.cuda_device_num,
        'stage_name': args.stage_name,
        'epochs': args.epochs,
        'train_episodes': args.train_episodes,
        'train_batch_size': args.train_batch_size,
        'scst_loss_weight': args.scst_loss_weight,
        'elite_loss_weight': args.elite_loss_weight,
        'elite_topk': args.elite_topk,
        'teacher_loss_weight': args.teacher_loss_weight,
        'teacher_use_2opt': args.teacher_use_2opt,
        'two_opt_teacher_max_iterations': args.two_opt_teacher_max_iterations,
        'max_grad_norm': args.max_grad_norm,
        'logging': {
            'model_save_interval': 100,
            'img_save_interval': 100,
            'log_image_params_1': {
                'json_foldername': 'log_image_style',
                'filename': 'style_tsp_100.json',
            },
            'log_image_params_2': {
                'json_foldername': 'log_image_style',
                'filename': 'style_loss_1.json',
            },
        },
        'model_load': {
            'enable': init_checkpoint is not None,
            'path': init_checkpoint,
        },
        'resume_load': {
            'enable': resume_checkpoint is not None,
            'path': resume_checkpoint,
        },
    }


def build_logger_params(args):
    stage_name = (args.stage_name or DEFAULT_STAGE_NAME).strip() or DEFAULT_STAGE_NAME
    desc = 'train__tsp_n{}__{}'.format(args.problem_size, stage_name)
    log_file = {
        'desc': desc,
        'filename': 'log.txt',
    }
    if args.result_dir is not None:
        log_file['filepath'] = os.path.abspath(args.result_dir)
    return {
        'log_file': log_file,
    }


##########################################################################################
# main

def main():
    args = build_parser().parse_args()
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError('--init_checkpoint and --resume_checkpoint are mutually exclusive.')
    if args.scst_loss_weight < 0 or args.elite_loss_weight < 0 or args.teacher_loss_weight < 0:
        raise ValueError('All loss weights must be non-negative.')
    if args.scst_loss_weight == 0 and args.elite_loss_weight == 0 and args.teacher_loss_weight == 0:
        raise ValueError('At least one training loss weight must be positive.')
    if args.elite_topk <= 0:
        raise ValueError('--elite_topk must be positive.')
    if args.two_opt_teacher_max_iterations < 0:
        raise ValueError('--two_opt_teacher_max_iterations must be non-negative.')
    if args.teacher_use_2opt and args.teacher_loss_weight <= 0:
        raise ValueError('--teacher_use_2opt=true requires --teacher_loss_weight > 0.')

    if args.debug:
        args.epochs = 2
        args.train_episodes = 64
        args.train_batch_size = 4

    env_params = build_env_params(args)
    if env_params['pomo_size'] > env_params['problem_size']:
        raise ValueError('--pomo_size must not exceed --problem_size.')

    model_params = build_model_params()
    optimizer_params = build_optimizer_params(args)
    trainer_params = build_trainer_params(args)
    logger_params = build_logger_params(args)

    create_logger(**logger_params)
    _print_config(args, env_params, model_params, optimizer_params, trainer_params)

    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
    )

    copy_all_src(trainer.result_folder)
    trainer.run()


def _print_config(args, env_params, model_params, optimizer_params, trainer_params):
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(args.debug))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(args.use_cuda, args.cuda_device_num))
    logger.info('env_params{}'.format(env_params))
    logger.info('model_params{}'.format(model_params))
    logger.info('optimizer_params{}'.format(optimizer_params))
    logger.info('trainer_params{}'.format(trainer_params))
    if args.init_checkpoint is not None:
        logger.info('init_checkpoint: {}'.format(os.path.abspath(args.init_checkpoint)))
    if args.resume_checkpoint is not None:
        logger.info('resume_checkpoint: {}'.format(os.path.abspath(args.resume_checkpoint)))


##########################################################################################

if __name__ == '__main__':
    main()
