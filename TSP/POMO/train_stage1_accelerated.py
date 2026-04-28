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
DEFAULT_MIN_TRAIN_BATCH_SIZE = 1
DEFAULT_BATCH_SCHEDULE = '100:64,150:48,200:32,250:24,300:20,500:12'
DEFAULT_BASE_REPLAY_PROBLEM_SIZE = 100
DEFAULT_CURRENT_STAGE_MIX_WEIGHT = 0.75
DEFAULT_PREVIOUS_STAGE_MIX_WEIGHT = 0.20
DEFAULT_BASE_REPLAY_MIX_WEIGHT = 0.05
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-6
DEFAULT_MILESTONES = [800, 1000]
DEFAULT_SCHEDULER_GAMMA = 0.2
DEFAULT_SCST_LOSS_WEIGHT = 1.0
DEFAULT_ELITE_LOSS_WEIGHT = 0.25
DEFAULT_ELITE_TOPK = 8
DEFAULT_TEACHER_LOSS_WEIGHT = 0.5
DEFAULT_TEACHER_LOSS_WEIGHT_STAGE_SCHEDULE = None
DEFAULT_TEACHER_USE_2OPT = True
DEFAULT_TWO_OPT_TEACHER_MAX_ITERATIONS = 4
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


def parse_batch_schedule(schedule_text):
    schedule = {}
    if schedule_text is None:
        return schedule

    for item in schedule_text.split(','):
        item = item.strip()
        if not item:
            continue
        problem_size_text, batch_size_text = item.split(':')
        schedule[int(problem_size_text)] = int(batch_size_text)
    return schedule


def parse_stage_weight_schedule(schedule_text, arg_name):
    if schedule_text is None:
        return None

    schedule_text = schedule_text.strip()
    if not schedule_text:
        return None

    schedule = []
    for item in schedule_text.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            start_epoch_text, weight_text = item.split(':', 1)
            start_epoch = int(start_epoch_text)
            weight = float(weight_text)
        except ValueError as exc:
            raise ValueError(
                '{} must be formatted as "local_epoch:weight,..."'.format(arg_name)
            ) from exc

        if start_epoch <= 0:
            raise ValueError('{} local epochs must be positive integers.'.format(arg_name))
        if weight < 0:
            raise ValueError('{} weights must be non-negative.'.format(arg_name))
        schedule.append((start_epoch, weight))

    if not schedule:
        return None

    schedule.sort(key=lambda item: item[0])
    if schedule[0][0] != 1:
        raise ValueError('{} must start at local epoch 1.'.format(arg_name))

    for idx in range(1, len(schedule)):
        if schedule[idx][0] == schedule[idx - 1][0]:
            raise ValueError('{} contains duplicate local epoch {}.'.format(
                arg_name,
                schedule[idx][0],
            ))

    return schedule


def resolve_curriculum_problem_sizes(args):
    if args.curriculum_problem_sizes is not None:
        return list(args.curriculum_problem_sizes)
    return [args.problem_size]


def resolve_curriculum_stage_epochs(args, curriculum_problem_sizes):
    if args.curriculum_stage_epochs is not None:
        stage_epochs = list(args.curriculum_stage_epochs)
        if len(stage_epochs) != len(curriculum_problem_sizes):
            raise ValueError(
                '--curriculum_stage_epochs must have the same length as --curriculum_problem_sizes.'
            )
        if any(stage_epoch <= 0 for stage_epoch in stage_epochs):
            raise ValueError('--curriculum_stage_epochs must contain only positive integers.')
        if sum(stage_epochs) != args.epochs:
            raise ValueError(
                '--epochs ({}) must match the sum of --curriculum_stage_epochs ({}).'.format(
                    args.epochs,
                    sum(stage_epochs),
                )
            )
        return stage_epochs

    if len(curriculum_problem_sizes) == 1:
        return [args.epochs]

    stage_count = len(curriculum_problem_sizes)
    base_stage_epoch = args.epochs // stage_count
    remainder = args.epochs % stage_count
    return [
        base_stage_epoch + (1 if stage_idx < remainder else 0)
        for stage_idx in range(stage_count)
    ]


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Accelerated stage1 training for TSP. Uses POMO multi-start rollouts with '
            'a self-critical greedy baseline, elite self-imitation, and optional 2-opt teacher distillation.'
        )
    )
    parser.add_argument('--problem_size', type=int, default=DEFAULT_PROBLEM_SIZE,
                        help='Fallback single problem size when --curriculum_problem_sizes is not provided.')
    parser.add_argument('--pomo_size', type=int, default=DEFAULT_POMO_SIZE,
                        help='Optional fixed POMO rollout width. Defaults to current problem_size.')
    parser.add_argument('--stage_name', default=DEFAULT_STAGE_NAME,
                        help='Short name included in result folder metadata.')
    parser.add_argument('--result_dir', default=None,
                        help='Optional explicit result directory for this run.')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of training epochs.')
    parser.add_argument('--train_episodes', type=int, default=DEFAULT_TRAIN_EPISODES,
                        help='Number of training episodes per epoch.')
    parser.add_argument('--train_batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE,
                        help='Fallback training batch size when a problem size is not in --batch_schedule.')
    parser.add_argument('--min_train_batch_size', type=int, default=DEFAULT_MIN_TRAIN_BATCH_SIZE,
                        help='Minimum batch size allowed when auto-reducing after CUDA OOM.')
    parser.add_argument('--batch_schedule', default=DEFAULT_BATCH_SCHEDULE,
                        help="Per-size batch schedule formatted as '100:64,150:48,200:32,...'.")
    parser.add_argument('--curriculum_problem_sizes', type=int, nargs='+', default=None,
                        help='Optional curriculum problem sizes used by stage1 training.')
    parser.add_argument('--curriculum_stage_epochs', type=int, nargs='+', default=None,
                        help='Optional per-stage epoch counts aligned with --curriculum_problem_sizes.')
    parser.add_argument('--base_replay_problem_size', type=int, default=DEFAULT_BASE_REPLAY_PROBLEM_SIZE,
                        help='Base problem size kept as replay throughout accelerated stage1 training.')
    parser.add_argument('--current_stage_mix_weight', type=float, default=DEFAULT_CURRENT_STAGE_MIX_WEIGHT,
                        help='Replay weight for the current curriculum stage.')
    parser.add_argument('--previous_stage_mix_weight', type=float, default=DEFAULT_PREVIOUS_STAGE_MIX_WEIGHT,
                        help='Replay weight for the previous curriculum stage.')
    parser.add_argument('--base_replay_mix_weight', type=float, default=DEFAULT_BASE_REPLAY_MIX_WEIGHT,
                        help='Replay weight for the base problem size.')
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
                        help='Fallback weight applied to the teacher imitation loss.')
    parser.add_argument('--teacher_loss_weight_stage_schedule',
                        default=DEFAULT_TEACHER_LOSS_WEIGHT_STAGE_SCHEDULE,
                        help=(
                            'Optional stage-relative teacher loss schedule formatted as '
                            '"local_epoch:weight,...". Resets at the start of each curriculum stage '
                            'and overrides --teacher_loss_weight when provided.'
                        ))
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

def build_env_params(args, curriculum_problem_sizes):
    base_problem_size = min([args.base_replay_problem_size] + curriculum_problem_sizes)
    pomo_size = base_problem_size if args.pomo_size is None else args.pomo_size
    return {
        'problem_size': base_problem_size,
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


def build_trainer_params(args, curriculum_problem_sizes, curriculum_stage_epochs,
                         teacher_loss_weight_stage_schedule):
    init_checkpoint = None if args.init_checkpoint is None else os.path.abspath(args.init_checkpoint)
    resume_checkpoint = None if args.resume_checkpoint is None else os.path.abspath(args.resume_checkpoint)
    return {
        'use_cuda': args.use_cuda,
        'cuda_device_num': args.cuda_device_num,
        'stage_name': args.stage_name,
        'epochs': args.epochs,
        'train_episodes': args.train_episodes,
        'train_batch_size': args.train_batch_size,
        'min_train_batch_size': args.min_train_batch_size,
        'train_batch_size_by_problem_size': parse_batch_schedule(args.batch_schedule),
        'pomo_size_override': args.pomo_size,
        'scst_loss_weight': args.scst_loss_weight,
        'elite_loss_weight': args.elite_loss_weight,
        'elite_topk': args.elite_topk,
        'teacher_loss_weight': args.teacher_loss_weight,
        'teacher_loss_weight_stage_schedule': teacher_loss_weight_stage_schedule,
        'teacher_use_2opt': args.teacher_use_2opt,
        'two_opt_teacher_max_iterations': args.two_opt_teacher_max_iterations,
        'max_grad_norm': args.max_grad_norm,
        'curriculum': {
            'problem_sizes': curriculum_problem_sizes,
            'stage_epochs': curriculum_stage_epochs,
            'base_replay_problem_size': args.base_replay_problem_size,
            'current_stage_mix_weight': args.current_stage_mix_weight,
            'previous_stage_mix_weight': args.previous_stage_mix_weight,
            'base_replay_mix_weight': args.base_replay_mix_weight,
        },
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


def build_logger_params(args, curriculum_problem_sizes):
    stage_name = (args.stage_name or DEFAULT_STAGE_NAME).strip() or DEFAULT_STAGE_NAME
    if len(curriculum_problem_sizes) == 1:
        desc = 'train__tsp_n{}__{}'.format(curriculum_problem_sizes[0], stage_name)
    else:
        desc = 'train__{}_curriculum_{}'.format(
            stage_name,
            '_'.join(map(str, curriculum_problem_sizes)),
        )
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
    teacher_loss_weight_stage_schedule = parse_stage_weight_schedule(
        args.teacher_loss_weight_stage_schedule,
        '--teacher_loss_weight_stage_schedule',
    )
    if teacher_loss_weight_stage_schedule is None:
        teacher_loss_weight_is_active = args.teacher_loss_weight > 0
    else:
        teacher_loss_weight_is_active = any(
            weight > 0 for _, weight in teacher_loss_weight_stage_schedule
        )

    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        raise ValueError('--init_checkpoint and --resume_checkpoint are mutually exclusive.')
    if args.scst_loss_weight < 0 or args.elite_loss_weight < 0 or args.teacher_loss_weight < 0:
        raise ValueError('All loss weights must be non-negative.')
    if args.scst_loss_weight == 0 and args.elite_loss_weight == 0 and not teacher_loss_weight_is_active:
        raise ValueError('At least one training loss weight must be positive.')
    if args.elite_topk <= 0:
        raise ValueError('--elite_topk must be positive.')
    if args.two_opt_teacher_max_iterations < 0:
        raise ValueError('--two_opt_teacher_max_iterations must be non-negative.')
    if args.teacher_use_2opt and not teacher_loss_weight_is_active:
        raise ValueError(
            '--teacher_use_2opt=true requires a positive teacher loss weight or schedule entry.'
        )

    if args.debug:
        args.epochs = 2
        args.train_episodes = 64
        args.train_batch_size = 4
        if args.curriculum_problem_sizes is not None:
            args.curriculum_problem_sizes = args.curriculum_problem_sizes[:2]
            if args.curriculum_stage_epochs is not None:
                args.curriculum_stage_epochs = args.curriculum_stage_epochs[:len(args.curriculum_problem_sizes)]
                args.epochs = sum(args.curriculum_stage_epochs)
            else:
                args.epochs = 4

    curriculum_problem_sizes = resolve_curriculum_problem_sizes(args)
    curriculum_stage_epochs = resolve_curriculum_stage_epochs(args, curriculum_problem_sizes)
    if any(problem_size <= 0 for problem_size in curriculum_problem_sizes):
        raise ValueError('--curriculum_problem_sizes must contain only positive integers.')
    if args.base_replay_problem_size <= 0:
        raise ValueError('--base_replay_problem_size must be positive.')
    if args.min_train_batch_size <= 0:
        raise ValueError('--min_train_batch_size must be positive.')
    if (
        args.current_stage_mix_weight <= 0 and
        args.previous_stage_mix_weight <= 0 and
        args.base_replay_mix_weight <= 0
    ):
        raise ValueError('Curriculum replay weights must sum to a positive value.')

    env_params = build_env_params(args, curriculum_problem_sizes)
    if args.pomo_size is not None and args.pomo_size > min(curriculum_problem_sizes + [args.base_replay_problem_size]):
        raise ValueError('--pomo_size must not exceed the smallest curriculum/base replay problem size.')

    model_params = build_model_params()
    optimizer_params = build_optimizer_params(args)
    trainer_params = build_trainer_params(
        args,
        curriculum_problem_sizes,
        curriculum_stage_epochs,
        teacher_loss_weight_stage_schedule,
    )
    logger_params = build_logger_params(args, curriculum_problem_sizes)

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
