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

from TSPPreferenceTrainer import TSPPreferenceTrainer as Trainer


##########################################################################################
# defaults

DEFAULT_BASE_CHECKPOINT = "./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt"
DEFAULT_CURRICULUM_SIZES = [150, 200, 300]
DEFAULT_CURRICULUM_STAGE_EPOCHS = [20, 30, 40]
DEFAULT_EPOCHS = sum(DEFAULT_CURRICULUM_STAGE_EPOCHS)
DEFAULT_BASE_REPLAY_PROBLEM_SIZE = 100
DEFAULT_CURRENT_STAGE_MIX_WEIGHT = 0.7
DEFAULT_PREVIOUS_STAGE_MIX_WEIGHT = 0.2
DEFAULT_BASE_REPLAY_MIX_WEIGHT = 0.1


##########################################################################################
# CLI helpers

def str2bool(value):
    if isinstance(value, bool):
        return value

    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_batch_schedule(schedule_text):
    schedule = {}
    for item in schedule_text.split(','):
        item = item.strip()
        if not item:
            continue
        problem_size_text, batch_size_text = item.split(':')
        schedule[int(problem_size_text)] = int(batch_size_text)
    return schedule


def resolve_curriculum_stage_epochs(args):
    if args.curriculum_stage_epochs is not None:
        stage_epochs = list(args.curriculum_stage_epochs)
        if len(stage_epochs) != len(args.curriculum_problem_sizes):
            raise ValueError(
                "--curriculum_stage_epochs must have the same length as "
                "--curriculum_problem_sizes."
            )
        if any(stage_epoch <= 0 for stage_epoch in stage_epochs):
            raise ValueError("--curriculum_stage_epochs must contain only positive integers.")
        if sum(stage_epochs) != args.epochs:
            raise ValueError(
                "--epochs ({}) must match the sum of --curriculum_stage_epochs ({}).".format(
                    args.epochs,
                    sum(stage_epochs),
                )
            )
        return stage_epochs

    if (
        args.curriculum_problem_sizes == DEFAULT_CURRICULUM_SIZES
        and args.epochs == DEFAULT_EPOCHS
    ):
        return list(DEFAULT_CURRICULUM_STAGE_EPOCHS)

    stage_count = len(args.curriculum_problem_sizes)
    if stage_count == 0:
        raise ValueError("--curriculum_problem_sizes must not be empty.")

    base_stage_epoch = args.epochs // stage_count
    remainder = args.epochs % stage_count
    return [
        base_stage_epoch + (1 if stage_idx < remainder else 0)
        for stage_idx in range(stage_count)
    ]


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run preference-optimization post-training for POMO with a curriculum over "
            "larger TSP problem sizes."
        )
    )
    parser.add_argument("--base_checkpoint", default=DEFAULT_BASE_CHECKPOINT,
                        help="Checkpoint used as both initialization and frozen reference model.")
    parser.add_argument("--resume_checkpoint", default=None,
                        help="Resume an interrupted post-training run from a saved checkpoint.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Number of post-training epochs.")
    parser.add_argument("--train_episodes", type=int, default=2048,
                        help="Number of training episodes per epoch.")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Fallback batch size when a problem size is not in --batch_schedule.")
    parser.add_argument("--min_train_batch_size", type=int, default=1,
                        help="Minimum batch size allowed when auto-reducing after CUDA OOM.")
    parser.add_argument(
        "--batch_schedule",
        default="100:32,150:32,200:24,300:12",
        help="Per-size batch schedule formatted as '100:32,150:32,200:24,...'.",
    )
    parser.add_argument(
        "--curriculum_problem_sizes",
        type=int,
        nargs='+',
        default=DEFAULT_CURRICULUM_SIZES,
        help="Problem sizes used by the curriculum in order.",
    )
    parser.add_argument(
        "--curriculum_stage_epochs",
        type=int,
        nargs='+',
        default=None,
        help=(
            "Optional per-stage epoch counts aligned with --curriculum_problem_sizes. "
            "Defaults to 20/30/40 for 150/200/300, otherwise evenly splits --epochs."
        ),
    )
    parser.add_argument("--base_replay_problem_size", type=int, default=DEFAULT_BASE_REPLAY_PROBLEM_SIZE,
                        help="Base problem size kept as replay throughout post-training.")
    parser.add_argument("--current_stage_mix_weight", type=float, default=DEFAULT_CURRENT_STAGE_MIX_WEIGHT,
                        help="Replay weight for the current curriculum stage.")
    parser.add_argument("--previous_stage_mix_weight", type=float, default=DEFAULT_PREVIOUS_STAGE_MIX_WEIGHT,
                        help="Replay weight for the previous curriculum stage.")
    parser.add_argument("--base_replay_mix_weight", type=float, default=DEFAULT_BASE_REPLAY_MIX_WEIGHT,
                        help="Replay weight for the base problem size.")
    parser.add_argument("--preference_beta", type=float, default=0.1,
                        help="Temperature used in the DPO-style preference loss.")
    parser.add_argument("--preference_pair_k", type=int, default=4,
                        help="Use top-k vs bottom-k sampled tours to build multiple preference pairs.")
    parser.add_argument("--preference_loss_weight", type=float, default=1.0,
                        help="Weight applied to the preference loss.")
    parser.add_argument("--use_reference_candidate_pool", type=str2bool, default=True,
                        help="Augment preference candidates with sampled rollouts from the frozen reference model.")
    parser.add_argument("--preference_gap_weight_power", type=float, default=1.0,
                        help="Exponent applied to normalized chosen-vs-rejected reward gaps.")
    parser.add_argument("--rl_loss_weight", type=float, default=0.2,
                        help="Weight applied to the original REINFORCE loss for stability.")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for post-training.")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="Weight decay for post-training.")
    parser.add_argument("--milestones", type=int, nargs='*', default=[45, 55],
                        help="LR decay milestones for MultiStepLR.")
    parser.add_argument("--scheduler_gamma", type=float, default=0.2,
                        help="LR decay factor for MultiStepLR.")
    parser.add_argument("--use_cuda", type=str2bool, default=DEFAULT_USE_CUDA,
                        help="Whether to use CUDA.")
    parser.add_argument("--cuda_device_num", type=int, default=DEFAULT_CUDA_DEVICE_NUM,
                        help="CUDA device id when --use_cuda=true.")
    parser.add_argument("--debug", type=str2bool, default=DEFAULT_DEBUG_MODE,
                        help="Use a short debug run.")
    return parser


##########################################################################################
# parameter builders

def build_env_params(args):
    base_problem_size = min([args.base_replay_problem_size] + args.curriculum_problem_sizes)
    return {
        'problem_size': base_problem_size,
        'pomo_size': base_problem_size,
    }


def build_model_params():
    return {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
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
            'weight_decay': args.weight_decay
        },
        'scheduler': {
            'milestones': args.milestones,
            'gamma': args.scheduler_gamma
        }
    }


def build_trainer_params(args):
    curriculum_stage_epochs = resolve_curriculum_stage_epochs(args)
    return {
        'use_cuda': args.use_cuda,
        'cuda_device_num': args.cuda_device_num,
        'epochs': args.epochs,
        'train_episodes': args.train_episodes,
        'train_batch_size': args.train_batch_size,
        'min_train_batch_size': args.min_train_batch_size,
        'train_batch_size_by_problem_size': parse_batch_schedule(args.batch_schedule),
        'preference_beta': args.preference_beta,
        'preference_pair_k': args.preference_pair_k,
        'preference_loss_weight': args.preference_loss_weight,
        'use_reference_candidate_pool': args.use_reference_candidate_pool,
        'preference_gap_weight_power': args.preference_gap_weight_power,
        'rl_loss_weight': args.rl_loss_weight,
        'curriculum': {
            'problem_sizes': args.curriculum_problem_sizes,
            'stage_epochs': curriculum_stage_epochs,
            'base_replay_problem_size': args.base_replay_problem_size,
            'current_stage_mix_weight': args.current_stage_mix_weight,
            'previous_stage_mix_weight': args.previous_stage_mix_weight,
            'base_replay_mix_weight': args.base_replay_mix_weight,
        },
        'logging': {
            'model_save_interval': 10,
            'img_save_interval': 10,
            'log_image_params_1': {
                'json_foldername': 'log_image_style',
                'filename': 'style_tsp_100.json'
            },
            'log_image_params_2': {
                'json_foldername': 'log_image_style',
                'filename': 'style_loss_1.json'
            },
        },
        'model_load': {
            'enable': True,
            'path': os.path.abspath(args.base_checkpoint),
        },
        'resume_load': {
            'enable': args.resume_checkpoint is not None,
            'path': os.path.abspath(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
        }
    }


def build_logger_params(args):
    desc = 'post_train__pref__curriculum_{}'.format('_'.join(map(str, args.curriculum_problem_sizes)))
    return {
        'log_file': {
            'desc': desc,
            'filename': 'log.txt'
        }
    }


##########################################################################################
# main

def main():
    args = build_parser().parse_args()
    if args.debug:
        args.train_episodes = 64
        args.curriculum_problem_sizes = args.curriculum_problem_sizes[:2]
        if args.curriculum_stage_epochs is not None:
            args.curriculum_stage_epochs = args.curriculum_stage_epochs[:2]
            args.epochs = sum(args.curriculum_stage_epochs)
        else:
            args.epochs = 4

    env_params = build_env_params(args)
    model_params = build_model_params()
    optimizer_params = build_optimizer_params(args)
    trainer_params = build_trainer_params(args)
    logger_params = build_logger_params(args)

    create_logger(**logger_params)
    _print_config(args, env_params, model_params, optimizer_params, trainer_params)

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

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
    logger.info(
        'Preference post-training uses a frozen reference checkpoint, top-k vs bottom-k '
        'multi-pair preference supervision, a 90-epoch mixed-replay curriculum over 150/200/300 by default '
        '(20/30/40 epochs per stage), '
        'and reward-gap weighted preference loss.'
    )


##########################################################################################

if __name__ == "__main__":
    main()
