##########################################################################################
# Machine Environment Config (default values; can be overridden by CLI)

DEFAULT_DEBUG_MODE = False
DEFAULT_USE_CUDA = not DEFAULT_DEBUG_MODE
DEFAULT_CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import pytz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

sys.path.insert(0, "../..")  # for utils
sys.path.insert(0, "..")  # for TSProblemDef

TSP_DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


##########################################################################################
# import

from utils.utils import create_logger, copy_all_src, get_result_folder

from TSPTester_EAS import TSPTester_EAS


##########################################################################################
# defaults

DEFAULT_DATA_PATH = os.path.join(TSP_DATA_ROOT, "data", "val")
DEFAULT_MODEL_DIR = "./result/saved_tsp100_model2_longTrain"
DEFAULT_MODEL_EPOCH = 3000
DEFAULT_AUGMENTATION_ENABLE = True
DEFAULT_AUG_FACTOR = 8
DEFAULT_NUM_SAMPLES = 8
DEFAULT_ENABLE_2OPT = True
DEFAULT_DETAILED_LOG = True
DEFAULT_TRAIN_LR_REFERENCE = 1e-4
DEFAULT_EAS_STEPS = 100
DEFAULT_EAS_PARAM_GROUP = "embedding"
DEFAULT_EAS_RECORD_INTERVAL = 10
DEFAULT_EAS_LOG_INTERVAL = 20
DEFAULT_EAS_OPTIMIZER = "adam"
DEFAULT_EAS_WEIGHT_DECAY = 0.0
DEFAULT_EAS_MOMENTUM = 0.0
DEFAULT_EAS_GRAD_CLIP = 1.0
DEFAULT_EAS_PATIENCE = 2
DEFAULT_EAS_RESTARTS = 2
DEFAULT_EAS_LOSS_TYPE = "reinforce"
DEFAULT_EAS_ELITE_RATIO = 0.25

MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** (1 / 2),
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}


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


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a TSP model with Efficient Active Search on a directory of TSPLIB instances."
        )
    )
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH, help="Directory containing .tsp files.")
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Exact checkpoint file path. If omitted, --model_dir and --epoch will be used.",
    )
    parser.add_argument(
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help="Checkpoint directory. Used only when --checkpoint_path is not provided.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=DEFAULT_MODEL_EPOCH,
        help="Checkpoint epoch. Used only when --checkpoint_path is not provided.",
    )
    parser.add_argument("--use_cuda", type=str2bool, default=DEFAULT_USE_CUDA, help="Whether to use CUDA.")
    parser.add_argument(
        "--cuda_device_num",
        type=int,
        default=DEFAULT_CUDA_DEVICE_NUM,
        help="CUDA device id when --use_cuda=true.",
    )
    parser.add_argument(
        "--augmentation_enable",
        type=str2bool,
        default=DEFAULT_AUGMENTATION_ENABLE,
        help="Enable test-time augmentation before EAS.",
    )
    parser.add_argument("--aug_factor", type=int, default=DEFAULT_AUG_FACTOR, help="Augmentation factor.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of sampled decoding rounds in the final post-EAS evaluation (1 = greedy only).",
    )
    parser.add_argument(
        "--enable_2opt",
        type=str2bool,
        default=DEFAULT_ENABLE_2OPT,
        help="Enable 2-opt local search in the final post-EAS evaluation.",
    )
    parser.add_argument(
        "--detailed_log",
        type=str2bool,
        default=DEFAULT_DETAILED_LOG,
        help="Dump per-instance lists to the log.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path for machine-readable evaluation output in JSON format.",
    )
    parser.add_argument(
        "--scale_min",
        type=int,
        default=0,
        help="Minimum instance size (inclusive) for filtering.",
    )
    parser.add_argument(
        "--scale_max",
        type=int,
        default=10000,
        help="Maximum instance size (exclusive) for filtering.",
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=DEFAULT_DEBUG_MODE,
        help="Use a smaller size filter for quick debugging.",
    )
    parser.add_argument(
        "--eas_steps",
        type=int,
        default=DEFAULT_EAS_STEPS,
        help="Total number of EAS updates budgeted for each test instance.",
    )
    parser.add_argument(
        "--eas_train_lr_reference",
        type=float,
        default=DEFAULT_TRAIN_LR_REFERENCE,
        help="Reference training LR used to derive the default EAS LR.",
    )
    parser.add_argument(
        "--eas_lr",
        type=float,
        default=None,
        help="Explicit EAS learning rate. Defaults to --eas_train_lr_reference / 10.",
    )
    parser.add_argument(
        "--eas_optimizer",
        choices=["sgd", "adam", "adamw"],
        default=DEFAULT_EAS_OPTIMIZER,
        help="Optimizer used inside EAS test-time adaptation.",
    )
    parser.add_argument(
        "--eas_weight_decay",
        type=float,
        default=DEFAULT_EAS_WEIGHT_DECAY,
        help="Weight decay applied by the EAS optimizer.",
    )
    parser.add_argument(
        "--eas_momentum",
        type=float,
        default=DEFAULT_EAS_MOMENTUM,
        help="Momentum used when --eas_optimizer=sgd.",
    )
    parser.add_argument(
        "--eas_grad_clip",
        type=float,
        default=DEFAULT_EAS_GRAD_CLIP,
        help="Gradient clipping threshold for EAS updates; <=0 disables clipping.",
    )
    parser.add_argument(
        "--eas_patience",
        type=int,
        default=DEFAULT_EAS_PATIENCE,
        help="Early-stop an EAS restart after this many recorded checkpoints without improvement; 0 disables it.",
    )
    parser.add_argument(
        "--eas_restarts",
        type=int,
        default=DEFAULT_EAS_RESTARTS,
        help="Split the total EAS update budget across multiple short restarts from the base checkpoint.",
    )
    parser.add_argument(
        "--eas_loss_type",
        choices=["reinforce", "elite_reinforce"],
        default=DEFAULT_EAS_LOSS_TYPE,
        help="Instance-level policy-gradient objective used during EAS.",
    )
    parser.add_argument(
        "--eas_elite_ratio",
        type=float,
        default=DEFAULT_EAS_ELITE_RATIO,
        help="Fraction of tours kept when --eas_loss_type=elite_reinforce.",
    )
    parser.add_argument(
        "--eas_param_group",
        choices=[
            "embedding",
            "decoder_wq_last",
            "decoder_combine",
            "decoder_last",
            "encoder_first2",
            "embedding_decoder",
        ],
        default=DEFAULT_EAS_PARAM_GROUP,
        help="Small parameter subset to fine-tune for each test instance.",
    )
    parser.add_argument(
        "--eas_record_interval",
        type=int,
        default=DEFAULT_EAS_RECORD_INTERVAL,
        help="Record an EAS candidate checkpoint every N updates and use the best recorded one for final inference.",
    )
    parser.add_argument(
        "--eas_log_interval",
        type=int,
        default=DEFAULT_EAS_LOG_INTERVAL,
        help="How often to log intermediate EAS updates.",
    )
    parser.add_argument(
        "--eas_selection_num_samples",
        type=int,
        default=None,
        help="Number of re-decode samples used to score recorded EAS checkpoints. Defaults to --num_samples.",
    )
    parser.add_argument(
        "--eas_selection_enable_2opt",
        type=str2bool,
        default=None,
        help="Whether recorded EAS checkpoints are scored with 2-opt. Defaults to --enable_2opt.",
    )
    return parser


def resolve_checkpoint_path(args):
    if args.checkpoint_path is not None:
        return os.path.abspath(args.checkpoint_path)
    return os.path.abspath(os.path.join(args.model_dir, f"checkpoint-{args.epoch}.pt"))


def build_tester_params(args):
    eas_lr = args.eas_lr
    if eas_lr is None:
        eas_lr = args.eas_train_lr_reference / 10

    selection_num_samples = args.eas_selection_num_samples
    if selection_num_samples is None:
        selection_num_samples = max(1, args.num_samples)

    selection_enable_2opt = args.eas_selection_enable_2opt
    if selection_enable_2opt is None:
        selection_enable_2opt = args.enable_2opt

    return {
        "use_cuda": args.use_cuda,
        "cuda_device_num": args.cuda_device_num,
        "checkpoint_path": resolve_checkpoint_path(args),
        "filename": os.path.abspath(args.data_path),
        "augmentation_enable": args.augmentation_enable,
        "aug_factor": args.aug_factor,
        "num_samples": max(1, args.num_samples),
        "enable_2opt": args.enable_2opt,
        "detailed_log": args.detailed_log,
        "scale_range_all": [[args.scale_min, args.scale_max]],
        "eas_steps": args.eas_steps,
        "eas_lr": eas_lr,
        "eas_optimizer": args.eas_optimizer,
        "eas_weight_decay": args.eas_weight_decay,
        "eas_momentum": args.eas_momentum,
        "eas_grad_clip": args.eas_grad_clip,
        "eas_patience": max(0, args.eas_patience),
        "eas_restarts": max(1, args.eas_restarts),
        "eas_loss_type": args.eas_loss_type,
        "eas_elite_ratio": args.eas_elite_ratio,
        "eas_param_group": args.eas_param_group,
        "eas_record_interval": max(1, args.eas_record_interval),
        "eas_log_interval": max(1, args.eas_log_interval),
        "eas_selection_num_samples": max(1, selection_num_samples),
        "eas_selection_enable_2opt": selection_enable_2opt,
        "eas_train_lr_reference": args.eas_train_lr_reference,
    }


def build_logger_params(tester_params):
    if tester_params["augmentation_enable"]:
        highlight = f'aug{tester_params["aug_factor"]}'
    else:
        highlight = "no_aug"
    highlight = f"{highlight}_eas{tester_params['eas_steps']}_{tester_params['eas_param_group']}"
    highlight = f"{highlight}_{tester_params['eas_optimizer']}"
    if tester_params["eas_restarts"] > 1:
        highlight = f"{highlight}_r{tester_params['eas_restarts']}"
    highlight = f"{highlight}_sample{tester_params['num_samples']}"
    if tester_params["enable_2opt"]:
        highlight = f"{highlight}_2opt"

    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
    return {
        "log_file": {
            "desc": f"{highlight}_test_TSPLIB_POMO",
            "filename": "run_log.txt",
            "filepath": "./result_lib/" + process_start_time.strftime("%Y%m%d_%H%M%S") + "{desc}",
        }
    }


def build_result_payload(tester_params, result):
    payload = {
        "interface_version": 1,
        "primary_metric": "avg_aug_gap",
        "primary_metric_value": result.avg_aug_gap,
        "avg_aug_gap": result.avg_aug_gap,
        "avg_no_aug_gap": result.avg_no_aug_gap,
        "augmentation_enable": tester_params["augmentation_enable"],
        "aug_factor": tester_params["aug_factor"],
        "num_samples": tester_params["num_samples"],
        "enable_2opt": tester_params["enable_2opt"],
        "eas_steps": tester_params["eas_steps"],
        "eas_lr": tester_params["eas_lr"],
        "eas_optimizer": tester_params["eas_optimizer"],
        "eas_weight_decay": tester_params["eas_weight_decay"],
        "eas_grad_clip": tester_params["eas_grad_clip"],
        "eas_patience": tester_params["eas_patience"],
        "eas_restarts": tester_params["eas_restarts"],
        "eas_loss_type": tester_params["eas_loss_type"],
        "eas_elite_ratio": tester_params["eas_elite_ratio"],
        "eas_param_group": tester_params["eas_param_group"],
        "eas_record_interval": tester_params["eas_record_interval"],
        "eas_selection_num_samples": tester_params["eas_selection_num_samples"],
        "eas_selection_enable_2opt": tester_params["eas_selection_enable_2opt"],
        "checkpoint_path": tester_params["checkpoint_path"],
        "data_path": tester_params["filename"],
        "solved_instance_num": result.solved_instance_num,
        "total_instance_num": result.total_instance_num,
    }
    payload.update(result.to_dict())
    return payload


def dump_json_if_needed(output_json, payload):
    if output_json is None:
        return

    output_dir = os.path.dirname(os.path.abspath(output_json))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


##########################################################################################
# main

def main():
    args = build_parser().parse_args()
    tester_params = build_tester_params(args)

    if args.debug:
        tester_params["scale_range_all"] = [[0, 100]]

    logger_params = build_logger_params(tester_params)

    create_logger(**logger_params)
    _print_config(args, tester_params)

    tester = TSPTester_EAS(model_params=MODEL_PARAMS, tester_params=tester_params)

    copy_all_src(get_result_folder())

    result = tester.run_lib()
    payload = build_result_payload(tester_params, result)
    dump_json_if_needed(args.output_json, payload)

    print("SUMMARY_JSON: " + json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _print_config(args, tester_params):
    logger = logging.getLogger("root")
    logger.info("DEBUG_MODE: {}".format(args.debug))
    logger.info("USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(args.use_cuda, args.cuda_device_num))
    logger.info("model_params{}".format(MODEL_PARAMS))
    logger.info("tester_params{}".format(tester_params))
    if args.output_json is not None:
        logger.info("output_json: {}".format(os.path.abspath(args.output_json)))
    logger.info(
        "EAS default LR policy: eas_lr = eas_train_lr_reference / 10 "
        "unless --eas_lr is explicitly provided."
    )
    logger.info(
        "EAS optimizer config: optimizer={}, restarts={}, grad_clip={}, patience={}".format(
            tester_params["eas_optimizer"],
            tester_params["eas_restarts"],
            tester_params["eas_grad_clip"],
            tester_params["eas_patience"],
        )
    )
    logger.info(
        "EAS loss config: loss_type={}, elite_ratio={}".format(
            tester_params["eas_loss_type"],
            tester_params["eas_elite_ratio"],
        )
    )
    logger.info(
        "EAS candidate selection: record every {} updates, score checkpoints with num_samples={}, "
        "enable_2opt={}, and use the best recorded checkpoint for final inference.".format(
            tester_params["eas_record_interval"],
            tester_params["eas_selection_num_samples"],
            tester_params["eas_selection_enable_2opt"],
        )
    )
    logger.info(
        "Final post-EAS test strategy: num_samples={}, enable_2opt={} "
        "(first N-1 rounds use softmax sampling, last round uses argmax).".format(
            tester_params["num_samples"],
            tester_params["enable_2opt"],
        )
    )
    logger.info(
        "Primary metric for EAS evaluation: avg_aug_gap "
        "(computed after augmentation + per-instance EAS when public/private optima are available)."
    )


##########################################################################################

if __name__ == "__main__":
    main()
