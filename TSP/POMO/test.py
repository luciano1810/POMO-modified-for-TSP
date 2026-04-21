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

# POMO/NEW_py_ver utils
sys.path.insert(0, "../..")  # for utils

# TSProblemDef (used by augmentation)
sys.path.insert(0, "..")  # for TSProblemDef

# Local TSPLIB dataset (bundled with POMO)
TSP_DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


##########################################################################################
# import

from utils.utils import create_logger, copy_all_src, get_result_folder

from TSPTester_LIB import TSPTester_LIB


##########################################################################################
# defaults

DEFAULT_DATA_PATH = os.path.join(TSP_DATA_ROOT, "data", "val")
DEFAULT_MODEL_DIR = "./result/saved_tsp100_model2_longTrain"
DEFAULT_MODEL_EPOCH = 3000
DEFAULT_AUGMENTATION_ENABLE = True
DEFAULT_AUG_FACTOR = 8
DEFAULT_DETAILED_LOG = True
DEFAULT_TRAIN_LR_REFERENCE = 1e-4
DEFAULT_EAS_ENABLE = True
DEFAULT_EAS_STEPS = 100
DEFAULT_EAS_PARAM_GROUP = "embedding"
DEFAULT_EAS_LOG_INTERVAL = 20

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
            "Evaluate a TSP model on a directory of TSPLIB instances. "
            "This is the standardized evaluation entrypoint for both public validation "
            "and course-staff hidden testing."
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
        help="Enable test-time augmentation. Official metric uses aug_gap when available.",
    )
    parser.add_argument("--aug_factor", type=int, default=DEFAULT_AUG_FACTOR, help="Augmentation factor.")
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
        "--eas_enable",
        type=str2bool,
        default=DEFAULT_EAS_ENABLE,
        help="Enable Efficient Active Search during testing.",
    )
    parser.add_argument(
        "--eas_steps",
        type=int,
        default=DEFAULT_EAS_STEPS,
        help="Number of SGD updates used by EAS on each test instance.",
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
        help="Explicit EAS SGD learning rate. Defaults to --eas_train_lr_reference / 10.",
    )
    parser.add_argument(
        "--eas_param_group",
        choices=["embedding", "decoder_last", "embedding_decoder"],
        default=DEFAULT_EAS_PARAM_GROUP,
        help="Small parameter subset to fine-tune for each test instance.",
    )
    parser.add_argument(
        "--eas_log_interval",
        type=int,
        default=DEFAULT_EAS_LOG_INTERVAL,
        help="How often to log intermediate EAS updates.",
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

    tester_params = {
        "use_cuda": args.use_cuda,
        "cuda_device_num": args.cuda_device_num,
        "checkpoint_path": resolve_checkpoint_path(args),
        "filename": os.path.abspath(args.data_path),
        "augmentation_enable": args.augmentation_enable,
        "aug_factor": args.aug_factor,
        "detailed_log": args.detailed_log,
        # Only EUC_2D / CEIL_2D are supported (same as ICAM's LIBUtils.TSPLIBReader)
        "scale_range_all": [[args.scale_min, args.scale_max]],
        "eas_enable": args.eas_enable,
        "eas_steps": args.eas_steps,
        "eas_lr": eas_lr,
        "eas_param_group": args.eas_param_group,
        "eas_log_interval": args.eas_log_interval,
        "eas_train_lr_reference": args.eas_train_lr_reference,
    }
    return tester_params


def build_logger_params(args, tester_params):
    if tester_params["augmentation_enable"]:
        highlight = f'aug{tester_params["aug_factor"]}'
    else:
        highlight = "no_aug"
    if tester_params["eas_enable"]:
        highlight = f"{highlight}_eas{tester_params['eas_steps']}_{tester_params['eas_param_group']}"

    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
    return {
        "log_file": {
            "desc": f"{highlight}_test_TSPLIB_POMO",
            "filename": "run_log.txt",
            "filepath": "./result_lib/" + process_start_time.strftime("%Y%m%d_%H%M%S") + "{desc}",
        }
    }


def build_result_payload(args, tester_params, result):
    payload = {
        "interface_version": 1,
        "primary_metric": "avg_aug_gap",
        "primary_metric_value": result.avg_aug_gap,
        "avg_aug_gap": result.avg_aug_gap,
        "avg_no_aug_gap": result.avg_no_aug_gap,
        "augmentation_enable": tester_params["augmentation_enable"],
        "aug_factor": tester_params["aug_factor"],
        "eas_enable": tester_params["eas_enable"],
        "eas_steps": tester_params["eas_steps"],
        "eas_lr": tester_params["eas_lr"],
        "eas_param_group": tester_params["eas_param_group"],
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

    logger_params = build_logger_params(args, tester_params)

    create_logger(**logger_params)
    _print_config(args, tester_params)

    tester = TSPTester_LIB(model_params=MODEL_PARAMS, tester_params=tester_params)

    # copy source snapshot
    copy_all_src(get_result_folder())

    result = tester.run_lib()
    payload = build_result_payload(args, tester_params, result)
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
        "Primary metric for official evaluation: avg_aug_gap "
        "(computed from augmented inference when public/private optima are available)."
    )


##########################################################################################

if __name__ == "__main__":
    main()
