import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_CHECKPOINT = "./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt"
DEFAULT_DATA_PATH = "../data/val"


STAGE_CONFIGS = [
    {
        "idx": 2,
        "name": "stage2_scale_dpo",
        "epochs": 90,
        "train_episodes": 2048,
        "curriculum_problem_sizes": [150, 200, 300],
        "curriculum_stage_epochs": [20, 30, 40],
        "batch_schedule": "100:32,150:32,200:24,300:12",
        "current_stage_mix_weight": 0.70,
        "previous_stage_mix_weight": 0.20,
        "base_replay_mix_weight": 0.10,
        "preference_beta": 0.10,
        "preference_pair_k": 4,
        "preference_gap_weight_power": 1.0,
        "rl_loss_weight": 0.20,
        "lr": 5e-5,
        "weight_decay": 1e-6,
        "milestones": [45, 55],
        "scheduler_gamma": 0.2,
    },
    {
        "idx": 3,
        "name": "stage3_ref_refresh_200_300_500",
        "epochs": 80,
        "train_episodes": 2048,
        "curriculum_problem_sizes": [200, 300, 500],
        "curriculum_stage_epochs": [20, 30, 30],
        "batch_schedule": "100:24,150:20,200:16,300:8,500:2",
        "current_stage_mix_weight": 0.65,
        "previous_stage_mix_weight": 0.25,
        "base_replay_mix_weight": 0.10,
        "preference_beta": 0.07,
        "preference_pair_k": 6,
        "preference_gap_weight_power": 1.5,
        "rl_loss_weight": 0.10,
        "lr": 2e-5,
        "weight_decay": 1e-6,
        "milestones": [50, 65],
        "scheduler_gamma": 0.3,
    },
]


def str2bool(value):
    if isinstance(value, bool):
        return value

    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def bool_arg(value):
    return "true" if value else "false"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 2 and Stage 3 post-training after the 3000-epoch POMO "
            "checkpoint, which is treated as the completed Stage 1 model."
        )
    )
    parser.add_argument("--base_checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--result_root", default=None,
                        help="Directory that will contain stage2/stage3 subdirectories.")
    parser.add_argument("--start_stage", type=int, default=2, choices=[2, 3])
    parser.add_argument("--stop_stage", type=int, default=3, choices=[2, 3])
    parser.add_argument("--stage2_checkpoint", default=None,
                        help="Existing Stage 2 checkpoint, required when starting directly from Stage 3.")
    parser.add_argument("--train_episodes", type=int, default=None,
                        help="Override train episodes per epoch for every stage.")
    parser.add_argument("--min_train_batch_size", type=int, default=1)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--cuda_device_num", type=int, default=0)
    parser.add_argument("--dry_run", type=str2bool, default=False,
                        help="Print commands without running training.")
    parser.add_argument("--eval_after_stage", type=str2bool, default=False,
                        help="Run standard augmented validation after each completed stage.")
    parser.add_argument("--eval_data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--eval_aug_factor", type=int, default=8)
    return parser


def resolve_result_root(args):
    if args.result_root is not None:
        return os.path.abspath(args.result_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.abspath(
        os.path.join(SCRIPT_DIR, "result", f"{timestamp}_post_train_stage2_stage3")
    )


def get_stage_config(stage_idx):
    for stage_config in STAGE_CONFIGS:
        if stage_config["idx"] == stage_idx:
            return stage_config
    raise ValueError("Unsupported post-training stage: {}".format(stage_idx))


def stage_result_dir(result_root, stage_config):
    return os.path.join(
        result_root,
        "stage{}_{}".format(stage_config["idx"], stage_config["name"]),
    )


def stage_final_checkpoint(result_root, stage_config):
    return os.path.join(
        stage_result_dir(result_root, stage_config),
        "checkpoint-{}.pt".format(stage_config["epochs"]),
    )


def existing_stage_checkpoint(args, result_root, stage_idx):
    override = getattr(args, f"stage{stage_idx}_checkpoint", None)
    if override:
        return os.path.abspath(override)

    stage_config = get_stage_config(stage_idx)
    checkpoint_path = stage_final_checkpoint(result_root, stage_config)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Missing checkpoint for completed Stage {}: {}. "
            "Pass --stage{}_checkpoint or use a --result_root that contains the stage output.".format(
                stage_idx,
                checkpoint_path,
                stage_idx,
            )
        )
    return checkpoint_path


def append_flag(command, name, value):
    command.append("--{}".format(name))
    if isinstance(value, bool):
        command.append(bool_arg(value))
    else:
        command.append(str(value))


def append_list_flag(command, name, values):
    command.append("--{}".format(name))
    command.extend(str(value) for value in values)


def build_stage_command(args, result_root, stage_config, init_checkpoint, reference_checkpoint):
    stage_dir = stage_result_dir(result_root, stage_config)
    command = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "post_train_preference.py"),
    ]

    append_flag(command, "stage_name", stage_config["name"])
    append_flag(command, "result_dir", stage_dir)
    append_flag(command, "init_checkpoint", init_checkpoint)
    append_flag(command, "reference_checkpoint", reference_checkpoint)
    append_flag(command, "epochs", stage_config["epochs"])
    append_flag(command, "train_episodes", args.train_episodes or stage_config["train_episodes"])
    append_flag(command, "min_train_batch_size", args.min_train_batch_size)
    append_flag(command, "batch_schedule", stage_config["batch_schedule"])
    append_list_flag(command, "curriculum_problem_sizes", stage_config["curriculum_problem_sizes"])
    append_list_flag(command, "curriculum_stage_epochs", stage_config["curriculum_stage_epochs"])
    append_flag(command, "current_stage_mix_weight", stage_config["current_stage_mix_weight"])
    append_flag(command, "previous_stage_mix_weight", stage_config["previous_stage_mix_weight"])
    append_flag(command, "base_replay_mix_weight", stage_config["base_replay_mix_weight"])
    append_flag(command, "preference_beta", stage_config["preference_beta"])
    append_flag(command, "preference_pair_k", stage_config["preference_pair_k"])
    append_flag(command, "preference_gap_weight_power", stage_config["preference_gap_weight_power"])
    append_flag(command, "rl_loss_weight", stage_config["rl_loss_weight"])
    append_flag(command, "lr", stage_config["lr"])
    append_flag(command, "weight_decay", stage_config["weight_decay"])
    append_list_flag(command, "milestones", stage_config["milestones"])
    append_flag(command, "scheduler_gamma", stage_config["scheduler_gamma"])
    append_flag(command, "use_reference_candidate_pool", True)
    append_flag(command, "use_cuda", args.use_cuda)
    append_flag(command, "cuda_device_num", args.cuda_device_num)
    return command


def build_eval_command(args, checkpoint_path, stage_dir):
    output_json = os.path.join(stage_dir, "eval_aug{}.json".format(args.eval_aug_factor))
    return [
        sys.executable,
        os.path.join(SCRIPT_DIR, "test.py"),
        "--data_path",
        args.eval_data_path,
        "--checkpoint_path",
        checkpoint_path,
        "--use_cuda",
        bool_arg(args.use_cuda),
        "--cuda_device_num",
        str(args.cuda_device_num),
        "--augmentation_enable",
        "true",
        "--aug_factor",
        str(args.eval_aug_factor),
        "--detailed_log",
        "false",
        "--output_json",
        output_json,
    ], output_json


def run_command(command, dry_run):
    printable = shlex.join(command)
    print(printable, flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def main():
    args = build_parser().parse_args()
    if args.start_stage > args.stop_stage:
        raise ValueError("--start_stage must be <= --stop_stage")

    result_root = resolve_result_root(args)
    if not args.dry_run:
        os.makedirs(result_root, exist_ok=True)

    completed_checkpoints = {
        1: os.path.abspath(args.base_checkpoint),
    }
    for stage_idx in range(2, args.start_stage):
        completed_checkpoints[stage_idx] = existing_stage_checkpoint(args, result_root, stage_idx)

    print("result_root: {}".format(result_root), flush=True)

    for stage_config in STAGE_CONFIGS:
        stage_idx = stage_config["idx"]
        if not (args.start_stage <= stage_idx <= args.stop_stage):
            continue

        init_checkpoint = completed_checkpoints[stage_idx - 1]
        reference_checkpoint = init_checkpoint
        command = build_stage_command(
            args=args,
            result_root=result_root,
            stage_config=stage_config,
            init_checkpoint=init_checkpoint,
            reference_checkpoint=reference_checkpoint,
        )
        run_command(command, args.dry_run)

        final_checkpoint = stage_final_checkpoint(result_root, stage_config)
        completed_checkpoints[stage_idx] = final_checkpoint

        if args.eval_after_stage:
            eval_command, output_json = build_eval_command(
                args=args,
                checkpoint_path=final_checkpoint,
                stage_dir=stage_result_dir(result_root, stage_config),
            )
            run_command(eval_command, args.dry_run)
            if not args.dry_run and os.path.exists(output_json):
                with open(output_json, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                print(
                    "Stage {} avg_aug_gap: {}".format(
                        stage_idx,
                        payload.get("avg_aug_gap"),
                    ),
                    flush=True,
                )

    print("final_checkpoint: {}".format(completed_checkpoints[args.stop_stage]), flush=True)


if __name__ == "__main__":
    main()
