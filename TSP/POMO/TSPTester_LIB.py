import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from TSPEnv import TSPEnv as Env, Step_State
from TSPModel import TSPModel as Model

from TSProblemDef import augment_xy_data_by_8_fold

from tsplib_utils import TSPLIBReader, tsplib_cost


def _normalize_to_unit_square(node_xy: torch.Tensor) -> torch.Tensor:
    """Normalize to [0,1] with uniform scaling (same style as ICAM script)."""
    xy_max = torch.max(node_xy, dim=1, keepdim=True).values
    xy_min = torch.min(node_xy, dim=1, keepdim=True).values
    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
    ratio[ratio == 0] = 1
    return (node_xy - xy_min) / ratio.expand(-1, 1, 2)


def _compute_dist_matrix(coords: torch.Tensor, edge_weight_type: str = 'EUC_2D'):
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dists = (diff ** 2).sum(dim=2).sqrt()
    if edge_weight_type == 'CEIL_2D':
        dists = torch.ceil(dists)
    elif edge_weight_type == 'EUC_2D':
        dists = torch.floor(dists + 0.5)
    return dists.detach().cpu().tolist()


def _two_opt(tour, dist_matrix, max_iter: int = 10000):
    tour = list(tour)
    n = len(tour)
    improved = True
    iter_count = 0

    while improved and iter_count < max_iter:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (
                    dist_matrix[a][c]
                    + dist_matrix[b][d]
                    - dist_matrix[a][b]
                    - dist_matrix[c][d]
                )
                if delta < -1e-6:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
                    break
            if improved:
                break
        iter_count += 1

    return tour


def _tour_length(tour, dist_matrix):
    n = len(tour)
    return sum(dist_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))


@dataclass
class LibResult:
    instances: List[str]
    optimal: List[Optional[float]]
    problem_size: List[int]
    no_aug_score: List[float]
    aug_score: List[float]
    no_aug_gap: List[Optional[float]]
    aug_gap: List[Optional[float]]
    total_instance_num: int = 0
    solved_instance_num: int = 0

    @staticmethod
    def _mean_valid(values: List[Optional[float]]) -> Optional[float]:
        valid_values = [value for value in values if value is not None]
        if not valid_values:
            return None
        return float(np.mean(valid_values))

    @property
    def avg_no_aug_gap(self) -> Optional[float]:
        return self._mean_valid(self.no_aug_gap)

    @property
    def avg_aug_gap(self) -> Optional[float]:
        return self._mean_valid(self.aug_gap)

    def to_dict(self) -> Dict[str, object]:
        return {
            "instances": self.instances,
            "optimal": self.optimal,
            "problem_size": self.problem_size,
            "no_aug_score": self.no_aug_score,
            "aug_score": self.aug_score,
            "no_aug_gap": self.no_aug_gap,
            "aug_gap": self.aug_gap,
            "total_instance_num": self.total_instance_num,
            "solved_instance_num": self.solved_instance_num,
            "avg_no_aug_gap": self.avg_no_aug_gap,
            "avg_aug_gap": self.avg_aug_gap,
        }


class TSPTester_LIB:
    def __init__(self, model_params, tester_params):
        self.model_params = model_params
        self.tester_params = tester_params

        self.logger = getLogger('root')

        use_cuda = self.tester_params['use_cuda']
        if use_cuda:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        self.model = Model(**self.model_params).to(self.device)

        checkpoint_fullname = tester_params.get('checkpoint_path')
        if checkpoint_fullname is None:
            model_load = tester_params['model_load']
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        total = sum([param.nelement() for param in self.model.parameters()])
        self.base_eval_type = self.model.model_params['eval_type']
        self.logger.info("Model loaded from: {}".format(checkpoint_fullname))
        self.logger.info("Number of parameters: %.2fM" % (total / 1e6))

    def run_lib(self) -> LibResult:
        filename = self.tester_params['filename']
        scale_range_all = self.tester_params.get('scale_range_all', [[0, 1000]])
        detailed_log = self.tester_params.get('detailed_log', False)

        start_time_all = time.time()
        all_instance_num = 0
        solved_instance_num = 0

        result = LibResult(
            instances=[],
            optimal=[],
            problem_size=[],
            no_aug_score=[],
            aug_score=[],
            no_aug_gap=[],
            aug_gap=[],
        )

        for scale_range in scale_range_all:
            self.logger.info("#################  Test scale range: {}  #################".format(scale_range))

            for root, _, files in os.walk(filename):
                for file in files:
                    if not file.endswith('.tsp'):
                        continue

                    full_path = os.path.join(root, file)
                    name, dimension, locs, ew_type = TSPLIBReader(full_path)

                    all_instance_num += 1

                    if name is None:
                        self.logger.info(f"Skip (unsupported or invalid TSPLIB): {full_path}")
                        continue

                    if not (scale_range[0] <= dimension < scale_range[1]):
                        continue

                    optimal = tsplib_cost.get(name, None)
                    if optimal is None:
                        self.logger.info(
                            f"Optimal not found for {name}. "
                            "Will report scores but leave gap fields empty."
                        )

                    self.logger.info("===============================================================")
                    self.logger.info("Instance name: {}, problem_size: {}, EDGE_WEIGHT_TYPE: {}".format(name, dimension, ew_type))

                    coords_orig_np = np.array(locs, dtype=np.float32)
                    coords_orig = torch.from_numpy(coords_orig_np).to(self.device)
                    node_coord = coords_orig[None, :, :]

                    nodes_xy_normalized = _normalize_to_unit_square(node_coord)

                    try:
                        no_aug_score, aug_score = self._test_one_instance(
                            nodes_xy_normalized=nodes_xy_normalized,
                            coords_orig=coords_orig,
                            ew_type=ew_type,
                        )
                    except Exception as e:
                        self.logger.exception(f"Failed on instance {name}: {e}")
                        continue

                    solved_instance_num += 1

                    if optimal is None:
                        no_aug_gap = None
                        aug_gap = None
                    else:
                        no_aug_gap = (no_aug_score - optimal) / optimal * 100
                        aug_gap = (aug_score - optimal) / optimal * 100

                    result.instances.append(name)
                    result.optimal.append(float(optimal) if optimal is not None else None)
                    result.problem_size.append(int(dimension))
                    result.no_aug_score.append(float(no_aug_score))
                    result.aug_score.append(float(aug_score))
                    result.no_aug_gap.append(float(no_aug_gap) if no_aug_gap is not None else None)
                    result.aug_gap.append(float(aug_gap) if aug_gap is not None else None)

                    if optimal is None:
                        self.logger.info(
                            "no public optimum. no_aug: {:.3f}, aug: {:.3f}".format(
                                no_aug_score, aug_score
                            )
                        )
                    else:
                        self.logger.info(
                            "optimal: {:.3f}, no_aug: {:.3f} (gap {:.3f}%), aug: {:.3f} (gap {:.3f}%)".format(
                                optimal, no_aug_score, no_aug_gap, aug_score, aug_gap
                            )
                        )

        end_time_all = time.time()
        result.total_instance_num = all_instance_num
        result.solved_instance_num = solved_instance_num

        self.logger.info("=========================== Summary ===========================")
        self.logger.info(
            "All done, solved instance number: {}/{}, total time: {:.2f}s".format(
                solved_instance_num, all_instance_num, end_time_all - start_time_all
            )
        )

        if solved_instance_num > 0 and result.avg_aug_gap is not None:
            self.logger.info(
                "Avg gap(no aug): {:.3f}%, Avg gap(aug): {:.3f}%".format(
                    result.avg_no_aug_gap,
                    result.avg_aug_gap,
                )
            )
        elif solved_instance_num > 0:
            self.logger.info(
                "Avg gap unavailable because public optimal tour lengths were not provided "
                "for the evaluated instances."
            )

        if detailed_log:
            self.logger.info("===============================================================")
            self.logger.info("instance: {}".format(result.instances))
            self.logger.info("optimal: {}".format(result.optimal))
            self.logger.info("problem_size: {}".format(result.problem_size))
            self.logger.info("no_aug_score: {}".format(result.no_aug_score))
            self.logger.info("aug_score: {}".format(result.aug_score))
            self.logger.info("no_aug_gap: {}".format(result.no_aug_gap))
            self.logger.info("aug_gap: {}".format(result.aug_gap))

        return result

    def _build_env(self, problems: torch.Tensor, coords_orig: torch.Tensor, ew_type: str) -> Env:
        effective_batch = problems.size(0)
        problem_size = problems.size(1)

        env = Env(problem_size=problem_size, pomo_size=problem_size)
        env.batch_size = effective_batch
        env.problems = problems.to(self.device)
        env.BATCH_IDX = torch.arange(effective_batch, device=self.device)[:, None].expand(effective_batch, env.pomo_size)
        env.POMO_IDX = torch.arange(env.pomo_size, device=self.device)[None, :].expand(effective_batch, env.pomo_size)

        env.original_node_xy_lib = coords_orig[None, :, :]
        env.edge_weight_type = ew_type
        return env

    def _rollout(self, env: Env, collect_prob: bool, lib_mode: bool, no_grad: bool):
        context = torch.no_grad() if no_grad else nullcontext()
        with context:
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            prob_list = []
            while not done:
                selected, prob = self.model(state)
                state, reward, done = env.step(selected, lib_mode=lib_mode)
                if collect_prob:
                    prob_list.append(prob[:, :, None])

        prob_tensor = None
        if collect_prob:
            prob_tensor = torch.cat(prob_list, dim=2)
        return reward, prob_tensor

    def _reset_env_for_redecode(self, env: Env):
        env.selected_count = 0
        env.current_node = None
        env.selected_node_list = torch.zeros(
            (env.batch_size, env.pomo_size, 0),
            dtype=torch.long,
            device=env.problems.device,
        )
        env.step_state = Step_State(BATCH_IDX=env.BATCH_IDX, POMO_IDX=env.POMO_IDX)
        env.step_state.ninf_mask = torch.zeros(
            (env.batch_size, env.pomo_size, env.problem_size),
            device=env.problems.device,
        )

    def _evaluate(self, env: Env) -> Tuple[float, float]:
        self.model.eval()
        num_samples = max(1, int(self.tester_params.get('num_samples', 1)))
        enable_2opt = self.tester_params.get('enable_2opt', False)
        dist_matrix = None
        if enable_2opt:
            dist_matrix = _compute_dist_matrix(env.original_node_xy_lib[0], env.edge_weight_type)

        all_sample_best = []
        saved_eval_type = self.model.model_params['eval_type']

        with torch.no_grad():
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)

            for sample_id in range(num_samples):
                self.model.set_eval_type(
                    'softmax' if num_samples > 1 and sample_id < num_samples - 1 else 'argmax'
                )
                self._reset_env_for_redecode(env)

                state, reward, done = env.pre_step()
                while not done:
                    selected, _ = self.model(state)
                    state, reward, done = env.step(selected, lib_mode=True)

                tour_lengths = -reward
                if enable_2opt:
                    batch_best_lengths = []
                    best_pomo_idx = tour_lengths.argmin(dim=1)
                    for batch_idx in range(env.batch_size):
                        tour = env.selected_node_list[batch_idx, best_pomo_idx[batch_idx].item()]
                        optimized_tour = _two_opt(tour.detach().cpu().tolist(), dist_matrix)
                        batch_best_lengths.append(_tour_length(optimized_tour, dist_matrix))
                    best_scores = torch.tensor(batch_best_lengths, dtype=torch.float32)
                else:
                    best_scores = tour_lengths.min(dim=1).values.detach().cpu()

                all_sample_best.append(best_scores)

        self.model.set_eval_type(saved_eval_type)

        all_sample_best = torch.stack(all_sample_best, dim=0)
        no_aug_score = all_sample_best[:, 0].min().item()
        aug_score = all_sample_best.min().item()
        return float(no_aug_score), float(aug_score)

    def _test_one_instance(self, nodes_xy_normalized: torch.Tensor, coords_orig: torch.Tensor, ew_type: str) -> Tuple[float, float]:
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
            if aug_factor != 8:
                raise NotImplementedError('Only aug_factor=8 is supported.')
        else:
            aug_factor = 1

        problems = nodes_xy_normalized
        if aug_factor > 1:
            problems = augment_xy_data_by_8_fold(problems)
        env = self._build_env(problems=problems, coords_orig=coords_orig, ew_type=ew_type)

        no_aug_score, aug_score = self._evaluate(env)
        return no_aug_score, aug_score
