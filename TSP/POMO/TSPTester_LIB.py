import os
import time
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from TSPEnv import TSPEnv as Env
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

        self.model = Model(**self.model_params)

        checkpoint_fullname = tester_params.get('checkpoint_path')
        if checkpoint_fullname is None:
            model_load = tester_params['model_load']
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        load_result = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if load_result.missing_keys:
            self.logger.info("Model load missing keys: {}".format(load_result.missing_keys))
        if load_result.unexpected_keys:
            self.logger.info("Model load unexpected keys: {}".format(load_result.unexpected_keys))
        self.base_model_state_dict = {
            key: value.detach().clone().cpu()
            for key, value in self.model.state_dict().items()
        }

        total = sum([param.nelement() for param in self.model.parameters()])
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

        problems = problems.to(self.device)
        coords_orig = coords_orig.to(self.device)

        if self.tester_params.get('eas_enable', False):
            self._reset_model_to_base()
            self._run_eas(problems)
        else:
            self.model.decoder.disable_eas()

        if self.tester_params.get('sgbs_enable', False):
            best_len_per_aug = self._run_sgbs(problems, coords_orig, ew_type)
        else:
            best_len_per_aug = self._run_pomo_greedy(problems, coords_orig, ew_type)

        no_aug_score = best_len_per_aug[0].item()
        aug_score = best_len_per_aug.min(dim=0).values.item()

        return float(no_aug_score), float(aug_score)

    def _reset_model_to_base(self):
        self.model.load_state_dict(self.base_model_state_dict, strict=True)
        self.model.decoder.disable_eas()

    def _make_env(self, problems: torch.Tensor, coords_orig: Optional[torch.Tensor] = None,
                  ew_type: Optional[str] = None) -> Env:
        effective_batch = problems.size(0)
        problem_size = problems.size(1)

        env = Env(problem_size=problem_size, pomo_size=problem_size)
        env.batch_size = effective_batch
        env.problems = problems.to(self.device)
        env.BATCH_IDX = torch.arange(effective_batch, device=self.device)[:, None].expand(effective_batch, env.pomo_size)
        env.POMO_IDX = torch.arange(env.pomo_size, device=self.device)[None, :].expand(effective_batch, env.pomo_size)

        if coords_orig is not None:
            env.original_node_xy_lib = coords_orig[None, :, :].to(self.device)
            env.edge_weight_type = ew_type

        return env

    def _run_pomo_greedy(self, problems: torch.Tensor, coords_orig: torch.Tensor, ew_type: str) -> torch.Tensor:
        env = self._make_env(problems, coords_orig=coords_orig, ew_type=ew_type)

        self.model.eval()
        with torch.no_grad():
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = env.step(selected, lib_mode=True)

        tour_lengths = -reward
        return tour_lengths.min(dim=1).values

    def _run_eas(self, problems: torch.Tensor):
        eas_steps = self.tester_params.get('eas_steps', 100)
        if eas_steps <= 0:
            return

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.decoder.reset_eas_adapter()
        self.model.decoder.enable_eas()
        for param in self.model.decoder.eas_adapter.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(
            self.model.decoder.eas_adapter.parameters(),
            lr=self.tester_params.get('eas_lr', 1e-3),
            weight_decay=self.tester_params.get('eas_weight_decay', 0.0),
        )
        entropy_beta = self.tester_params.get('eas_entropy_beta', 0.0)

        self.model.train()
        for _ in range(eas_steps):
            loss = self._eas_one_step(problems, entropy_beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for param in self.model.parameters():
            param.requires_grad = True
        self.model.eval()

    def _eas_one_step(self, problems: torch.Tensor, entropy_beta: float) -> torch.Tensor:
        env = self._make_env(problems)
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)

        batch_size = env.batch_size
        pomo_size = env.pomo_size
        prob_list = torch.zeros(size=(batch_size, pomo_size, 0), device=self.device)
        entropy_list = torch.zeros(size=(batch_size, pomo_size, 0), device=self.device)

        state, reward, done = env.pre_step()
        while not done:
            selected, prob = self.model(state)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            if self.model.last_probs is None:
                entropy = torch.zeros_like(prob)
            else:
                probs = self.model.last_probs.clamp_min(1e-12)
                entropy = -(probs * probs.log()).sum(dim=2)
            entropy_list = torch.cat((entropy_list, entropy[:, :, None]), dim=2)

        pomo_size = reward.size(1)
        if pomo_size > 1:
            baseline = (reward.sum(dim=1, keepdim=True) - reward) / (pomo_size - 1)
        else:
            baseline = reward.float().mean(dim=1, keepdims=True)
        advantage = reward - baseline
        log_prob = prob_list.clamp_min(1e-12).log().sum(dim=2)
        entropy = entropy_list.sum(dim=2)
        loss = -advantage * log_prob - entropy_beta * entropy
        return loss.mean()

    def _run_sgbs(self, problems: torch.Tensor, coords_orig: torch.Tensor, ew_type: str) -> torch.Tensor:
        batch_size = problems.size(0)
        problem_size = problems.size(1)
        beam_width = max(1, self.tester_params.get('sgbs_beam_width', 4))
        expand_width = max(1, self.tester_params.get('sgbs_expand_width', beam_width))
        max_beam_count = self.tester_params.get('sgbs_max_beam_count', problem_size * beam_width)
        max_beam_count = max(1, min(max_beam_count, problem_size * beam_width))
        use_simulation = self.tester_params.get('sgbs_simulation_enable', True)

        self.model.eval()
        with torch.no_grad():
            env = self._make_env(problems)
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)

            first_nodes = torch.arange(problem_size, device=self.device)[None, :].expand(batch_size, problem_size)
            current_node = first_nodes.clone()
            selected_list = current_node[:, :, None]
            ninf_mask = torch.zeros((batch_size, problem_size, problem_size), device=self.device)
            batch_idx = torch.arange(batch_size, device=self.device)[:, None].expand(batch_size, problem_size)
            beam_idx = torch.arange(problem_size, device=self.device)[None, :].expand(batch_size, problem_size)
            ninf_mask[batch_idx, beam_idx, current_node] = float('-inf')
            beam_log_prob = torch.zeros((batch_size, problem_size), device=self.device)

            while selected_list.size(2) < problem_size:
                probs = self.model.get_action_probs(current_node, ninf_mask, first_nodes=first_nodes)
                remaining = problem_size - selected_list.size(2)
                cur_expand_width = min(expand_width, remaining)
                top_prob, top_node = probs.topk(cur_expand_width, dim=2)

                parent_beam_count = current_node.size(1)
                candidate_count = parent_beam_count * cur_expand_width
                parent_idx = torch.arange(parent_beam_count, device=self.device)[None, :, None].expand(
                    batch_size, parent_beam_count, cur_expand_width
                ).reshape(batch_size, candidate_count)

                candidate_first = first_nodes.gather(dim=1, index=parent_idx)
                candidate_current = top_node.reshape(batch_size, candidate_count)
                candidate_log_prob = beam_log_prob.gather(dim=1, index=parent_idx)
                candidate_log_prob = candidate_log_prob + top_prob.clamp_min(1e-12).log().reshape(batch_size, candidate_count)

                selected_len = selected_list.size(2)
                candidate_selected = selected_list.gather(
                    dim=1,
                    index=parent_idx[:, :, None].expand(batch_size, candidate_count, selected_len),
                )
                candidate_selected = torch.cat((candidate_selected, candidate_current[:, :, None]), dim=2)

                candidate_mask = ninf_mask.gather(
                    dim=1,
                    index=parent_idx[:, :, None].expand(batch_size, candidate_count, problem_size),
                ).clone()
                candidate_batch_idx = torch.arange(batch_size, device=self.device)[:, None].expand(batch_size, candidate_count)
                candidate_beam_idx = torch.arange(candidate_count, device=self.device)[None, :].expand(batch_size, candidate_count)
                candidate_mask[candidate_batch_idx, candidate_beam_idx, candidate_current] = float('-inf')

                if candidate_selected.size(2) == problem_size:
                    final_lengths = self._tour_lengths(candidate_selected, coords_orig, ew_type=ew_type)
                    return final_lengths.min(dim=1).values

                keep_count = min(max_beam_count, candidate_count)
                if use_simulation:
                    guide_cost = self._sgbs_greedy_rollout_cost(
                        problems,
                        candidate_first,
                        candidate_current,
                        candidate_selected,
                        candidate_mask,
                    )
                    keep_idx = guide_cost.topk(keep_count, dim=1, largest=False).indices
                else:
                    keep_idx = candidate_log_prob.topk(keep_count, dim=1).indices

                first_nodes = candidate_first.gather(dim=1, index=keep_idx)
                current_node = candidate_current.gather(dim=1, index=keep_idx)
                beam_log_prob = candidate_log_prob.gather(dim=1, index=keep_idx)
                selected_list = candidate_selected.gather(
                    dim=1,
                    index=keep_idx[:, :, None].expand(batch_size, keep_count, candidate_selected.size(2)),
                )
                ninf_mask = candidate_mask.gather(
                    dim=1,
                    index=keep_idx[:, :, None].expand(batch_size, keep_count, problem_size),
                )

        final_lengths = self._tour_lengths(selected_list, coords_orig, ew_type=ew_type)
        return final_lengths.min(dim=1).values

    def _sgbs_greedy_rollout_cost(self, problems: torch.Tensor, first_nodes: torch.Tensor,
                                  current_node: torch.Tensor, selected_list: torch.Tensor,
                                  ninf_mask: torch.Tensor) -> torch.Tensor:
        problem_size = problems.size(1)
        rollout_first = first_nodes.clone()
        rollout_current = current_node.clone()
        rollout_selected = selected_list.clone()
        rollout_mask = ninf_mask.clone()
        batch_size = rollout_current.size(0)

        while rollout_selected.size(2) < problem_size:
            probs = self.model.get_action_probs(rollout_current, rollout_mask, first_nodes=rollout_first)
            selected = probs.argmax(dim=2)
            beam_count = selected.size(1)
            rollout_selected = torch.cat((rollout_selected, selected[:, :, None]), dim=2)
            batch_idx = torch.arange(batch_size, device=self.device)[:, None].expand(batch_size, beam_count)
            beam_idx = torch.arange(beam_count, device=self.device)[None, :].expand(batch_size, beam_count)
            rollout_mask[batch_idx, beam_idx, selected] = float('-inf')
            rollout_current = selected

        return self._tour_lengths(rollout_selected, problems)

    def _tour_lengths(self, selected_list: torch.Tensor, node_xy: torch.Tensor,
                      ew_type: Optional[str] = None) -> torch.Tensor:
        batch_size, beam_count, problem_size = selected_list.size()
        if node_xy.dim() == 2:
            coords = node_xy[None, None, :, :].expand(batch_size, beam_count, problem_size, 2)
        else:
            coords = node_xy[:, None, :, :].expand(batch_size, beam_count, problem_size, 2)

        gathering_index = selected_list[:, :, :, None].expand(batch_size, beam_count, problem_size, 2)
        ordered_seq = coords.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()

        if ew_type == 'CEIL_2D':
            segment_lengths = torch.ceil(segment_lengths)
        elif ew_type == 'EUC_2D':
            segment_lengths = torch.floor(segment_lengths + 0.5)

        return segment_lengths.sum(2)
