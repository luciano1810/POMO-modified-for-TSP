from typing import List, Tuple

import torch

from TSPTester_LIB import TSPTester_LIB
from TSProblemDef import augment_xy_data_by_8_fold
from utils.utils import create_progress_bar


class TSPTester_EAS(TSPTester_LIB):
    def __init__(self, model_params, tester_params):
        super().__init__(model_params=model_params, tester_params=tester_params)

        self.eas_params, self.eas_param_names = self.model.get_eas_parameters(
            self.tester_params['eas_param_group']
        )
        self.base_eas_state = self._capture_eas_state()
        self.selection_num_samples = max(
            1,
            int(
                self.tester_params.get(
                    'eas_selection_num_samples',
                    self.tester_params.get('num_samples', 1),
                )
            ),
        )
        self.selection_enable_2opt = self.tester_params.get(
            'eas_selection_enable_2opt',
            self.tester_params.get('enable_2opt', False),
        )
        eas_param_total = sum(param.nelement() for param in self.eas_params)
        self.logger.info(
            "EAS enabled: steps={}, restarts={}, optimizer={}, lr={:.2e}, grad_clip={}, "
            "patience={}, target={}, trainable_params={}".format(
                self.tester_params['eas_steps'],
                self.tester_params['eas_restarts'],
                self.tester_params['eas_optimizer'],
                self.tester_params['eas_lr'],
                self.tester_params['eas_grad_clip'],
                self.tester_params['eas_patience'],
                self.tester_params['eas_param_group'],
                eas_param_total,
            )
        )
        self.logger.info("EAS parameter names: {}".format(self.eas_param_names))
        self.logger.info(
            "EAS checkpoint selection metric: num_samples={}, enable_2opt={}, record_interval={}".format(
                self.selection_num_samples,
                self.selection_enable_2opt,
                self.tester_params['eas_record_interval'],
            )
        )

    def _capture_eas_state(self) -> List[torch.Tensor]:
        return [param.detach().clone() for param in self.eas_params]

    def _load_eas_state(self, eas_state: List[torch.Tensor]):
        with torch.no_grad():
            for param, saved_param in zip(self.eas_params, eas_state):
                param.copy_(saved_param)

    def _restore_base_model(self):
        self._load_eas_state(self.base_eas_state)
        self.model.set_eval_type(self.base_eval_type)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _set_eas_trainable_params(self):
        for param in self.model.parameters():
            param.requires_grad_(False)

        for param in self.eas_params:
            param.requires_grad_(True)
        return self.eas_params

    def _build_eas_optimizer(self, eas_params):
        optimizer_name = self.tester_params['eas_optimizer'].lower()
        lr = self.tester_params['eas_lr']
        weight_decay = self.tester_params['eas_weight_decay']

        if optimizer_name == 'sgd':
            return torch.optim.SGD(
                eas_params,
                lr=lr,
                momentum=self.tester_params['eas_momentum'],
                weight_decay=weight_decay,
            )
        if optimizer_name == 'adam':
            return torch.optim.Adam(
                eas_params,
                lr=lr,
                weight_decay=weight_decay,
            )
        if optimizer_name == 'adamw':
            return torch.optim.AdamW(
                eas_params,
                lr=lr,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unsupported EAS optimizer: {optimizer_name}")

    @staticmethod
    def _split_restart_steps(total_steps: int, num_restarts: int) -> List[int]:
        effective_restarts = max(1, min(total_steps, num_restarts))
        base_steps = total_steps // effective_restarts
        remainder = total_steps % effective_restarts
        step_budgets = []
        for restart_idx in range(effective_restarts):
            steps = base_steps + (1 if restart_idx < remainder else 0)
            if steps > 0:
                step_budgets.append(steps)
        return step_budgets

    def _score_current_candidate(self, env) -> Tuple[float, float]:
        no_aug_score, aug_score = self._evaluate(
            env,
            num_samples=self.selection_num_samples,
            enable_2opt=self.selection_enable_2opt,
        )
        self.model.set_eval_type('softmax')
        self.model.train()
        return no_aug_score, aug_score

    def _compute_eas_loss(self, reward: torch.Tensor, prob_list: torch.Tensor) -> torch.Tensor:
        advantage = reward - reward.float().mean(dim=1, keepdim=True)
        log_prob = prob_list.clamp_min(1e-12).log().sum(dim=2)
        loss_type = self.tester_params['eas_loss_type']

        if loss_type == 'reinforce':
            return (-advantage * log_prob).mean()

        if loss_type == 'elite_reinforce':
            elite_ratio = float(self.tester_params['eas_elite_ratio'])
            elite_ratio = min(1.0, max(0.0, elite_ratio))
            elite_k = max(1, int(round(reward.size(1) * elite_ratio)))
            tour_lengths = -reward.detach()
            elite_indices = tour_lengths.topk(k=elite_k, dim=1, largest=False).indices
            elite_weights = torch.zeros_like(log_prob)
            elite_weights.scatter_(1, elite_indices, 1.0)
            elite_weights = elite_weights / elite_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (-(advantage * log_prob) * elite_weights).sum(dim=1).mean()

        raise ValueError(f"Unsupported EAS loss type: {loss_type}")

    @staticmethod
    def _is_better_candidate(
        no_aug_score: float,
        aug_score: float,
        best_no_aug_score: float,
        best_aug_score: float,
    ) -> bool:
        if aug_score < best_aug_score:
            return True
        if aug_score == best_aug_score and no_aug_score < best_no_aug_score:
            return True
        return False

    def _run_eas(self, env) -> Tuple[float, float]:
        eas_steps = self.tester_params['eas_steps']
        record_interval = max(1, self.tester_params['eas_record_interval'])
        log_interval = max(1, self.tester_params['eas_log_interval'])
        patience = max(0, self.tester_params['eas_patience'])
        grad_clip = float(self.tester_params['eas_grad_clip'])
        restart_step_budgets = self._split_restart_steps(
            total_steps=eas_steps,
            num_restarts=max(1, self.tester_params['eas_restarts']),
        )
        best_recorded_no_aug_score, best_recorded_aug_score = self._score_current_candidate(env)
        best_recorded_restart = 0
        best_recorded_step = 0
        best_recorded_state = self._capture_eas_state()
        self.logger.info(
            "EAS base incumbent before adaptation: select_no_aug {:.3f}, select_aug {:.3f}".format(
                best_recorded_no_aug_score,
                best_recorded_aug_score,
            )
        )

        for restart_idx, restart_steps in enumerate(restart_step_budgets, start=1):
            self._restore_base_model()
            eas_params = self._set_eas_trainable_params()
            optimizer = self._build_eas_optimizer(eas_params)
            records_without_improvement = 0

            self.model.set_eval_type('softmax')
            self.model.train()

            with create_progress_bar(
                total=restart_steps,
                desc='EAS {}/{}'.format(restart_idx, len(restart_step_budgets)),
                unit='step',
                leave=False,
                position=1,
            ) as progress_bar:
                for step in range(restart_steps):
                    optimizer.zero_grad(set_to_none=True)

                    reward, prob_list = self._rollout(env, collect_prob=True, lib_mode=True, no_grad=False)
                    loss = self._compute_eas_loss(reward=reward, prob_list=prob_list)
                    loss.backward()

                    grad_norm_value = None
                    if grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(eas_params, grad_clip)
                        grad_norm_value = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

                    optimizer.step()

                    tour_lengths = -reward.detach()
                    best_len_per_aug = tour_lengths.min(dim=1).values
                    sampled_no_aug_score = float(best_len_per_aug[0].item())
                    sampled_aug_score = float(best_len_per_aug.min().item())

                    should_record = ((step + 1) % record_interval == 0) or ((step + 1) == restart_steps)
                    if should_record:
                        candidate_no_aug_score, candidate_aug_score = self._score_current_candidate(env)
                        if self._is_better_candidate(
                            no_aug_score=candidate_no_aug_score,
                            aug_score=candidate_aug_score,
                            best_no_aug_score=best_recorded_no_aug_score,
                            best_aug_score=best_recorded_aug_score,
                        ):
                            best_recorded_no_aug_score = candidate_no_aug_score
                            best_recorded_aug_score = candidate_aug_score
                            best_recorded_restart = restart_idx
                            best_recorded_step = step + 1
                            best_recorded_state = self._capture_eas_state()
                            records_without_improvement = 0
                            self.logger.info(
                                "EAS candidate updated at restart {}/{} step {}/{}: "
                                "select_no_aug {:.3f}, select_aug {:.3f}, sampled_no_aug {:.3f}, sampled_aug {:.3f}".format(
                                    restart_idx,
                                    len(restart_step_budgets),
                                    step + 1,
                                    restart_steps,
                                    candidate_no_aug_score,
                                    candidate_aug_score,
                                    sampled_no_aug_score,
                                    sampled_aug_score,
                                )
                            )
                        else:
                            records_without_improvement += 1

                    if (step + 1) == 1 or (step + 1) == restart_steps or (step + 1) % log_interval == 0:
                        log_message = (
                            "EAS restart {}/{} step {}/{}: loss {:.6f}, sampled_no_aug {:.3f}, "
                            "sampled_aug {:.3f}".format(
                                restart_idx,
                                len(restart_step_budgets),
                                step + 1,
                                restart_steps,
                                loss.item(),
                                sampled_no_aug_score,
                                sampled_aug_score,
                            )
                        )
                        if grad_norm_value is not None:
                            log_message = "{}, grad_norm {:.3f}".format(log_message, grad_norm_value)
                        self.logger.info(log_message)

                    progress_bar.update(1)

                    if patience > 0 and records_without_improvement >= patience:
                        self.logger.info(
                            "EAS early stop at restart {}/{} step {}/{} after {} recorded checkpoints without improvement.".format(
                                restart_idx,
                                len(restart_step_budgets),
                                step + 1,
                                restart_steps,
                                patience,
                            )
                        )
                        break

        self._load_eas_state(best_recorded_state)
        if best_recorded_restart == 0:
            self.logger.info(
                "EAS final inference will keep the base checkpoint: select_no_aug {:.3f}, select_aug {:.3f}".format(
                    best_recorded_no_aug_score,
                    best_recorded_aug_score,
                )
            )
        else:
            self.logger.info(
                "EAS final inference will use recorded checkpoint at restart {}/{} step {}/{}: "
                "select_no_aug {:.3f}, select_aug {:.3f}".format(
                    best_recorded_restart,
                    len(restart_step_budgets),
                    best_recorded_step,
                    restart_step_budgets[best_recorded_restart - 1],
                    best_recorded_no_aug_score,
                    best_recorded_aug_score,
                )
            )
        self.model.set_eval_type(self.base_eval_type)
        self.model.eval()
        return best_recorded_no_aug_score, best_recorded_aug_score

    def _test_one_instance(self, nodes_xy_normalized, coords_orig, ew_type) -> Tuple[float, float]:
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

        self._restore_base_model()
        base_no_aug_score, base_aug_score = self._evaluate(env)
        if self.tester_params['eas_steps'] <= 0:
            return base_no_aug_score, base_aug_score
        eas_no_aug_score, eas_aug_score = self._run_eas(env)

        no_aug_score, aug_score = self._evaluate(env)
        no_aug_score = min(no_aug_score, base_no_aug_score)
        aug_score = min(aug_score, base_aug_score)
        no_aug_score = min(no_aug_score, eas_no_aug_score)
        aug_score = min(aug_score, eas_aug_score)
        self.logger.info(
            "EAS incumbent best merged with base + post-EAS eval: no_aug {:.3f}, aug {:.3f}".format(
                no_aug_score,
                aug_score,
            )
        )
        return no_aug_score, aug_score
