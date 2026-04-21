import copy
from typing import Tuple

import torch
from torch.nn.utils import clip_grad_norm_

from TSPTester_LIB import TSPTester_LIB
from TSProblemDef import augment_xy_data_by_8_fold


class TSPTester_EAS(TSPTester_LIB):
    def __init__(self, model_params, tester_params):
        super().__init__(model_params=model_params, tester_params=tester_params)

        self.base_model_state = copy.deepcopy(self.model.state_dict())

        eas_params, eas_param_names = self.model.get_eas_parameters(self.tester_params['eas_param_group'])
        self.eas_param_names = eas_param_names
        eas_param_total = sum(param.nelement() for param in eas_params)
        self.logger.info(
            "EAS enabled: steps={}, lr={:.2e}, optimizer={}, target={}, trainable_params={}".format(
                self.tester_params['eas_steps'],
                self.tester_params['eas_lr'],
                self.tester_params['eas_optimizer'],
                self.tester_params['eas_param_group'],
                eas_param_total,
            )
        )
        self.logger.info("EAS parameter names: {}".format(eas_param_names))
        self.logger.info(
            "EAS optimizer config: momentum={}, weight_decay={}, grad_clip_norm={}, early_stop_patience={}, min_delta={}".format(
                self.tester_params['eas_momentum'],
                self.tester_params['eas_weight_decay'],
                self.tester_params['eas_grad_clip_norm'],
                self.tester_params['eas_early_stop_patience'],
                self.tester_params['eas_early_stop_min_delta'],
            )
        )

    def _restore_base_model(self):
        self.model.load_state_dict(self.base_model_state)
        self.model.set_eval_type(self.base_eval_type)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _set_eas_trainable_params(self):
        for param in self.model.parameters():
            param.requires_grad_(False)

        eas_params, _ = self.model.get_eas_parameters(self.tester_params['eas_param_group'])
        for param in eas_params:
            param.requires_grad_(True)
        return eas_params

    def _build_eas_optimizer(self, eas_params):
        optimizer_name = self.tester_params['eas_optimizer']
        lr = self.tester_params['eas_lr']
        weight_decay = self.tester_params['eas_weight_decay']

        if optimizer_name == 'sgd':
            return torch.optim.SGD(
                eas_params,
                lr=lr,
                momentum=self.tester_params['eas_momentum'],
                weight_decay=weight_decay,
            )
        if optimizer_name == 'adamw':
            return torch.optim.AdamW(
                eas_params,
                lr=lr,
                weight_decay=weight_decay,
            )

        raise ValueError(f"Unsupported EAS optimizer: {optimizer_name}")

    def _capture_eas_state(self):
        named_params = dict(self.model.named_parameters())
        return {
            param_name: named_params[param_name].detach().clone()
            for param_name in self.eas_param_names
        }

    def _restore_eas_state(self, state_dict):
        named_params = dict(self.model.named_parameters())
        with torch.no_grad():
            for param_name, param_value in state_dict.items():
                named_params[param_name].copy_(param_value)

    def _run_eas(self, env) -> Tuple[float, float]:
        eas_params = self._set_eas_trainable_params()
        optimizer = self._build_eas_optimizer(eas_params)

        best_no_aug_score = float('inf')
        best_aug_score = float('inf')
        best_state = self._capture_eas_state()
        eas_steps = self.tester_params['eas_steps']
        log_interval = max(1, self.tester_params['eas_log_interval'])
        grad_clip_norm = self.tester_params['eas_grad_clip_norm']
        early_stop_patience = self.tester_params['eas_early_stop_patience']
        early_stop_min_delta = self.tester_params['eas_early_stop_min_delta']
        no_improvement_steps = 0

        self.model.set_eval_type('softmax')
        self.model.train()

        for step in range(eas_steps):
            optimizer.zero_grad(set_to_none=True)

            reward, prob_list = self._rollout(env, collect_prob=True, lib_mode=True, no_grad=False)
            advantage = reward - reward.float().mean(dim=1, keepdim=True)
            log_prob = prob_list.clamp_min(1e-12).log().sum(dim=2)
            loss = (-advantage * log_prob).mean()
            loss.backward()

            grad_norm = None
            if grad_clip_norm > 0:
                grad_norm = clip_grad_norm_(eas_params, grad_clip_norm)

            optimizer.step()

            tour_lengths = -reward.detach()
            best_len_per_aug = tour_lengths.min(dim=1).values
            current_no_aug_score = float(best_len_per_aug[0].item())
            current_aug_score = float(best_len_per_aug.min().item())

            improved = current_aug_score < (best_aug_score - early_stop_min_delta)
            if improved:
                best_no_aug_score = current_no_aug_score
                best_aug_score = current_aug_score
                best_state = self._capture_eas_state()
                no_improvement_steps = 0
            else:
                no_improvement_steps += 1

            if (step + 1) == 1 or (step + 1) == eas_steps or (step + 1) % log_interval == 0:
                grad_norm_str = "n/a" if grad_norm is None else f"{float(grad_norm):.4f}"
                self.logger.info(
                    "EAS step {}/{}: loss {:.6f}, sampled_no_aug {:.3f}, sampled_aug {:.3f}, best_aug {:.3f}, grad_norm {}, patience {}/{}".format(
                        step + 1,
                        eas_steps,
                        loss.item(),
                        current_no_aug_score,
                        current_aug_score,
                        best_aug_score,
                        grad_norm_str,
                        no_improvement_steps,
                        early_stop_patience if early_stop_patience > 0 else "off",
                    )
                )

            if early_stop_patience > 0 and no_improvement_steps >= early_stop_patience:
                self.logger.info(
                    "EAS early stop at step {}/{} after {} non-improving updates.".format(
                        step + 1,
                        eas_steps,
                        no_improvement_steps,
                    )
                )
                break

        self._restore_eas_state(best_state)

        self.model.set_eval_type(self.base_eval_type)
        self.model.eval()
        return best_no_aug_score, best_aug_score

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
        if self.tester_params['eas_steps'] <= 0:
            return self._evaluate(env)
        eas_no_aug_score, eas_aug_score = self._run_eas(env)

        no_aug_score, aug_score = self._evaluate(env)
        no_aug_score = min(no_aug_score, eas_no_aug_score)
        aug_score = min(aug_score, eas_aug_score)
        self.logger.info(
            "EAS incumbent best merged with greedy eval: no_aug {:.3f}, aug {:.3f}".format(
                no_aug_score,
                aug_score,
            )
        )
        return no_aug_score, aug_score
