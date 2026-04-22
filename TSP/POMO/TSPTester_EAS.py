import copy
from typing import Tuple

import torch

from TSPTester_LIB import TSPTester_LIB
from TSProblemDef import augment_xy_data_by_8_fold


class TSPTester_EAS(TSPTester_LIB):
    def __init__(self, model_params, tester_params):
        super().__init__(model_params=model_params, tester_params=tester_params)

        self.base_model_state = copy.deepcopy(self.model.state_dict())

        eas_params, eas_param_names = self.model.get_eas_parameters(self.tester_params['eas_param_group'])
        eas_param_total = sum(param.nelement() for param in eas_params)
        self.logger.info(
            "EAS enabled: steps={}, lr={:.2e}, target={}, trainable_params={}".format(
                self.tester_params['eas_steps'],
                self.tester_params['eas_lr'],
                self.tester_params['eas_param_group'],
                eas_param_total,
            )
        )
        self.logger.info("EAS parameter names: {}".format(eas_param_names))

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

    def _run_eas(self, env) -> Tuple[float, float]:
        eas_params = self._set_eas_trainable_params()
        optimizer = torch.optim.SGD(eas_params, lr=self.tester_params['eas_lr'])

        best_no_aug_score = float('inf')
        best_aug_score = float('inf')
        eas_steps = self.tester_params['eas_steps']
        log_interval = max(1, self.tester_params['eas_log_interval'])

        self.model.set_eval_type('softmax')
        self.model.train()

        for step in range(eas_steps):
            optimizer.zero_grad(set_to_none=True)

            reward, prob_list = self._rollout(env, collect_prob=True, lib_mode=True, no_grad=False)
            advantage = reward - reward.float().mean(dim=1, keepdim=True)
            log_prob = prob_list.clamp_min(1e-12).log().sum(dim=2)
            loss = (-advantage * log_prob).mean()
            loss.backward()
            optimizer.step()

            tour_lengths = -reward.detach()
            best_len_per_aug = tour_lengths.min(dim=1).values
            best_no_aug_score = min(best_no_aug_score, float(best_len_per_aug[0].item()))
            best_aug_score = min(best_aug_score, float(best_len_per_aug.min().item()))

            if (step + 1) == 1 or (step + 1) == eas_steps or (step + 1) % log_interval == 0:
                self.logger.info(
                    "EAS step {}/{}: loss {:.6f}, sampled_no_aug {:.3f}, sampled_aug {:.3f}".format(
                        step + 1,
                        eas_steps,
                        loss.item(),
                        best_len_per_aug[0].item(),
                        best_len_per_aug.min().item(),
                    )
                )

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
            "EAS incumbent best merged with post-EAS eval: no_aug {:.3f}, aug {:.3f}".format(
                no_aug_score,
                aug_score,
            )
        )
        return no_aug_score, aug_score
