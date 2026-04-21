import math
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *


class TSPPreferenceTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        use_cuda = self.trainer_params['use_cuda']
        if use_cuda:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        self.model = Model(**self.model_params)
        self.reference_model = Model(**self.model_params)

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        self.start_epoch = 1
        self._load_reference_checkpoint()
        self._freeze_reference_model()

        self.time_estimator = TimeEstimator()

    def _load_reference_checkpoint(self):
        model_load = self.trainer_params['model_load']
        if not model_load.get('enable', False):
            raise ValueError('Preference post-training requires model_load.enable=True.')

        checkpoint_fullname = model_load['path']
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        model_state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(model_state_dict)
        self.reference_model.load_state_dict(model_state_dict)

        total = sum(param.nelement() for param in self.model.parameters())
        self.logger.info('Reference checkpoint loaded from: {}'.format(checkpoint_fullname))
        self.logger.info('Number of parameters: %.2fM' % (total / 1e6))

    def _freeze_reference_model(self):
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad_(False)

    def _get_curriculum_stage_info(self, epoch):
        curriculum = self.trainer_params['curriculum']
        problem_sizes = curriculum['problem_sizes']
        if not problem_sizes:
            raise ValueError('curriculum.problem_sizes must not be empty.')

        stage_length = max(1, math.ceil(self.trainer_params['epochs'] / len(problem_sizes)))
        stage_idx = min((epoch - 1) // stage_length, len(problem_sizes) - 1)
        stage_start = stage_idx * stage_length + 1
        stage_end = min(self.trainer_params['epochs'], (stage_idx + 1) * stage_length)
        return {
            'stage_idx': stage_idx,
            'problem_size': problem_sizes[stage_idx],
            'stage_start': stage_start,
            'stage_end': stage_end,
        }

    def _get_train_batch_size(self, problem_size):
        batch_schedule = self.trainer_params.get('train_batch_size_by_problem_size')
        if batch_schedule is not None and problem_size in batch_schedule:
            return batch_schedule[problem_size]
        return self.trainer_params['train_batch_size']

    def _build_env(self, problem_size):
        env = Env(problem_size=problem_size, pomo_size=problem_size)
        return env

    def run(self):
        self.time_estimator.reset(self.start_epoch)

        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')

            self.scheduler.step()

            stage_info = self._get_curriculum_stage_info(epoch)
            problem_size = stage_info['problem_size']
            self.logger.info(
                'Epoch {:3d}: curriculum stage {}/{} -> problem_size={} (epochs {}-{})'.format(
                    epoch,
                    stage_info['stage_idx'] + 1,
                    len(self.trainer_params['curriculum']['problem_sizes']),
                    problem_size,
                    stage_info['stage_start'],
                    stage_info['stage_end'],
                )
            )

            train_score, train_loss, train_pref_loss, train_rl_loss = self._train_one_epoch(epoch, problem_size)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            self.result_log.append('train_pref_loss', epoch, train_pref_loss)
            self.result_log.append('train_rl_loss', epoch, train_rl_loss)
            self.result_log.append('train_problem_size', epoch, problem_size)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_1'],
                    self.result_log,
                    labels=['train_score']
                )
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_2'],
                    self.result_log,
                    labels=['train_loss', 'train_pref_loss', 'train_rl_loss']
                )

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'reference_model_state_dict': self.reference_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'curriculum_problem_sizes': self.trainer_params['curriculum']['problem_sizes'],
                    'preference_beta': self.trainer_params['preference_beta'],
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_1'],
                    self.result_log,
                    labels=['train_score']
                )
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_2'],
                    self.result_log,
                    labels=['train_loss', 'train_pref_loss', 'train_rl_loss']
                )

            if all_done:
                self.logger.info(" *** Preference Post-Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch, problem_size):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        pref_loss_AM = AverageMeter()
        rl_loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        batch_size = self._get_train_batch_size(problem_size)
        episode = 0
        loop_cnt = 0

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            current_batch_size = min(batch_size, remaining)

            avg_score, avg_loss, avg_pref_loss, avg_rl_loss = self._train_one_batch(
                batch_size=current_batch_size,
                problem_size=problem_size,
            )
            score_AM.update(avg_score, current_batch_size)
            loss_AM.update(avg_loss, current_batch_size)
            pref_loss_AM.update(avg_pref_loss, current_batch_size)
            rl_loss_AM.update(avg_rl_loss, current_batch_size)

            episode += current_batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  '
                        'Score: {:.4f}, Loss: {:.4f}, Pref: {:.4f}, RL: {:.4f}'.format(
                            epoch,
                            episode,
                            train_num_episode,
                            100. * episode / train_num_episode,
                            score_AM.avg,
                            loss_AM.avg,
                            pref_loss_AM.avg,
                            rl_loss_AM.avg,
                        )
                    )

        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}, Loss: {:.4f}, Pref: {:.4f}, RL: {:.4f}'.format(
                epoch,
                100. * episode / train_num_episode,
                score_AM.avg,
                loss_AM.avg,
                pref_loss_AM.avg,
                rl_loss_AM.avg,
            )
        )

        return score_AM.avg, loss_AM.avg, pref_loss_AM.avg, rl_loss_AM.avg

    def _rollout(self, model, env, collect_prob, no_grad=False, forced_actions=None):
        context = torch.no_grad() if no_grad else nullcontext()
        with context:
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            prob_list = []
            selected_list = []
            step_idx = 0

            while not done:
                selected_override = None
                if forced_actions is not None:
                    selected_override = forced_actions[:, :, step_idx]

                selected, prob = model(state, selected_override=selected_override)
                state, reward, done = env.step(selected)

                selected_list.append(selected[:, :, None])
                if collect_prob:
                    prob_list.append(prob[:, :, None])

                step_idx += 1

        selected_tensor = torch.cat(selected_list, dim=2)
        prob_tensor = None
        if collect_prob:
            prob_tensor = torch.cat(prob_list, dim=2)

        return reward, prob_tensor, selected_tensor

    @staticmethod
    def _gather_pair_values(values, chosen_idx, rejected_idx):
        chosen = values.gather(dim=1, index=chosen_idx[:, None]).squeeze(1)
        rejected = values.gather(dim=1, index=rejected_idx[:, None]).squeeze(1)
        return chosen, rejected

    def _train_one_batch(self, batch_size, problem_size):
        self.model.train()
        self.model.set_eval_type('softmax')

        env = self._build_env(problem_size)
        env.load_problems(batch_size)

        reward, prob_list, selected_actions = self._rollout(
            self.model,
            env,
            collect_prob=True,
            no_grad=False,
        )

        log_prob = prob_list.clamp_min(1e-12).log().sum(dim=2)
        advantage = reward - reward.float().mean(dim=1, keepdim=True)
        rl_loss = (-advantage * log_prob).mean()

        chosen_idx = reward.argmax(dim=1)
        rejected_idx = reward.argmin(dim=1)

        chosen_log_prob, rejected_log_prob = self._gather_pair_values(log_prob, chosen_idx, rejected_idx)

        self.reference_model.eval()
        _, ref_prob_list, _ = self._rollout(
            self.reference_model,
            env,
            collect_prob=True,
            no_grad=True,
            forced_actions=selected_actions.detach(),
        )
        ref_log_prob = ref_prob_list.clamp_min(1e-12).log().sum(dim=2)
        ref_chosen_log_prob, ref_rejected_log_prob = self._gather_pair_values(ref_log_prob, chosen_idx, rejected_idx)

        beta = self.trainer_params['preference_beta']
        preference_logits = beta * (
            (chosen_log_prob - rejected_log_prob) -
            (ref_chosen_log_prob - ref_rejected_log_prob)
        )
        preference_loss = -F.logsigmoid(preference_logits).mean()

        total_loss = (
            self.trainer_params['preference_loss_weight'] * preference_loss +
            self.trainer_params['rl_loss_weight'] * rl_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        best_pomo_reward = reward.max(dim=1).values
        score_mean = -best_pomo_reward.float().mean()

        self.model.set_eval_type(self.model_params['eval_type'])
        return (
            score_mean.item(),
            total_loss.item(),
            preference_loss.item(),
            rl_loss.item(),
        )
