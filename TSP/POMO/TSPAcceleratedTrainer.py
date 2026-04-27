import math
from contextlib import nullcontext
from logging import getLogger

import torch

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *


class TSPAcceleratedTrainer:
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

        self.model = Model(**self.model_params).to(self.device)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        self.start_epoch = 1
        self.global_batch_step = 0
        self._load_initial_state()

        total_params = sum(param.nelement() for param in self.model.parameters())
        self.logger.info(
            'Stage1 accelerated training: SCST baseline + elite imitation + teacher distillation.'
        )
        self.logger.info('Number of parameters: %.2fM' % (total_params / 1e6))
        self.logger.info(
            'Loss weights: scst={}, elite={}, teacher={}.'.format(
                self.trainer_params['scst_loss_weight'],
                self.trainer_params['elite_loss_weight'],
                self.trainer_params['teacher_loss_weight'],
            )
        )
        self.logger.info(
            'Elite top-k={}, teacher_use_2opt={}, two_opt_max_iterations={}, grad_clip={}.'.format(
                self.trainer_params['elite_topk'],
                self.trainer_params['teacher_use_2opt'],
                self.trainer_params['two_opt_teacher_max_iterations'],
                self.trainer_params['max_grad_norm'],
            )
        )

        self.time_estimator = TimeEstimator()

    def _load_initial_state(self):
        resume_load = self.trainer_params.get('resume_load', {})
        if resume_load.get('enable', False):
            checkpoint = torch.load(resume_load['path'], map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self._move_optimizer_state_to_device()
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'result_log' in checkpoint:
                self.result_log.set_raw_data(checkpoint['result_log'])
            self.start_epoch = int(checkpoint['epoch']) + 1
            self.global_batch_step = int(checkpoint.get('global_batch_step', 0))
            self.logger.info('Resumed accelerated trainer from: {}'.format(resume_load['path']))
            self.logger.info('Resume start epoch: {}'.format(self.start_epoch))
            return

        model_load = self.trainer_params.get('model_load', {})
        if model_load.get('enable', False):
            checkpoint = torch.load(model_load['path'], map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info('Initialized accelerated trainer from: {}'.format(model_load['path']))

    def _move_optimizer_state_to_device(self):
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')
            self.scheduler.step()

            (
                train_score,
                train_greedy_score,
                train_teacher_score,
                train_loss,
                train_scst_loss,
                train_elite_loss,
                train_teacher_loss,
            ) = self._train_one_epoch(epoch)

            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_greedy_score', epoch, train_greedy_score)
            self.result_log.append('train_teacher_score', epoch, train_teacher_score)
            self.result_log.append('train_loss', epoch, train_loss)
            self.result_log.append('train_scst_loss', epoch, train_scst_loss)
            self.result_log.append('train_elite_loss', epoch, train_elite_loss)
            self.result_log.append('train_teacher_loss', epoch, train_teacher_loss)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                epoch,
                self.trainer_params['epochs'],
            )
            self.logger.info(
                'Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]'.format(
                    epoch,
                    self.trainer_params['epochs'],
                    elapsed_time_str,
                    remain_time_str,
                )
            )

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:
                self.logger.info('Saving log_image')
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_1'],
                    self.result_log,
                    labels=['train_score', 'train_greedy_score', 'train_teacher_score'],
                )
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_2'],
                    self.result_log,
                    labels=[
                        'train_loss',
                        'train_scst_loss',
                        'train_elite_loss',
                        'train_teacher_loss',
                    ],
                )

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info('Saving trained_model')
                checkpoint_dict = {
                    'epoch': epoch,
                    'global_batch_step': self.global_batch_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'algorithm': 'accelerated_stage1',
                    'trainer_params': {
                        'problem_size': self.env_params['problem_size'],
                        'pomo_size': self.env_params['pomo_size'],
                        'scst_loss_weight': self.trainer_params['scst_loss_weight'],
                        'elite_loss_weight': self.trainer_params['elite_loss_weight'],
                        'elite_topk': self.trainer_params['elite_topk'],
                        'teacher_loss_weight': self.trainer_params['teacher_loss_weight'],
                        'teacher_use_2opt': self.trainer_params['teacher_use_2opt'],
                        'two_opt_teacher_max_iterations': self.trainer_params['two_opt_teacher_max_iterations'],
                        'max_grad_norm': self.trainer_params['max_grad_norm'],
                    },
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_1'],
                    self.result_log,
                    labels=['train_score', 'train_greedy_score', 'train_teacher_score'],
                )
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_2'],
                    self.result_log,
                    labels=[
                        'train_loss',
                        'train_scst_loss',
                        'train_elite_loss',
                        'train_teacher_loss',
                    ],
                )

            if all_done:
                self.logger.info(' *** Accelerated Stage1 Training Done *** ')
                self.logger.info('Now, printing log array...')
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        greedy_score_AM = AverageMeter()
        teacher_score_AM = AverageMeter()
        loss_AM = AverageMeter()
        scst_loss_AM = AverageMeter()
        elite_loss_AM = AverageMeter()
        teacher_loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            (
                avg_score,
                avg_greedy_score,
                avg_teacher_score,
                avg_loss,
                avg_scst_loss,
                avg_elite_loss,
                avg_teacher_loss,
            ) = self._train_one_batch(batch_size)

            score_AM.update(avg_score, batch_size)
            greedy_score_AM.update(avg_greedy_score, batch_size)
            teacher_score_AM.update(avg_teacher_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            scst_loss_AM.update(avg_scst_loss, batch_size)
            elite_loss_AM.update(avg_elite_loss, batch_size)
            teacher_loss_AM.update(avg_teacher_loss, batch_size)
            episode += batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%) Score: {:.4f}, Greedy: {:.4f}, '
                        'Teacher: {:.4f}, Loss: {:.4f}, SCST: {:.4f}, Elite: {:.4f}, Tchr: {:.4f}'.format(
                            epoch,
                            episode,
                            train_num_episode,
                            100. * episode / train_num_episode,
                            score_AM.avg,
                            greedy_score_AM.avg,
                            teacher_score_AM.avg,
                            loss_AM.avg,
                            scst_loss_AM.avg,
                            elite_loss_AM.avg,
                            teacher_loss_AM.avg,
                        )
                    )

        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%) Score: {:.4f}, Greedy: {:.4f}, Teacher: {:.4f}, '
            'Loss: {:.4f}, SCST: {:.4f}, Elite: {:.4f}, Tchr: {:.4f}'.format(
                epoch,
                100. * episode / train_num_episode,
                score_AM.avg,
                greedy_score_AM.avg,
                teacher_score_AM.avg,
                loss_AM.avg,
                scst_loss_AM.avg,
                elite_loss_AM.avg,
                teacher_loss_AM.avg,
            )
        )

        return (
            score_AM.avg,
            greedy_score_AM.avg,
            teacher_score_AM.avg,
            loss_AM.avg,
            scst_loss_AM.avg,
            elite_loss_AM.avg,
            teacher_loss_AM.avg,
        )

    def _prepare_train_batch(self, batch_size):
        self.env.load_problems(batch_size)
        self.env.problems = self.env.problems.to(self.device)
        self.env.BATCH_IDX = self.env.BATCH_IDX.to(self.device)
        self.env.POMO_IDX = self.env.POMO_IDX.to(self.device)

    def _build_env_from_problems(self, problems, pomo_size):
        env = Env(problem_size=problems.size(1), pomo_size=pomo_size)
        env.batch_size = problems.size(0)
        env.problems = problems
        device = problems.device
        env.BATCH_IDX = torch.arange(env.batch_size, device=device)[:, None].expand(env.batch_size, pomo_size)
        env.POMO_IDX = torch.arange(pomo_size, device=device)[None, :].expand(env.batch_size, pomo_size)
        return env

    def _rollout(self, env, collect_prob, no_grad=False, forced_actions=None, eval_type=None):
        original_eval_type = self.model.model_params['eval_type']
        original_training = self.model.training

        if eval_type is not None:
            self.model.set_eval_type(eval_type)
        if no_grad:
            self.model.eval()
        else:
            self.model.train()

        context = torch.no_grad() if no_grad else nullcontext()
        with context:
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            prob_list = []
            selected_list = []
            step_idx = 0

            while not done:
                selected_override = None
                if forced_actions is not None:
                    selected_override = forced_actions[:, :, step_idx]

                selected, prob = self.model(state, selected_override=selected_override)
                state, reward, done = env.step(selected)

                selected_list.append(selected[:, :, None])
                if collect_prob:
                    prob_list.append(prob[:, :, None])

                step_idx += 1

        self.model.set_eval_type(original_eval_type)
        if original_training:
            self.model.train()
        else:
            self.model.eval()

        selected_tensor = torch.cat(selected_list, dim=2)
        prob_tensor = None
        if collect_prob:
            prob_tensor = torch.cat(prob_list, dim=2)

        return reward, prob_tensor, selected_tensor

    @staticmethod
    def _gather(values, index):
        return values.gather(dim=1, index=index)

    def _compute_scst_loss(self, reward, greedy_reward, sampled_log_prob):
        advantage = reward - greedy_reward
        advantage_scale = advantage.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized_advantage = (advantage / advantage_scale).clamp(min=-10.0, max=10.0)
        return -(normalized_advantage.detach() * sampled_log_prob).mean()

    def _compute_elite_loss(self, reward, sampled_log_prob):
        elite_k = max(1, min(int(self.trainer_params['elite_topk']), reward.size(1)))
        elite_reward, elite_idx = reward.topk(elite_k, dim=1)
        elite_log_prob = self._gather(sampled_log_prob, elite_idx)

        reward_center = reward.mean(dim=1, keepdim=True)
        elite_gap = (elite_reward - reward_center).clamp_min(0.0)
        gap_sum = elite_gap.sum(dim=1, keepdim=True)
        uniform_weights = torch.full_like(elite_gap, 1.0 / elite_k)
        elite_weights = torch.where(
            gap_sum > 0,
            elite_gap / gap_sum.clamp_min(1e-12),
            uniform_weights,
        )
        return -(elite_weights.detach() * elite_log_prob).sum(dim=1).mean()

    def _build_teacher_actions(self, problems, sampled_reward, sampled_actions, greedy_reward, greedy_actions):
        batch_size = sampled_reward.size(0)
        batch_idx = torch.arange(batch_size, device=sampled_reward.device)

        sampled_best_reward, sampled_best_idx = sampled_reward.max(dim=1)
        greedy_best_reward, greedy_best_idx = greedy_reward.max(dim=1)

        sampled_best_actions = sampled_actions[batch_idx, sampled_best_idx]
        greedy_best_actions = greedy_actions[batch_idx, greedy_best_idx]

        use_sampled_teacher = sampled_best_reward >= greedy_best_reward
        teacher_source_actions = torch.where(
            use_sampled_teacher[:, None],
            sampled_best_actions,
            greedy_best_actions,
        )
        teacher_source_reward = torch.maximum(sampled_best_reward, greedy_best_reward)

        if not self.trainer_params['teacher_use_2opt']:
            return teacher_source_actions[:, None, :], teacher_source_reward

        teacher_actions = self._build_2opt_teacher_actions(
            problems=problems,
            source_actions=teacher_source_actions,
        )
        return teacher_actions, teacher_source_reward

    def _build_2opt_teacher_actions(self, problems, source_actions):
        max_iterations = int(self.trainer_params['two_opt_teacher_max_iterations'])
        teacher_actions = source_actions.detach().clone()
        batch_size, problem_size = teacher_actions.size()
        if problem_size < 4 or max_iterations <= 0:
            return teacher_actions[:, None, :]

        device = teacher_actions.device
        inf = torch.tensor(float('inf'), device=device)
        position = torch.arange(problem_size, device=device)
        valid_swap_mask = torch.triu(
            torch.ones(
                (problem_size, problem_size),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )
        valid_swap_mask[0, :] = False
        valid_swap_mask[-1, :] = False
        flat_position = torch.arange(problem_size * problem_size, device=device)

        with torch.no_grad():
            dist_matrix = torch.cdist(problems.detach(), problems.detach(), p=2)
            batch_idx = torch.arange(batch_size, device=device)[:, None, None]

            for _ in range(max_iterations):
                prev_left = teacher_actions[:, (position - 1) % problem_size]
                left = teacher_actions
                right = teacher_actions
                next_right = teacher_actions[:, (position + 1) % problem_size]

                delta = (
                    dist_matrix[batch_idx, prev_left[:, :, None], right[:, None, :]] +
                    dist_matrix[batch_idx, left[:, :, None], next_right[:, None, :]] -
                    dist_matrix[batch_idx, prev_left[:, :, None], left[:, :, None]] -
                    dist_matrix[batch_idx, right[:, None, :], next_right[:, None, :]]
                )
                delta = delta.masked_fill(~valid_swap_mask[None, :, :], inf)

                improvement = delta.reshape(batch_size, -1) < -1e-9
                first_improvement = flat_position[None, :].masked_fill(
                    ~improvement,
                    problem_size * problem_size,
                ).min(dim=1).values
                improved = first_improvement < problem_size * problem_size
                if not bool(improved.any()):
                    break

                flat_idx = first_improvement.clamp_max(problem_size * problem_size - 1)
                left_idx = (flat_idx // problem_size)[:, None]
                right_idx = (flat_idx % problem_size)[:, None]
                position_2d = position[None, :].expand(batch_size, problem_size)
                reversed_position = left_idx + right_idx - position_2d
                reverse_mask = (
                    improved[:, None] &
                    (position_2d >= left_idx) &
                    (position_2d <= right_idx)
                )
                gather_position = torch.where(
                    reverse_mask,
                    reversed_position,
                    position_2d,
                )
                teacher_actions = teacher_actions.gather(dim=1, index=gather_position)

        return teacher_actions.clone()[:, None, :]

    def _compute_teacher_loss(self, problems, teacher_actions, teacher_source_reward):
        teacher_env = self._build_env_from_problems(
            problems=problems,
            pomo_size=teacher_actions.size(1),
        )
        teacher_reward, teacher_prob_list, _ = self._rollout(
            teacher_env,
            collect_prob=True,
            no_grad=False,
            forced_actions=teacher_actions.detach(),
            eval_type='softmax',
        )
        teacher_log_prob = teacher_prob_list.clamp_min(1e-12).log().sum(dim=2).squeeze(1)
        teacher_reward = teacher_reward.squeeze(1)

        teacher_gap = (teacher_reward - teacher_source_reward).clamp_min(0.0)
        if bool((teacher_gap > 0).any()):
            teacher_scale = teacher_gap.mean().clamp_min(1e-12)
            teacher_weights = 1.0 + teacher_gap / teacher_scale
        else:
            teacher_weights = torch.ones_like(teacher_reward)

        teacher_loss = -(teacher_weights.detach() * teacher_log_prob).mean()
        teacher_score = -teacher_reward.float().mean()
        return teacher_loss, teacher_score

    def _train_one_batch(self, batch_size):
        self.global_batch_step += 1
        self._prepare_train_batch(batch_size)

        sampled_reward, sampled_prob_list, sampled_actions = self._rollout(
            self.env,
            collect_prob=True,
            no_grad=False,
            eval_type='softmax',
        )
        sampled_log_prob = sampled_prob_list.clamp_min(1e-12).log().sum(dim=2)

        greedy_reward, _, greedy_actions = self._rollout(
            self.env,
            collect_prob=False,
            no_grad=True,
            eval_type='argmax',
        )

        scst_loss = sampled_log_prob.sum() * 0.0
        if self.trainer_params['scst_loss_weight'] > 0:
            scst_loss = self._compute_scst_loss(
                reward=sampled_reward,
                greedy_reward=greedy_reward,
                sampled_log_prob=sampled_log_prob,
            )

        elite_loss = sampled_log_prob.sum() * 0.0
        if self.trainer_params['elite_loss_weight'] > 0:
            elite_loss = self._compute_elite_loss(
                reward=sampled_reward,
                sampled_log_prob=sampled_log_prob,
            )

        teacher_loss = sampled_log_prob.sum() * 0.0
        teacher_score = -torch.maximum(
            sampled_reward.max(dim=1).values,
            greedy_reward.max(dim=1).values,
        ).float().mean()
        if self.trainer_params['teacher_loss_weight'] > 0:
            teacher_actions, teacher_source_reward = self._build_teacher_actions(
                problems=self.env.problems,
                sampled_reward=sampled_reward,
                sampled_actions=sampled_actions,
                greedy_reward=greedy_reward,
                greedy_actions=greedy_actions,
            )
            teacher_loss, teacher_score = self._compute_teacher_loss(
                problems=self.env.problems,
                teacher_actions=teacher_actions,
                teacher_source_reward=teacher_source_reward,
            )

        total_loss = (
            self.trainer_params['scst_loss_weight'] * scst_loss +
            self.trainer_params['elite_loss_weight'] * elite_loss +
            self.trainer_params['teacher_loss_weight'] * teacher_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        max_grad_norm = self.trainer_params['max_grad_norm']
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

        self.optimizer.step()

        sampled_best_reward = sampled_reward.max(dim=1).values
        greedy_best_reward = greedy_reward.max(dim=1).values
        score_mean = -sampled_best_reward.float().mean()
        greedy_score_mean = -greedy_best_reward.float().mean()

        return (
            score_mean.item(),
            greedy_score_mean.item(),
            teacher_score.item(),
            total_loss.item(),
            scst_loss.item(),
            elite_loss.item(),
            teacher_loss.item(),
        )
