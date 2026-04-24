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
        self._load_initial_states()
        self._freeze_reference_model()

        self.logger.info(
            'Preference pairing: top-k vs bottom-k with k={} (cartesian pairs per instance).'.format(
                self.trainer_params['preference_pair_k']
            )
        )
        if self.trainer_params['use_reference_candidate_pool']:
            self.logger.info('Preference candidate pool: current-policy rollouts + sampled reference-policy rollouts.')
        else:
            self.logger.info('Preference candidate pool: current-policy rollouts only.')
        if self.trainer_params.get('use_2opt_teacher_candidate', False):
            self.logger.info(
                'Preference candidate pool: appending one 2-opt teacher tour per instance '
                '(max_iterations={}, interval={}, batch_limit={}).'.format(
                    self.trainer_params['two_opt_teacher_max_iterations'],
                    self.trainer_params['two_opt_teacher_interval'],
                    self.trainer_params['two_opt_teacher_batch_limit'],
                )
            )
        self.logger.info(
            'Preference gap weighting enabled with normalized_gap^{}.'.format(
                self.trainer_params['preference_gap_weight_power']
            )
        )

        self.time_estimator = TimeEstimator()
        self._two_opt_valid_mask_cache = {}
        self.train_batch_counter = 0

    def _load_initial_states(self):
        model_load = self.trainer_params['model_load']
        if not model_load.get('enable', False):
            raise ValueError('Preference post-training requires model_load.enable=True.')

        resume_load = self.trainer_params.get('resume_load', {})
        if resume_load.get('enable', False):
            self._resume_from_checkpoint(resume_load['path'])
            return

        checkpoint_fullname = model_load['path']
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        model_state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(model_state_dict)
        self.reference_model.load_state_dict(model_state_dict)

        total = sum(param.nelement() for param in self.model.parameters())
        self.logger.info('Reference checkpoint loaded from: {}'.format(checkpoint_fullname))
        self.logger.info('Number of parameters: %.2fM' % (total / 1e6))

    def _resume_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        reference_model_state_dict = checkpoint.get('reference_model_state_dict')
        if reference_model_state_dict is None:
            raise KeyError(
                'resume checkpoint is missing reference_model_state_dict: {}. '
                'Refusing to fall back to base_checkpoint because resume must preserve '
                'the exact frozen reference used before interruption.'.format(checkpoint_path)
            )
        self.reference_model.load_state_dict(reference_model_state_dict)

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self._move_optimizer_state_to_device()
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'result_log' in checkpoint:
            self.result_log.set_raw_data(checkpoint['result_log'])

        self.start_epoch = int(checkpoint['epoch']) + 1

        total = sum(param.nelement() for param in self.model.parameters())
        self.logger.info('Resumed post-training checkpoint from: {}'.format(checkpoint_path))
        self.logger.info('Resume start epoch: {}'.format(self.start_epoch))
        self.logger.info('Number of parameters: %.2fM' % (total / 1e6))

    def _move_optimizer_state_to_device(self):
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _freeze_reference_model(self):
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad_(False)

    def _get_curriculum_stage_info(self, epoch):
        curriculum = self.trainer_params['curriculum']
        problem_sizes = curriculum['problem_sizes']
        if not problem_sizes:
            raise ValueError('curriculum.problem_sizes must not be empty.')

        stage_epochs = curriculum.get('stage_epochs')
        if stage_epochs is None:
            stage_length = max(1, math.ceil(self.trainer_params['epochs'] / len(problem_sizes)))
            stage_idx = min((epoch - 1) // stage_length, len(problem_sizes) - 1)
            stage_start = stage_idx * stage_length + 1
            stage_end = min(self.trainer_params['epochs'], (stage_idx + 1) * stage_length)
        else:
            if len(stage_epochs) != len(problem_sizes):
                raise ValueError(
                    'curriculum.stage_epochs must have the same length as curriculum.problem_sizes.'
                )
            if sum(stage_epochs) != self.trainer_params['epochs']:
                raise ValueError(
                    'Sum of curriculum.stage_epochs must equal trainer_params["epochs"].'
                )

            running_epoch = 0
            stage_idx = None
            stage_start = None
            stage_end = None
            for candidate_stage_idx, stage_length in enumerate(stage_epochs):
                if stage_length <= 0:
                    raise ValueError('curriculum.stage_epochs must contain only positive integers.')
                candidate_stage_start = running_epoch + 1
                running_epoch += stage_length
                candidate_stage_end = running_epoch
                if epoch <= candidate_stage_end:
                    stage_idx = candidate_stage_idx
                    stage_start = candidate_stage_start
                    stage_end = candidate_stage_end
                    break

            if stage_idx is None:
                raise ValueError(
                    'Epoch {} exceeded configured curriculum stage epochs.'.format(epoch)
                )

        problem_mix_entries = self._build_stage_problem_mix_entries(stage_idx, problem_sizes)
        return {
            'stage_idx': stage_idx,
            'problem_size': problem_sizes[stage_idx],
            'stage_start': stage_start,
            'stage_end': stage_end,
            'problem_mix_entries': problem_mix_entries,
        }

    def _build_stage_problem_mix_entries(self, stage_idx, problem_sizes):
        curriculum = self.trainer_params['curriculum']
        raw_entries = [
            (problem_sizes[stage_idx], curriculum['current_stage_mix_weight']),
            (curriculum['base_replay_problem_size'], curriculum['base_replay_mix_weight']),
        ]
        if stage_idx > 0:
            raw_entries.append((problem_sizes[stage_idx - 1], curriculum['previous_stage_mix_weight']))

        combined_weights = {}
        order = []
        for problem_size, weight in raw_entries:
            if weight <= 0:
                continue
            if problem_size not in combined_weights:
                combined_weights[problem_size] = 0.0
                order.append(problem_size)
            combined_weights[problem_size] += weight

        total_weight = sum(combined_weights.values())
        if total_weight <= 0:
            raise ValueError('Curriculum replay weights must sum to a positive value.')

        return [
            (problem_size, combined_weights[problem_size] / total_weight)
            for problem_size in order
        ]

    @staticmethod
    def _format_problem_mix_entries(problem_mix_entries):
        return ', '.join(
            '{}:{:.1f}%'.format(problem_size, mix_weight * 100.0)
            for problem_size, mix_weight in problem_mix_entries
        )

    @staticmethod
    def _allocate_episode_targets(train_num_episode, problem_mix_entries):
        raw_targets = {
            problem_size: train_num_episode * mix_weight
            for problem_size, mix_weight in problem_mix_entries
        }
        episode_targets = {
            problem_size: int(math.floor(raw_target))
            for problem_size, raw_target in raw_targets.items()
        }

        remaining = train_num_episode - sum(episode_targets.values())
        if remaining > 0:
            ordered_problem_sizes = sorted(
                raw_targets.keys(),
                key=lambda problem_size: (
                    raw_targets[problem_size] - episode_targets[problem_size],
                    raw_targets[problem_size],
                ),
                reverse=True,
            )
            for idx in range(remaining):
                episode_targets[ordered_problem_sizes[idx % len(ordered_problem_sizes)]] += 1

        return episode_targets

    @staticmethod
    def _select_next_problem_size(episode_targets, episode_done_by_size):
        active_problem_sizes = [
            problem_size for problem_size, target in episode_targets.items()
            if episode_done_by_size[problem_size] < target
        ]
        if not active_problem_sizes:
            raise RuntimeError('No active problem sizes remain for the current epoch.')

        return max(
            active_problem_sizes,
            key=lambda problem_size: (
                (episode_targets[problem_size] - episode_done_by_size[problem_size]) / max(1, episode_targets[problem_size]),
                episode_targets[problem_size] - episode_done_by_size[problem_size],
            ),
        )

    def _get_train_batch_size(self, problem_size):
        batch_schedule = self.trainer_params.get('train_batch_size_by_problem_size')
        if batch_schedule is not None and problem_size in batch_schedule:
            return batch_schedule[problem_size]
        return self.trainer_params['train_batch_size']

    def _handle_oom(self, problem_size, attempted_batch_size, error):
        if 'out of memory' not in str(error).lower():
            raise error

        reduced_batch_size = attempted_batch_size // 2
        if reduced_batch_size < self.trainer_params['min_train_batch_size']:
            raise error

        self.logger.warning(
            'CUDA OOM at problem_size={}, batch_size={}. Reducing batch size to {} and retrying.'.format(
                problem_size,
                attempted_batch_size,
                reduced_batch_size,
            )
        )

        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.trainer_params['train_batch_size_by_problem_size'][problem_size] = reduced_batch_size
        return reduced_batch_size

    def _build_env(self, problem_size):
        env = Env(problem_size=problem_size, pomo_size=problem_size)
        return env

    def _build_env_from_problems(self, problems, pomo_size):
        env = Env(problem_size=problems.size(1), pomo_size=pomo_size)
        env.batch_size = problems.size(0)
        env.problems = problems
        device = problems.device
        env.BATCH_IDX = torch.arange(env.batch_size, device=device)[:, None].expand(env.batch_size, pomo_size)
        env.POMO_IDX = torch.arange(pomo_size, device=device)[None, :].expand(env.batch_size, pomo_size)
        return env

    def run(self):
        self.time_estimator.reset(self.start_epoch)

        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')

            self.scheduler.step()

            stage_info = self._get_curriculum_stage_info(epoch)
            problem_size = stage_info['problem_size']
            self.logger.info(
                'Epoch {:3d}: curriculum stage {}/{} -> problem_size={} (epochs {}-{}), mix=[{}]'.format(
                    epoch,
                    stage_info['stage_idx'] + 1,
                    len(self.trainer_params['curriculum']['problem_sizes']),
                    problem_size,
                    stage_info['stage_start'],
                    stage_info['stage_end'],
                    self._format_problem_mix_entries(stage_info['problem_mix_entries']),
                )
            )

            train_score, train_loss, train_pref_loss, train_rl_loss = self._train_one_epoch(epoch, stage_info)
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
                    'reference_is_frozen_from_run_start': True,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'curriculum_problem_sizes': self.trainer_params['curriculum']['problem_sizes'],
                    'curriculum_stage_epochs': self.trainer_params['curriculum'].get('stage_epochs'),
                    'base_replay_problem_size': self.trainer_params['curriculum']['base_replay_problem_size'],
                    'curriculum_mix_weights': {
                        'current_stage': self.trainer_params['curriculum']['current_stage_mix_weight'],
                        'previous_stage': self.trainer_params['curriculum']['previous_stage_mix_weight'],
                        'base_replay': self.trainer_params['curriculum']['base_replay_mix_weight'],
                    },
                    'preference_beta': self.trainer_params['preference_beta'],
                    'preference_pair_k': self.trainer_params['preference_pair_k'],
                    'use_reference_candidate_pool': self.trainer_params['use_reference_candidate_pool'],
                    'use_2opt_teacher_candidate': self.trainer_params.get('use_2opt_teacher_candidate', False),
                    'two_opt_teacher_max_iterations': self.trainer_params.get('two_opt_teacher_max_iterations'),
                    'two_opt_teacher_interval': self.trainer_params.get('two_opt_teacher_interval'),
                    'two_opt_teacher_batch_limit': self.trainer_params.get('two_opt_teacher_batch_limit'),
                    'preference_gap_weight_power': self.trainer_params['preference_gap_weight_power'],
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

    def _train_one_epoch(self, epoch, stage_info):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        pref_loss_AM = AverageMeter()
        rl_loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode_targets = self._allocate_episode_targets(train_num_episode, stage_info['problem_mix_entries'])
        episode_done_by_size = {
            problem_size: 0 for problem_size in episode_targets.keys()
        }
        episode = 0
        loop_cnt = 0

        with create_progress_bar(
            total=train_num_episode,
            desc='Epoch {:3d}'.format(epoch),
            unit='ep',
            leave=False,
        ) as progress_bar:
            while episode < train_num_episode:
                problem_size = self._select_next_problem_size(episode_targets, episode_done_by_size)
                remaining_for_problem_size = episode_targets[problem_size] - episode_done_by_size[problem_size]
                batch_size = self._get_train_batch_size(problem_size)
                current_batch_size = min(batch_size, remaining_for_problem_size)

                try:
                    avg_score, avg_loss, avg_pref_loss, avg_rl_loss = self._train_one_batch(
                        batch_size=current_batch_size,
                        problem_size=problem_size,
                    )
                except RuntimeError as error:
                    current_batch_size = self._handle_oom(problem_size, current_batch_size, error)
                    continue
                score_AM.update(avg_score, current_batch_size)
                loss_AM.update(avg_loss, current_batch_size)
                pref_loss_AM.update(avg_pref_loss, current_batch_size)
                rl_loss_AM.update(avg_rl_loss, current_batch_size)

                episode_done_by_size[problem_size] += current_batch_size
                episode += current_batch_size
                progress_bar.update(current_batch_size)

                if epoch == self.start_epoch:
                    loop_cnt += 1
                    if loop_cnt <= 10:
                        self.logger.info(
                            'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%) size={} '
                            'Score: {:.4f}, Loss: {:.4f}, Pref: {:.4f}, RL: {:.4f}'.format(
                                epoch,
                                episode,
                                train_num_episode,
                                100. * episode / train_num_episode,
                                problem_size,
                                score_AM.avg,
                                loss_AM.avg,
                                pref_loss_AM.avg,
                                rl_loss_AM.avg,
                            )
                        )

        replay_summary = ', '.join(
            '{}:{}/{}'.format(
                problem_size,
                episode_done_by_size[problem_size],
                episode_targets[problem_size],
            )
            for problem_size, _ in stage_info['problem_mix_entries']
        )
        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}, Loss: {:.4f}, Pref: {:.4f}, RL: {:.4f}, Replay[{}]'.format(
                epoch,
                100. * episode / train_num_episode,
                score_AM.avg,
                loss_AM.avg,
                pref_loss_AM.avg,
                rl_loss_AM.avg,
                replay_summary,
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
    def _build_preference_pair_indices(reward, pair_k):
        pomo_size = reward.size(1)
        effective_pair_k = max(1, min(pair_k, max(1, pomo_size // 2)))
        chosen_idx = reward.topk(effective_pair_k, dim=1, largest=True).indices
        rejected_idx = reward.topk(effective_pair_k, dim=1, largest=False).indices
        return chosen_idx, rejected_idx, effective_pair_k

    @staticmethod
    def _gather_multi_values(values, index):
        return values.gather(dim=1, index=index)

    def _sample_reference_candidates(self, env):
        original_eval_type = self.reference_model.model_params['eval_type']
        self.reference_model.set_eval_type('softmax')
        try:
            reward, prob_list, selected_actions = self._rollout(
                self.reference_model,
                env,
                collect_prob=True,
                no_grad=True,
            )
        finally:
            self.reference_model.set_eval_type(original_eval_type)

        log_prob = prob_list.clamp_min(1e-12).log().sum(dim=2)
        return reward, log_prob, selected_actions

    def _should_append_2opt_teacher_candidate(self):
        if not self.trainer_params.get('use_2opt_teacher_candidate', False):
            return False
        if int(self.trainer_params['two_opt_teacher_max_iterations']) <= 0:
            return False

        interval = max(1, int(self.trainer_params['two_opt_teacher_interval']))
        return (self.train_batch_counter % interval) == 0

    def _get_two_opt_valid_pair_mask(self, node_count, device):
        cache_key = (node_count, device.type, device.index)
        if cache_key not in self._two_opt_valid_mask_cache:
            left_idx = torch.arange(node_count, device=device)[:, None]
            right_idx = torch.arange(node_count, device=device)[None, :]
            self._two_opt_valid_mask_cache[cache_key] = (
                (left_idx >= 1) &
                (left_idx <= node_count - 2) &
                (right_idx > left_idx) &
                (right_idx <= node_count - 1)
            )
        return self._two_opt_valid_mask_cache[cache_key]

    def _batched_first_improvement_2opt(self, tours, dist_matrix, max_iterations):
        batch_size, node_count = tours.size()
        if node_count < 4 or max_iterations <= 0:
            return tours

        valid_pair_mask = self._get_two_opt_valid_pair_mask(node_count, tours.device)
        batch_line_idx = torch.arange(batch_size, device=tours.device)[:, None]
        batch_idx = torch.arange(batch_size, device=tours.device)[:, None, None]
        positions = torch.arange(node_count, device=tours.device)[None, :]

        for _ in range(max_iterations):
            prev_left = tours.roll(shifts=1, dims=1)
            left = tours
            right = tours
            next_right = tours.roll(shifts=-1, dims=1)

            removed_prev_left_to_left = dist_matrix[
                batch_line_idx,
                prev_left,
                left,
            ]
            removed_right_to_next_right = dist_matrix[
                batch_line_idx,
                right,
                next_right,
            ]

            delta = (
                dist_matrix[batch_idx, prev_left[:, :, None], right[:, None, :]] +
                dist_matrix[batch_idx, left[:, :, None], next_right[:, None, :]] -
                removed_prev_left_to_left[:, :, None] -
                removed_right_to_next_right[:, None, :]
            )
            improving_pairs = (delta < -1e-9) & valid_pair_mask[None, :, :]
            flattened_improving_pairs = improving_pairs.reshape(batch_size, -1)
            has_improvement = flattened_improving_pairs.any(dim=1)

            first_improvement = flattened_improving_pairs.to(torch.int64).argmax(dim=1)
            left_idx = first_improvement // node_count
            right_idx = first_improvement % node_count

            reverse_positions = left_idx[:, None] + right_idx[:, None] - positions
            should_reverse = (
                has_improvement[:, None] &
                (positions >= left_idx[:, None]) &
                (positions <= right_idx[:, None])
            )
            gather_positions = torch.where(should_reverse, reverse_positions, positions)
            tours = tours.gather(dim=1, index=gather_positions)

        return tours

    def _build_2opt_teacher_actions(self, problems, source_actions):
        max_iterations = int(self.trainer_params['two_opt_teacher_max_iterations'])
        teacher_actions = source_actions.detach().clone()

        batch_limit = int(self.trainer_params.get('two_opt_teacher_batch_limit', 0))
        active_count = problems.size(0)
        if batch_limit > 0:
            active_count = min(active_count, batch_limit)

        if active_count > 0:
            with torch.no_grad():
                active_problems = problems[:active_count].detach()
                dist_matrix = torch.cdist(active_problems, active_problems, p=2)
                teacher_actions[:active_count] = self._batched_first_improvement_2opt(
                    tours=teacher_actions[:active_count],
                    dist_matrix=dist_matrix,
                    max_iterations=max_iterations,
                )

        return teacher_actions[:, None, :]

    def _append_2opt_teacher_candidate(
        self,
        problems,
        candidate_reward,
        candidate_log_prob,
        candidate_ref_log_prob,
        candidate_actions,
    ):
        best_candidate_idx = candidate_reward.argmax(dim=1)
        batch_idx = torch.arange(candidate_reward.size(0), device=candidate_reward.device)
        source_actions = candidate_actions[batch_idx, best_candidate_idx]
        teacher_actions = self._build_2opt_teacher_actions(
            problems=problems,
            source_actions=source_actions,
        )

        teacher_env = self._build_env_from_problems(
            problems=problems,
            pomo_size=teacher_actions.size(1),
        )
        teacher_reward, teacher_prob_list, _ = self._rollout(
            self.model,
            teacher_env,
            collect_prob=True,
            no_grad=False,
            forced_actions=teacher_actions.detach(),
        )
        teacher_log_prob = teacher_prob_list.clamp_min(1e-12).log().sum(dim=2)

        self.reference_model.eval()
        ref_teacher_env = self._build_env_from_problems(
            problems=problems,
            pomo_size=teacher_actions.size(1),
        )
        _, ref_teacher_prob_list, _ = self._rollout(
            self.reference_model,
            ref_teacher_env,
            collect_prob=True,
            no_grad=True,
            forced_actions=teacher_actions.detach(),
        )
        teacher_ref_log_prob = ref_teacher_prob_list.clamp_min(1e-12).log().sum(dim=2)

        return (
            torch.cat((candidate_reward, teacher_reward), dim=1),
            torch.cat((candidate_log_prob, teacher_log_prob), dim=1),
            torch.cat((candidate_ref_log_prob, teacher_ref_log_prob), dim=1),
            torch.cat((candidate_actions, teacher_actions), dim=1),
        )

    def _compute_gap_weighted_preference_loss(
        self,
        preference_logits,
        chosen_reward,
        rejected_reward,
    ):
        pair_loss = -F.logsigmoid(preference_logits)
        reward_gap = (chosen_reward[:, :, None] - rejected_reward[:, None, :]).clamp_min(0.0)
        gap_scale = reward_gap.mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        gap_weights = reward_gap / gap_scale

        gap_weight_power = self.trainer_params['preference_gap_weight_power']
        if gap_weight_power != 1.0:
            gap_weights = gap_weights.pow(gap_weight_power)

        total_gap_weight = gap_weights.sum()
        if total_gap_weight.item() <= 0:
            return pair_loss.mean()

        return (pair_loss * gap_weights).sum() / total_gap_weight

    def _train_one_batch(self, batch_size, problem_size):
        self.train_batch_counter += 1
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

        self.reference_model.eval()
        _, ref_prob_list, _ = self._rollout(
            self.reference_model,
            env,
            collect_prob=True,
            no_grad=True,
            forced_actions=selected_actions.detach(),
        )
        ref_log_prob = ref_prob_list.clamp_min(1e-12).log().sum(dim=2)

        candidate_reward = reward
        candidate_log_prob = log_prob
        candidate_ref_log_prob = ref_log_prob
        candidate_actions = selected_actions
        if self.trainer_params['use_reference_candidate_pool']:
            ref_candidate_reward, ref_candidate_log_prob, ref_candidate_actions = self._sample_reference_candidates(env)
            _, current_on_ref_prob_list, _ = self._rollout(
                self.model,
                env,
                collect_prob=True,
                no_grad=False,
                forced_actions=ref_candidate_actions.detach(),
            )
            current_on_ref_log_prob = current_on_ref_prob_list.clamp_min(1e-12).log().sum(dim=2)

            candidate_reward = torch.cat((candidate_reward, ref_candidate_reward), dim=1)
            candidate_log_prob = torch.cat((candidate_log_prob, current_on_ref_log_prob), dim=1)
            candidate_ref_log_prob = torch.cat((candidate_ref_log_prob, ref_candidate_log_prob), dim=1)
            candidate_actions = torch.cat((candidate_actions, ref_candidate_actions), dim=1)

        if self._should_append_2opt_teacher_candidate():
            (
                candidate_reward,
                candidate_log_prob,
                candidate_ref_log_prob,
                candidate_actions,
            ) = self._append_2opt_teacher_candidate(
                problems=env.problems,
                candidate_reward=candidate_reward,
                candidate_log_prob=candidate_log_prob,
                candidate_ref_log_prob=candidate_ref_log_prob,
                candidate_actions=candidate_actions,
            )

        chosen_idx, rejected_idx, _ = self._build_preference_pair_indices(
            candidate_reward,
            self.trainer_params['preference_pair_k'],
        )
        chosen_log_prob = self._gather_multi_values(candidate_log_prob, chosen_idx)
        rejected_log_prob = self._gather_multi_values(candidate_log_prob, rejected_idx)
        ref_chosen_log_prob = self._gather_multi_values(candidate_ref_log_prob, chosen_idx)
        ref_rejected_log_prob = self._gather_multi_values(candidate_ref_log_prob, rejected_idx)
        chosen_reward = self._gather_multi_values(candidate_reward, chosen_idx)
        rejected_reward = self._gather_multi_values(candidate_reward, rejected_idx)

        beta = self.trainer_params['preference_beta']
        chosen_delta = chosen_log_prob[:, :, None] - rejected_log_prob[:, None, :]
        ref_delta = ref_chosen_log_prob[:, :, None] - ref_rejected_log_prob[:, None, :]
        preference_logits = beta * (
            chosen_delta -
            ref_delta
        )
        preference_loss = self._compute_gap_weighted_preference_loss(
            preference_logits,
            chosen_reward,
            rejected_reward,
        )

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
