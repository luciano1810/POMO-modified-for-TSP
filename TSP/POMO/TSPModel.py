
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def get_lora_parameters(self):
        params = []
        param_names = []
        for name, param in self.named_parameters():
            if '_lora.' in name:
                params.append(param)
                param_names.append(name)
        return params, param_names

    def freeze_non_lora_parameters(self):
        lora_params, _ = self.get_lora_parameters()
        lora_param_ids = {id(param) for param in lora_params}
        for param in self.parameters():
            param.requires_grad_(id(param) in lora_param_ids)
        return lora_params

    def get_lora_state_dict(self):
        return {
            key: value.detach().clone()
            for key, value in self.state_dict().items()
            if '_lora.' in key
        }

    def merged_state_dict(self):
        merged = {
            key: value.detach().clone()
            for key, value in self.state_dict().items()
            if '_lora.' not in key
        }
        self.decoder.merge_lora_into_state_dict(merged, prefix='decoder.')
        return merged

    def pre_forward(self, reset_state):
        self.encoded_nodes = self.encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

    def set_eval_type(self, eval_type):
        self.model_params['eval_type'] = eval_type

    def get_eas_parameters(self, target):
        encoder_first2 = [
            (f'encoder.layers.{layer_idx}', layer)
            for layer_idx, layer in enumerate(self.encoder.layers[:2])
        ]
        group_specs = {
            'embedding': [
                ('encoder.embedding', self.encoder.embedding),
            ],
            'decoder_wq_last': [
                ('decoder.Wq_last', self.decoder.Wq_last),
            ],
            'decoder_combine': [
                ('decoder.multi_head_combine', self.decoder.multi_head_combine),
            ],
            'decoder_last': [
                ('decoder.Wq_last', self.decoder.Wq_last),
                ('decoder.multi_head_combine', self.decoder.multi_head_combine),
            ],
            'encoder_first2': encoder_first2,
            'embedding_decoder': [
                ('encoder.embedding', self.encoder.embedding),
                ('decoder.Wq_last', self.decoder.Wq_last),
                ('decoder.multi_head_combine', self.decoder.multi_head_combine),
            ],
        }
        if target not in group_specs:
            raise ValueError(f"Unsupported EAS parameter group: {target}")

        params = []
        param_names = []
        seen_param_ids = set()
        for module_prefix, module in group_specs[target]:
            for param_name, param in module.named_parameters():
                param_id = id(param)
                if param_id in seen_param_ids:
                    continue
                seen_param_ids.add(param_id)
                params.append(param)
                param_names.append(f'{module_prefix}.{param_name}')

        return params, param_names

    def forward(self, state, selected_override=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            if selected_override is None:
                selected = torch.arange(
                    pomo_size,
                    device=state.BATCH_IDX.device,
                )[None, :].expand(batch_size, pomo_size)
            else:
                selected = selected_override
            prob = torch.ones(
                size=(batch_size, pomo_size),
                device=state.BATCH_IDX.device,
            )

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)

            if selected_override is not None:
                selected = selected_override
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                    .reshape(batch_size, pomo_size)
            elif self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None


        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        out = embedded_input
        for layer in self.layers:
            out = layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        qkv_out_dim = head_num * qkv_dim

        self.Wq_first = nn.Linear(embedding_dim, qkv_out_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, qkv_out_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, qkv_out_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, qkv_out_dim, bias=False)

        self.multi_head_combine = nn.Linear(qkv_out_dim, embedding_dim)

        lora_targets = _resolve_lora_targets(model_params)
        lora_rank = int(model_params.get('lora_rank', 4))
        lora_alpha = float(model_params.get('lora_alpha', lora_rank))
        lora_dropout = float(model_params.get('lora_dropout', 0.0))
        self.Wq_last_lora = _make_lora_delta(
            enabled=('decoder_wq_last' in lora_targets),
            in_features=embedding_dim,
            out_features=qkv_out_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        self.multi_head_combine_lora = _make_lora_delta(
            enabled=('decoder_combine' in lora_targets),
            in_features=qkv_out_dim,
            out_features=embedding_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q_last_linear = self.Wq_last(encoded_last_node)
        if self.Wq_last_lora is not None:
            q_last_linear = q_last_linear + self.Wq_last_lora(encoded_last_node)
        q_last = reshape_by_heads(q_last_linear, head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = self.q_first + q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        if self.multi_head_combine_lora is not None:
            mh_atten_out = mh_atten_out + self.multi_head_combine_lora(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs

    def merge_lora_into_state_dict(self, state_dict, prefix):
        if self.Wq_last_lora is not None:
            state_dict[prefix + 'Wq_last.weight'] += self.Wq_last_lora.delta_weight().to(
                state_dict[prefix + 'Wq_last.weight'].device
            )
        if self.multi_head_combine_lora is not None:
            state_dict[prefix + 'multi_head_combine.weight'] += self.multi_head_combine_lora.delta_weight().to(
                state_dict[prefix + 'multi_head_combine.weight'].device
            )


########################################
# NN SUB CLASS / FUNCTIONS
########################################

class LoRALinearDelta(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, dropout=0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError('LoRA rank must be positive.')

        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x):
        return F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling

    def delta_weight(self):
        return torch.matmul(self.lora_B, self.lora_A) * self.scaling


def _make_lora_delta(enabled, in_features, out_features, rank, alpha, dropout):
    if not enabled:
        return None
    return LoRALinearDelta(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )


def _resolve_lora_targets(model_params):
    if not model_params.get('lora_enable', False):
        return set()

    raw_targets = model_params.get('lora_targets', ['decoder_last'])
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]

    targets = set()
    for target in raw_targets:
        if target == 'decoder_last':
            targets.update(['decoder_wq_last', 'decoder_combine'])
        elif target in {'decoder_wq_last', 'decoder_combine'}:
            targets.add(target)
        else:
            raise ValueError(f'Unsupported LoRA target: {target}')
    return targets

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / math.sqrt(key_dim)
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
