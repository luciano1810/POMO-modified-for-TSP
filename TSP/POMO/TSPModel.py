import torch
import torch.nn as nn
import torch.nn.functional as F


########################################
# K-MEANS CLUSTERING (BATCHED)
########################################

def kmeans_batch(coords, num_clusters, num_iters=10):
    """
    Batched k-means on 2D node coordinates.
    coords:  (batch, problem, 2)
    Returns: cluster_ids (batch, problem) LongTensor
             centroids  (batch, num_clusters, 2)
    """
    batch, problem, dim = coords.shape
    device = coords.device

    # Each sample gets its own random initial centroids
    perm = torch.stack([
        torch.randperm(problem, device=device)[:num_clusters]
        for _ in range(batch)
    ])  # (batch, K)
    centroids = coords.gather(
        1, perm.unsqueeze(-1).expand(batch, num_clusters, dim)
    ).clone()   # (batch, K, 2)

    cluster_ids = torch.zeros(batch, problem, dtype=torch.long, device=device)

    for _ in range(num_iters):
        dists = torch.cdist(coords, centroids)       # (batch, problem, K)
        cluster_ids = dists.argmin(dim=2)            # (batch, problem)

        idx_exp = cluster_ids.unsqueeze(-1).expand(batch, problem, dim)
        new_centroids = torch.zeros(batch, num_clusters, dim, device=device)
        new_centroids.scatter_add_(1, idx_exp, coords)
        counts = torch.zeros(batch, num_clusters, 1, device=device)
        counts.scatter_add_(
            1, cluster_ids.unsqueeze(-1),
            torch.ones(batch, problem, 1, device=device)
        )
        centroids = new_centroids / counts.clamp(min=1)

    return cluster_ids, centroids


def boundary_augment(cluster_ids, coords, centroids, boundary_ratio=0.70, swap_prob=0.5):
    """
    Training augmentation: randomly re-assign boundary nodes to a neighbouring cluster.
    boundary_ratio=0.70 (looser than 0.85) captures more boundary nodes.
    """
    batch, problem = cluster_ids.shape
    device = cluster_ids.device

    dists = torch.cdist(coords, centroids)          # (batch, problem, K)
    sorted_dists, sorted_idx = dists.sort(dim=2)

    nearest_dist   = sorted_dists[:, :, 0]          # (batch, problem)
    second_dist    = sorted_dists[:, :, 1]          # (batch, problem)
    second_cluster = sorted_idx[:, :, 1]            # (batch, problem)

    ratio = nearest_dist / second_dist.clamp(min=1e-8)
    is_boundary = ratio > boundary_ratio            # close to cluster boundary

    swap_mask    = torch.rand(batch, problem, device=device) < swap_prob
    augment_mask = is_boundary & swap_mask

    return torch.where(augment_mask, second_cluster, cluster_ids)


########################################
# MAIN MODEL
########################################

class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder        = TSP_Encoder(**model_params)
        self.decoder        = TSP_Decoder(**model_params)
        self.cluster_encoder = ClusterEncoder(**model_params)
        self.confidence_net = ClusterConfidenceNetwork(**model_params)

        self.encoded_nodes = None   # (batch, problem, embedding_dim)
        self.g_node        = None   # (batch, problem, embedding_dim)
        self.h_global      = None   # (batch, 1, embedding_dim)
        self.step_t        = 0

    def pre_forward(self, reset_state):
        problems = reset_state.problems                     # (batch, problem, 2)
        self.encoded_nodes = self.encoder(problems)         # (batch, problem, embedding_dim)

        problem_size = problems.size(1)
        num_clusters = max(2, int(problem_size ** 0.5))
        cluster_ids, centroids = kmeans_batch(problems, num_clusters)

        if self.training:
            cluster_ids = boundary_augment(cluster_ids, problems, centroids)

        _, self.g_node = self.cluster_encoder(
            self.encoded_nodes, cluster_ids, num_clusters
        )
        # g_node: (batch, problem, embedding_dim)

        self.h_global = self.encoded_nodes.mean(dim=1, keepdim=True)
        # (batch, 1, embedding_dim)

        self.step_t = 0
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size  = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            self.decoder.set_q1(encoded_first_node)

        else:
            problem_size = self.encoded_nodes.size(1)

            r_i = self.confidence_net(
                self.encoded_nodes,
                self.g_node,
                self.h_global,
                self.step_t,
                problem_size,
            )
            # r_i shape: (batch, problem)

            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(
                encoded_last_node,
                ninf_mask=state.ninf_mask,
                cluster_confidence=r_i,
            )
            # shape: (batch, pomo, problem)

            self.step_t += 1

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = None

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size    = node_index_to_pick.size(0)
    pomo_size     = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes    = encoded_nodes.gather(dim=1, index=gathering_index)

    return picked_nodes


########################################
# CLUSTER ENCODER
########################################

class ClusterEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, encoded_nodes, cluster_ids, num_clusters):
        # encoded_nodes: (batch, problem, embedding)
        # cluster_ids:   (batch, problem)
        batch, problem, embedding = encoded_nodes.shape
        device = encoded_nodes.device

        idx_exp = cluster_ids.unsqueeze(-1).expand(batch, problem, embedding)

        cluster_embs = torch.zeros(batch, num_clusters, embedding, device=device)
        cluster_embs.scatter_add_(1, idx_exp, encoded_nodes)
        counts = torch.zeros(batch, num_clusters, 1, device=device)
        counts.scatter_add_(
            1, cluster_ids.unsqueeze(-1),
            torch.ones(batch, problem, 1, device=device)
        )
        cluster_embs = cluster_embs / counts.clamp(min=1)  # (batch, K, embedding)
        cluster_embs = self.proj(cluster_embs)              # (batch, K, embedding)

        g_node = cluster_embs.gather(1, idx_exp)            # (batch, problem, embedding)

        return cluster_embs, g_node


########################################
# CLUSTER CONFIDENCE NETWORK
########################################

class ClusterConfidenceNetwork(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        input_dim = 3 * embedding_dim + 1   # h_i | g_{c(i)} | h_global | d_t
        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        # Small initial lambda: let POMO stabilise before confidence kicks in
        self.lambda_conf = nn.Parameter(torch.tensor(0.1))

    def forward(self, h_nodes, g_node, h_global, step_t, problem_size):
        # h_nodes:  (batch, problem, embedding)
        # g_node:   (batch, problem, embedding)
        # h_global: (batch, 1, embedding)
        batch, problem, embedding = h_nodes.shape

        h_global_exp = h_global.expand(batch, problem, embedding)
        d_t = torch.full(
            (batch, problem, 1), step_t / problem_size, device=h_nodes.device
        )

        x = torch.cat([h_nodes, g_node, h_global_exp, d_t], dim=-1)
        r = self.net(x).squeeze(-1)                      # (batch, problem)
        return self.lambda_conf * torch.tanh(r)          # bounded output


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params  = model_params
        embedding_dim      = self.model_params['embedding_dim']
        encoder_layer_num  = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(**model_params) for _ in range(encoder_layer_num)]
        )

    def forward(self, data):
        embedded_input = self.embedding(data)
        out = embedded_input
        for layer in self.layers:
            out = layer(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num      = self.model_params['head_num']
        qkv_dim       = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward          = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat     = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num      = self.model_params['head_num']
        qkv_dim       = self.model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last  = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk       = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv       = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k               = None
        self.v               = None
        self.single_head_key = None
        self.q_first         = None

    def set_kv(self, encoded_nodes):
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def set_q1(self, encoded_q1):
        head_num = self.model_params['head_num']
        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)

    def forward(self, encoded_last_node, ninf_mask, cluster_confidence=None):
        head_num = self.model_params['head_num']

        q_last     = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        q          = self.q_first + q_last
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        score = torch.matmul(mh_atten_out, self.single_head_key)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping     = self.model_params['logit_clipping']

        score_scaled  = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)

        # u'_i = u_i + λ * tanh(r_i)
        # cluster_confidence: (batch, problem) → broadcast over pomo dim
        if cluster_confidence is not None:
            score_clipped = score_clipped + cluster_confidence.unsqueeze(1)

        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    batch_s      = qkv.size(0)
    n            = qkv.size(1)
    q_reshaped   = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s  = q.size(0)
    head_num = q.size(1)
    n        = q.size(2)
    key_dim  = q.size(3)
    input_s  = k.size(2)

    score        = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(
            batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(
            batch_s, head_num, n, input_s)

    weights    = nn.Softmax(dim=3)(score_scaled)
    out        = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added      = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)
        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))