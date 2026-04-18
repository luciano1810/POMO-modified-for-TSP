
import torch
import torch.nn as nn
import torch.nn.functional as F


########################################
# SOFT K-MEANS CLUSTERING (EMBEDDING SPACE)
########################################

def soft_kmeans_batch(embeddings, num_clusters, num_iters=5, temperature=1.0):
    """
    Differentiable soft k-means on node embeddings.

    Clusters in the encoder's embedding space (richer than raw coordinates).
    Soft assignments naturally handle boundary nodes — no hard cluster borders.
    Gradients flow back through the final assignment step to the encoder.

    embeddings:  (batch, problem, dim)
    Returns:
        assignments: (batch, problem, K)  soft cluster weights, sum-to-1 over K
        centroids:   (batch, K, dim)
    """
    batch, problem, dim = embeddings.shape
    device = embeddings.device

    # Initialise centroids from randomly selected node embeddings
    perm_idx = torch.randperm(problem, device=device)[:num_clusters]
    centroids = embeddings[:, perm_idx, :].detach().clone()  # (batch, K, dim)

    # Iterative EM — centroids are treated as non-differentiable during updates
    # to stabilise training (similar to stop-gradient in VQ-VAE)
    for _ in range(num_iters - 1):
        dists = torch.cdist(embeddings.detach(), centroids)      # (batch, problem, K)
        assignments = F.softmax(-dists / temperature, dim=2)     # (batch, problem, K)

        # Weighted mean update
        weights = assignments.transpose(1, 2)                    # (batch, K, problem)
        centroids = torch.bmm(weights, embeddings.detach())      # (batch, K, dim)
        centroids = centroids / weights.sum(dim=2, keepdim=True).clamp(min=1e-8)

    # Final assignment with gradients enabled so encoder can learn cluster-friendly reprs
    dists = torch.cdist(embeddings, centroids)
    assignments = F.softmax(-dists / temperature, dim=2)         # (batch, problem, K)

    return assignments, centroids


########################################
# MAIN MODEL
########################################

class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.cluster_encoder = ClusterEncoder(**model_params)
        self.confidence_net = ClusterConfidenceNetwork(**model_params)

        self.encoded_nodes = None   # (batch, problem, embedding_dim)
        self.g_node = None          # (batch, problem, embedding_dim)
        self.h_global = None        # (batch, 1, embedding_dim)
        self.step_t = 0             # decoding step counter

    def pre_forward(self, reset_state):
        problems = reset_state.problems                      # (batch, problem, 2)
        self.encoded_nodes = self.encoder(problems)          # (batch, problem, embedding_dim)

        # Soft clustering in embedding space
        # num_clusters scales with sqrt(problem_size): ~sqrt(n) nodes per cluster
        problem_size = problems.size(1)
        num_clusters = max(2, int(problem_size ** 0.5))
        assignments, _ = soft_kmeans_batch(self.encoded_nodes, num_clusters)
        _, self.g_node = self.cluster_encoder(self.encoded_nodes, assignments)
        # g_node: (batch, problem, embedding_dim)

        self.h_global = self.encoded_nodes.mean(dim=1, keepdim=True)
        # (batch, 1, embedding_dim)

        self.step_t = 0
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size  = state.BATCH_IDX.size(1)

        if state.current_node is None:
            # First step: each POMO rollout starts from its own initial node
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            problem_size = self.encoded_nodes.size(1)

            # Compute per-node cluster confidence scores
            r_i = self.confidence_net(
                self.encoded_nodes,   # h_i
                self.g_node,          # g_{c(i)}
                self.h_global,        # h_global
                self.step_t,
                problem_size,
            )
            # r_i shape: (batch, problem)

            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
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

    batch_size    = node_index_to_pick.size(0)
    pomo_size     = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# CLUSTER ENCODER
########################################

class ClusterEncoder(nn.Module):
    """
    Aggregates node embeddings by cluster (mean pooling) and projects them.
    Returns both cluster-level and per-node cluster embeddings.
    """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, encoded_nodes, assignments):
        # encoded_nodes: (batch, problem, embedding)
        # assignments:   (batch, problem, K)  soft cluster weights from soft_kmeans_batch
        batch, problem, embedding = encoded_nodes.shape

        # Soft weighted mean per cluster: (batch, K, problem) × (batch, problem, embedding)
        weights = assignments.transpose(1, 2)                        # (batch, K, problem)
        cluster_embs = torch.bmm(weights, encoded_nodes)             # (batch, K, embedding)
        cluster_embs = cluster_embs / weights.sum(dim=2, keepdim=True).clamp(min=1e-8)

        cluster_embs = self.proj(cluster_embs)                       # (batch, K, embedding)

        # Per-node cluster embedding: soft weighted sum over clusters
        # (batch, problem, K) × (batch, K, embedding) = (batch, problem, embedding)
        g_node = torch.bmm(assignments, cluster_embs)                # (batch, problem, embedding)

        return cluster_embs, g_node


########################################
# CLUSTER CONFIDENCE NETWORK
########################################

class ClusterConfidenceNetwork(nn.Module):
    """
    b_i^cluster = f(h_i, g_{c(i)}, h_global, d_t)
    Outputs raw cluster bias per node (no internal lambda — gating is done
    dynamically by GateNetwork inside the decoder).
    """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        input_dim = 3 * embedding_dim + 1   # h_i | g_{c(i)} | h_global | d_t
        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

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
        return self.net(x).squeeze(-1)       # (batch, problem)


########################################
# GATE NETWORK
########################################

class GateNetwork(nn.Module):
    """
    Dynamic gate:  λ_t = σ( g(h_t) )

    h_t = mh_atten_out from the decoder at step t.
    It already encodes: first node embedding, last selected node embedding,
    and attention-weighted summary of all remaining nodes — a compact
    representation of the partial tour state.

    The gate is a scalar per (batch, pomo) rollout, so the model can learn
    to rely on cluster bias in structured regions and ignore it elsewhere.
    """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
        )
        # Initialise output bias to -4 so σ(-4) ≈ 0.018 at the start of training.
        # The model begins as near-vanilla POMO and only opens the gate when
        # gradient evidence shows cluster bias is genuinely useful.
        nn.init.constant_(self.net[-1].bias, -4.0)

    def forward(self, mh_atten_out):
        # mh_atten_out: (batch, pomo, embedding)
        # returns:      (batch, pomo, 1)  in (0, 1)
        return torch.sigmoid(self.net(mh_atten_out))


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim    = self.model_params['embedding_dim']
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
        # input1.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # shape: (batch, HEAD_NUM, problem, KEY_DIM)

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
        head_num      = self.model_params['head_num']
        qkv_dim       = self.model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last  = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk       = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv       = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.gate_net = GateNetwork(**model_params)

        self.k              = None  # saved key,  multi-head attention
        self.v              = None  # saved value, multi-head attention
        self.single_head_key = None  # saved, single-head attention
        self.q_first        = None  # saved q_first, multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q1.shape: (batch, n, embedding)
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, ninf_mask, cluster_confidence=None):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape:         (batch, pomo, problem)
        # cluster_confidence:      (batch, problem)  or None

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = self.q_first + q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping     = self.model_params['logit_clipping']

        score_scaled  = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)

        # Dynamic gated cluster bias: logit_i = logit_i^pomo + λ_t * b_i^cluster
        # λ_t = σ(g(mh_atten_out)) — per-rollout gate in (0,1)
        # When cluster bias is uninformative, the gate learns to close (→ 0),
        # recovering vanilla POMO behaviour exactly.
        if cluster_confidence is not None:
            lambda_t = self.gate_net(mh_atten_out)          # (batch, pomo, 1)
            score_clipped = score_clipped + lambda_t * cluster_confidence.unsqueeze(1)
            # cluster_confidence: (batch, problem) → (batch, 1, problem) broadcast

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped   = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s  = q.size(0)
    head_num = q.size(1)
    n        = q.size(2)
    key_dim  = q.size(3)
    input_s  = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
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

        added     = input1 + input2
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
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
