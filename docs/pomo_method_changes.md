# Current Method Design: Latest Code vs. Vanilla POMO

## 1. 文档目的

这份文档不再按照每次 commit 的时间顺序描述修改历史，而是直接以**当前仓库最新代码**为准，说明它相对原始 POMO 方法做了哪些方法层面的扩展、为什么要这样改、这些改动在算法上分别起什么作用。

这里的对比对象是“原始 POMO 在 TSP 上的典型设定”，也就是：

- 用随机生成的二维 Euclidean 实例训练。
- 用 POMO 多起点 rollout 做策略梯度训练。
- 训练损失是标准 REINFORCE。
- 测试时直接用训练好的固定参数解码，可叠加 augmentation，但不额外做 per-instance post-training。

当前项目并没有推翻 POMO 主干，而是在它之上增加了**面向课程评测场景的适配层**和**面向性能提升的后训练层**。

---

## 2. 原始 POMO 的核心方法

为了清楚说明“改了什么”，先把原始 POMO 的核心逻辑整理一下。

### 2.1 模型主干

原始 POMO 使用 encoder-decoder 结构解决组合优化问题。在这个仓库里，这个主干仍然保留在 [TSP/POMO/TSPModel.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPModel.py:7)。

- encoder 负责把节点坐标编码成节点表示。
- decoder 结合当前状态和已访问掩码，输出下一个访问节点的分布。
- POMO 的关键不是只从一个起点 rollout，而是从多个起点并行 rollout，提高探索性并降低策略梯度的方差。

### 2.2 训练目标

基础训练仍然遵循标准 REINFORCE 形式，在 [TSP/POMO/TSPTrainer.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPTrainer.py:145)。

环境给出的 reward 是 tour length 的相反数：

$$
reward = - tour\_length
$$

因此最大化 reward 就等价于最小化路径长度。

对每个 batch 内的多条 POMO 轨迹，原始 POMO 训练的是：

$$
L_{\mathrm{RL}}
=
- \left(R - \bar{R}\right)\log \pi
$$

其中：

- $R$ 是某条 rollout 的 reward。
- $\bar{R}$ 是同一实例内多个 POMO rollout 的平均 reward。
- $\pi$ 是当前策略。

这里的 baseline 不是额外训练出来的 critic，而是直接用同一实例内部的 POMO 平均 reward 作为 advantage baseline。

### 2.3 原始 POMO 的默认假设

原始 POMO 隐含了几个假设：

1. 训练分布和测试分布接近。
2. 训练规模和测试规模接近。
3. 固定的全局参数足以泛化到测试实例。
4. 测试阶段不需要再对单个实例做额外优化。

而当前课程项目恰好会系统性破坏这些假设，这也是为什么需要在 POMO 之上继续加方法。

---

## 3. 当前项目相对原始 POMO 的总体变化

当前代码可以概括成下面这个结构：

$$
\text{POMO backbone}
\;+\;
\text{TSPLIB-aware evaluation}
\;+\;
\text{optional EAS at test time}
\;+\;
\text{preference-based post-training}
\;+\;
\text{curriculum over larger sizes}
$$

换句话说，项目的变化不是“换 backbone”，而是把原始 POMO 扩展成一个更适合课程评测场景的完整系统。

从功能上看，可以把这些变化分成两大类：

### 3.1 面向评测场景的变化

- TSPLIB 文件读取与统一评测接口。
- TSPLIB 风格整数距离计算。
- 标准测试脚本与结果导出。

### 3.2 面向性能提升的变化

- EAS test-time adaptation。
- Preference Optimization post-training。
- 大规模问题的 curriculum learning。

下面分别展开。

---

## 4. 改动一：TSPLIB-aware evaluation

### 4.1 为什么必须做这个改动

原始 POMO 更像一个“研究原型”：训练、测试、数据格式、距离定义都围绕随机 Euclidean TSP 组织。但课程项目的实际场景是：

- 输入是 `.tsp` 文件。
- 评测实例来自 TSPLIB 风格数据。
- 部分实例的最优值公开，可计算 gap。
- 最终课程验收需要统一的脚本接口和 JSON 输出。

如果不做 TSPLIB-aware adaptation，会出现三个严重错位：

1. **输入错位**  
   原始训练直接从随机坐标生成问题，而课程评测给的是 TSPLIB 文件。

2. **目标函数错位**  
   原始 POMO 默认用连续欧氏距离，而 TSPLIB 的 `EUC_2D` 和 `CEIL_2D` 是离散整数规则。

3. **评测协议错位**  
   原始代码没有统一的 hidden-test 友好接口，也没有标准化输出 `avg_aug_gap`。

### 4.2 当前代码怎么做

标准评测逻辑在 [TSP/POMO/test.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/test.py:79) 和 [TSP/POMO/TSPTester_LIB.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPTester_LIB.py:71)。

当前实现把“模型输入”和“真实评测代价”分开处理：

- 模型输入使用归一化到单位方形的坐标，[TSP/POMO/TSPTester_LIB.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPTester_LIB.py:19)。
- 真实路径长度仍用原始 TSPLIB 坐标计算，[TSP/POMO/TSPEnv.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPEnv.py:122)。

这样设计的原因很直接：

- 归一化输入更稳定，符合 POMO 原始网络的数值习惯。
- 原始坐标计分保证最终 score 和 TSPLIB 规则一致。

对于 `EUC_2D`，当前实现使用：

$$
\lfloor x + 0.5 \rfloor
$$

来模拟 TSPLIB 的 `NINT` 规则，见 [TSP/POMO/TSPEnv.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPEnv.py:139)。

### 4.3 这个改动的本质

这一步不属于“提升模型表达能力”的方法创新，而是解决**训练目标与评测目标不一致**的问题。

如果没有这一步，模型就算解得不错，也可能因为距离定义不一致而在课程评测中吃亏。

### 4.4 相对原始 POMO 的意义

原始 POMO 更关注“在标准随机 Euclidean TSP 分布上做学习”。当前项目则额外关注“让学到的策略能被课程评测协议正确调用和正确计分”。

这一步可以理解为：把 POMO 从一个研究训练脚本，扩展成一个可用于统一验收的系统。

---

## 5. 改动二：把 EAS 作为独立的 test-time adaptation 模块

### 5.1 原始 POMO 的局限

原始 POMO 在测试时默认做的是：

- 用训练好的固定参数
- 直接 rollout / argmax 解码
- 最多加一些 augmentation

这种方式的优点是简单、稳定、成本低，但它有一个明显问题：**模型参数是全局共享的，而测试实例是具体且异质的**。

对于一些结构特殊的实例，固定参数可能离最优策略还有距离。

### 5.2 为什么引入 EAS

EAS 的核心思想是：  
**不要假设一个全局固定模型已经对每个测试实例都最优，而是在测试时允许模型对当前实例做少量适配。**

这和原始 POMO 的差别很大：

- 原始 POMO：只依赖 train-time 学到的通用策略。
- 当前 EAS：在通用策略之上，再加一层 per-instance 的局部优化。

### 5.3 当前 EAS 的具体实现

当前 EAS 独立成了专门的测试入口：

- 标准推理脚本：[TSP/POMO/test.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/test.py:79)
- EAS 脚本：[TSP/POMO/test_eas.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/test_eas.py:79)
- EAS 核心实现：[TSP/POMO/TSPTester_EAS.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPTester_EAS.py:10)

每个测试实例的流程是：

1. 加载原始 checkpoint，并保存一份 base model state，[TSP/POMO/TSPTester_EAS.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPTester_EAS.py:14)。
2. 对每个实例开始前，把模型恢复到 base state，[TSP/POMO/TSPTester_EAS.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPTester_EAS.py:28)。
3. 只解冻小参数子集，而不是全模型：
   - `embedding`
   - `decoder_last`
   - `embedding_decoder`  
   选择逻辑在 [TSP/POMO/TSPModel.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPModel.py:26)。
4. 在当前实例上跑若干步基于 sampled rollout 的策略梯度更新，[TSP/POMO/TSPTester_EAS.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPTester_EAS.py:44)。
5. 记录 adaptation 过程中找到的最好解，再与最终 greedy 评估结果合并取优，[TSP/POMO/TSPTester_EAS.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPTester_EAS.py:105)。

### 5.4 当前 EAS 优化的目标

EAS 里仍然使用策略梯度思想，损失可以写成：

$$
L_{\mathrm{eas}}
=
-\left(R - \mathrm{mean}(R)\right)\log \pi
$$

这里的区别不是公式换了，而是“优化对象”变了：

- 在 base training 中，优化对象是整个训练分布上的通用参数。
- 在 EAS 中，优化对象是当前单个测试实例上的局部参数调整。

### 5.5 为什么只更新小参数子集

如果测试时直接更新整个模型，会有几个问题：

- 速度慢。
- 显存大。
- 单实例过拟合风险高。
- 对每个实例都全模型反传，成本太高。

因此当前实现只开放一个很小的参数子空间。这可以理解成在“适应能力”和“计算代价”之间做折中：

- 参数太少，适配能力不够。
- 参数太多，成本太高且不稳定。

### 5.6 为什么要把 EAS 从 `test.py` 拆出来

这是一个很重要的设计选择。当前代码明确区分：

- `test.py`：标准课程评测接口。
- `test_eas.py`：增强版实验接口。

原因是：

1. 课程组需要一个稳定、统一的标准入口。
2. EAS 是一种可选的 test-time enhancement，不应该强绑到基础评测里。
3. 在报告和实验中，需要清楚区分“模型本体性能”和“加搜索/适配之后的性能”。

### 5.7 这个改动的本质

EAS 的本质不是重新训练一个更强的全局模型，而是增加了一个**测试时的局部优化层**。

它和原始 POMO 的关系可以总结成：

- 原始 POMO解决“如何学一个通用策略”。
- EAS 解决“如何让通用策略在具体实例上再多走一步”。

---

## 6. 改动三：加入 Preference Optimization Post-Training

### 6.1 为什么原始 POMO 不够

原始 POMO 的 base training 是典型的从零开始 RL 训练。这个流程能学到一个可用的策略，但在课程项目场景下有几个现实问题：

1. 已经有一个 baseline checkpoint，可以从它继续优化，没有必要一切从零开始。
2. 纯 REINFORCE 的监督很粗，只知道“这条轨迹整体奖励高还是低”，但没有显式建模“哪条比哪条更好”。
3. 后期微调时，如果只继续用 RL，优化方向容易受采样噪声影响，提升不够稳定。

因此，当前项目引入了一个新的 post-training 阶段：  
**先保留原始 POMO 学到的能力，再用偏好优化继续把策略往更优解方向推。**

### 6.2 当前 post-training 的整体框架

入口脚本是 [TSP/POMO/post_train_preference.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/post_train_preference.py:63)，训练器是 [TSP/POMO/TSPPreferenceTrainer.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPPreferenceTrainer.py:16)。

整体结构不是换模型，而是把 post-training 写成“当前模型 + 冻结 reference model”的双模型框架：

1. 当前模型 $\pi$：需要继续更新。
2. 参考模型 $\pi_{\mathrm{ref}}$：固定不动，只负责提供对照。

初始化时，两者都从同一个 baseline checkpoint 出发，[TSP/POMO/TSPPreferenceTrainer.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPPreferenceTrainer.py:71)。

### 6.3 为什么要有 reference model

如果没有 reference，只做“当前模型采样好解然后提高它的概率”，训练很容易退化成一种 noisy self-imitation：

- 学习信号过于依赖当前模型自身的采样噪声。
- 不容易知道“当前模型到底比原先版本进步了多少”。

引入冻结的 reference model 后，就可以把问题改写成：

> 当前模型是否比旧模型更偏好好的 tour，而不是差的 tour？

这就把后训练从“只看单个解的 reward”变成了“比较两个解的相对偏好，并且比较新旧策略之间的相对偏好变化”。

### 6.4 当前 preference 数据是怎么构造的

当前代码不依赖人工标注，也不需要外部最优解标签，而是在线构造 preference pair。

过程在 [TSP/POMO/TSPPreferenceTrainer.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPPreferenceTrainer.py:361)。

对于一个 batch：

1. 当前模型先对每个实例采样多条 POMO rollout。
2. 每条 rollout 都有最终 reward。
3. reward 越大，表示 tour 越短，因为：

$$
reward = -tour\_length
$$

所以：

- reward 高的轨迹就是更好的轨迹。
- reward 低的轨迹就是更差的轨迹。

### 6.5 为什么不是单对 `best vs worst`

当前实现不是只取一条最好和一条最差，而是采用 `top-k vs bottom-k` 的多对偏好监督，[TSP/POMO/TSPPreferenceTrainer.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPPreferenceTrainer.py:349)。

也就是说，对每个实例：

- 选前 $k$ 条较好的 sampled tours。
- 选后 $k$ 条较差的 sampled tours。
- 再做笛卡尔积构造成多组偏好对。

为什么这么做：

1. 单个 `best vs worst` 监督太稀疏。
2. 极端样本可能带来较大的随机性。
3. 多对 pairwise supervision 可以让梯度更平滑、更稳定。

这个设计比单对偏好更接近“排序学习”而不是“只记住一个极端样本”。

### 6.6 reference model 为什么要“重放相同动作”

一个非常关键的设计在 [TSP/POMO/TSPModel.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPModel.py:58)，当前模型支持 `selected_override`。

这使得 reference model 不需要重新采样自己的轨迹，而是可以在**同一条动作序列**上计算 log-prob。

这样做非常重要，因为如果让 reference 自己重新采样，会出现比较对象不一致的问题：

- 当前模型和 reference model 对应的不是同一条轨迹。
- 这样就没法公平比较“同一好解/坏解在新旧策略下的偏好强度”。

通过动作重放，可以确保比较的是同一条 $y^{+}$ 和同一条 $y^{-}$。

### 6.7 当前 preference loss 的形式

当前实现本质上是 DPO 风格的偏好损失：

$$
L_{\mathrm{pref}}
=
-\log \sigma \left(
\beta \left[
\left(\log \pi(y^{+}) - \log \pi(y^{-})\right)
-
\left(\log \pi_{\mathrm{ref}}(y^{+}) - \log \pi_{\mathrm{ref}}(y^{-})\right)
\right]
\right)
$$

这里：

- $y^{+}$ 表示更好的 tour。
- $y^{-}$ 表示更差的 tour。
- $\pi$ 是当前模型。
- $\pi_{\mathrm{ref}}$ 是冻结 reference model。

这个式子的直观解释是：

- 如果当前模型相比 reference，更明显地提高了好解的相对概率并压低了差解的相对概率，那么 loss 会减小。
- 如果当前模型和 reference 差不多，或者对好坏解区分得更差，那么 loss 会变大。

### 6.8 为什么还保留 RL loss

当前实现不是只优化 preference loss，而是：

$$
L_{\mathrm{total}}
=
\lambda_{\mathrm{pref}} L_{\mathrm{pref}}
+
\lambda_{\mathrm{rl}} L_{\mathrm{rl}}
$$

见 [TSP/POMO/TSPPreferenceTrainer.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPPreferenceTrainer.py:407)。

这样设计的原因是：

1. preference loss 只关注“相对排序”，不直接保证 reward 本身朝正确方向提升。
2. 原始 RL loss 能继续给出“最终目标仍然是缩短 tour”的直接信号。
3. 两者混合后，preference 提供更细粒度监督，RL 提供更稳的全局方向。

也就是说：

- `preference loss` 负责“把排序学细”。
- `RL loss` 负责“别偏离主目标太远”。

### 6.9 这个改动相对原始 POMO 的方法意义

原始 POMO 主要回答的是：

> 如何从零开始用 RL 学出一个解 TSP 的策略？

当前 post-training 进一步回答的是：

> 已经有了一个可用策略之后，怎样在不推翻原模型的前提下，用更细的偏好监督继续把它推强？

因此，这一步是当前项目里最像“后训练方法创新”的部分。

---

## 7. 改动四：加入针对大规模实例的 Curriculum Learning

### 7.1 为什么需要课程学习

基础训练脚本 [TSP/POMO/train.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/train.py:24) 默认仍然是：

- $problem\_size = 100$
- $pomo\_size = 100$

这意味着 baseline 本质上是一个在 `TSP100` 分布上学出来的策略。

但课程的公开验证集和隐藏测试集并不局限于 `100` 节点，实际还会覆盖更大的规模。于是会出现一个明显 mismatch：

- 模型在 `100` 节点问题上的策略结构，未必能直接外推到 `150/200/300/500`。

如果直接拿 `TSP100` 的模型去打更大规模测试，常常会面临：

- 搜索空间暴涨。
- decoder 的选择分布更尖锐或更不稳定。
- 训练时从没看过的规模导致泛化下降。

### 7.2 当前课程学习怎么做

当前课程学习集成在 post-training 中，[TSP/POMO/post_train_preference.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/post_train_preference.py:87)。

默认规模序列是：

- `150`
- `200`
- `300`
- `500`

训练器会把总 epoch 切成若干阶段，[TSP/POMO/TSPPreferenceTrainer.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPPreferenceTrainer.py:119)。

如果总共训练 `100` 个 epoch，那么默认可以理解为：

- 第 1 阶段：先适应 `150`
- 第 2 阶段：再适应 `200`
- 第 3 阶段：再适应 `300`
- 第 4 阶段：最后适应 `500`

### 7.3 为什么这种由小到大的顺序合理

课程学习的核心逻辑是：

> 不要求模型一步跨越所有分布差异，而是先解决较近的迁移，再逐步面对更难的问题。

从 `100` 到 `150` 的变化，相比从 `100` 直接到 `500`，要平缓得多。

这个过程可以理解成逐层扩大搜索空间：

- 节点更多，决策步数更多。
- 候选节点更多，选择难度更高。
- 错误早期决策带来的后果更严重。

因此，先在较小增量上适应，再往更大规模扩展，是一种更稳定的迁移路径。

### 7.4 为什么课程学习放在 post-training 而不是 base training

当前项目把课程学习放进 post-training，而不是改写原始 `train.py`，这背后的考虑是：

1. 保留 baseline 的清晰性。  
   原始 POMO baseline 仍然是标准 `TSP100` 训练。

2. 把“基础能力学习”和“面向课程评测的迁移增强”分开。  
   这样在实验上更好做 ablation。

3. 计算更经济。  
   不需要从零开始就在大规模问题上训练全部 epoch。

### 7.5 当前课程学习的局限

需要强调的是，当前课程学习主要解决的是**规模迁移**，并没有完全解决**数据分布迁移**。

原因是：

- 当前 post-training 里训练样本仍然来自随机生成的 Euclidean TSP。
- 课程测试使用的是 TSPLIB 风格实例。

因此，这一步能帮助模型适应更大规模，但还不能完全等价于“在 TSPLIB 分布上后训练”。

---

## 8. 改动五：工程层支持机制

这部分不是方法创新的主角，但它们决定了上面的方法能不能在现实资源约束下跑通。

### 8.1 OOM 自动缩 batch

实现见 [TSP/POMO/TSPPreferenceTrainer.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPPreferenceTrainer.py:142)。

当课程学习进入更大规模，尤其是 `300`、`500` 时，显存占用会显著增加。当前代码在遇到 CUDA OOM 时，会：

1. 识别出是 OOM 而不是其他 RuntimeError。
2. 把当前 problem size 对应的 batch size 减半。
3. 清空 cache。
4. 继续重试。

它的价值不在于“提高指标”，而在于让更大规模的后训练在有限显存条件下仍然可运行。

### 8.2 Resume from checkpoint

实现见 [TSP/POMO/TSPPreferenceTrainer.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/TSPPreferenceTrainer.py:81) 和 [TSP/POMO/post_train_preference.py](/Users/syw/Desktop/深度学习与大模型/project/SDM-5031-2026-Spring/TSP/POMO/post_train_preference.py:70)。

恢复内容包括：

- 当前模型
- 冻结 reference model
- optimizer state
- scheduler state
- 历史日志
- epoch 位置

对于长时间 post-training，这一步很重要，因为一旦训练在中间因为显存或环境问题中断，不需要从头重跑。

---

## 9. 当前项目的方法结构应该怎样理解

当前项目最合适的理解方式，不是把它看成“一个改过的 POMO”，而是把它看成“以 POMO 为 backbone 的三阶段系统”。

### 9.1 第一阶段：Base POMO Training

用原始 POMO 思路学到一个通用基础策略：

- 主干结构不变。
- REINFORCE 不变。
- 训练规模仍然是 `TSP100` baseline。

### 9.2 第二阶段：Preference-based Post-Training with Curriculum

在已有 checkpoint 上继续训练：

- 用 frozen reference model 提供对照。
- 用 `top-k vs bottom-k` preference loss 提供更细的排序监督。
- 用 RL loss 保持原目标稳定。
- 用 curriculum 把模型逐步推向更大规模实例。

### 9.3 第三阶段：Optional Test-Time EAS

在最终推理时，还可以选择再加一层 per-instance adaptation：

- 对单个测试实例做局部优化。
- 不改变全局 checkpoint 的训练流程。
- 提供额外的性能上限。

---

## 10. 如果写课程报告，哪些是主方法，哪些是辅助设计

### 10.1 最适合写成“方法贡献”的部分

如果要突出方法创新，最值得重点写的是这两项：

1. **Preference-based Post-Training for POMO**  
   这是从训练目标层面做的增强，核心是 frozen reference model、pairwise preference、DPO-style loss、RL stabilization。

2. **Curriculum Adaptation to Larger TSP Sizes**  
   这是从训练分布层面做的增强，核心是让模型从 `TSP100` 平滑迁移到更大规模。

### 10.2 更适合作为实验增强或附加模块的部分

- **Independent EAS evaluation**
- **TSPLIB-aware standardized evaluation**

### 10.3 更适合作为工程实现细节的部分

- OOM 自动缩 batch
- Resume from checkpoint
- 统一日志与 JSON 输出

---

## 11. 最后的总结

相对原始 POMO，当前项目的变化可以一句话概括为：

> 它保留了 POMO 的 backbone 和基础强化学习训练方式，但为了适应课程评测场景和进一步提升性能，在其上叠加了 TSPLIB-aware 评测、实例级 EAS、基于偏好优化的 post-training，以及面向更大问题规模的课程学习。

如果进一步压缩成最核心的方法点，那么当前项目真正的主创新是：

$$
\text{Preference Optimization}
\;+\;
\text{Curriculum Learning}
\;+\;
\text{Optional EAS}
$$

其中：

- **Preference Optimization** 负责把后训练监督变细。
- **Curriculum Learning** 负责把模型往更大规模问题迁移。
- **EAS** 负责在测试时继续榨取单实例性能。

而 TSPLIB-aware evaluation 则确保这些改动最终能在课程的真实评测协议下被正确衡量。
