# 第一轮选择
def first_round_select(
    Xs, # 全部无标签数据，embedding 向量格式
    r, # 半径，首轮运行时，所有数据点用人工指定的半径
    budget, # 预算，最大人工标注的样本数目
):
    # 选中的样本索引，最终返回
    selected_indices = []  
    # 移除候选的索引，不能被选中（包括已选择的点，及其覆盖的点）
    discarded_indices = set()
    
    # 贪心选择
    while len(selected_indices) < budget:
        # 计算每个点在半径 r 球内覆盖 Xs 中点的数目，作为分数
        scores = [get_cover_number(X, Xs, r) for X in Xs]
        # 选择分数最高的点的索引，但是被移除的点除外
        max_id = argmax(scores, discarded_indices)
        selected_indices.append(max_id)
        # max_id 覆盖的点（包括自身），分数设为 -1，不再被取出
        for i in cover_indices(Xs[max_id], Xs, r):
            discarded_indices.add(i)
    return selected_indices


# 后续轮选择
def subsequent_round_select(
    Xs, # 全部无标签数据，embedding 向量格式
    labeled_indices, # 前面所有轮次，人工标注的数据的索引列表
    label_dict, # 前面所有轮次，人工标注结果
    model, # 当前训练的模型
    r_max, # 最大半径，计算的半径最大限制为 r_max
    budget, # 预算，最大人工标注的样本数目
):
    # 计算每个人工标注点的半径，半径最大限制为 r_max
    # get_radius 具体算法有单独伪代码描述
    radii = [get_radius(Xs[i], label_dict[i], Xs, model) for i in labeled_indices]

    # 计算总的覆盖率
    cover_ratio = get_cover_ratio(Xs, radii, labeled_indices)
    
    # 覆盖率低于20%，则按照 r_max 扩大半径，然后重新计算覆盖率
    if cover_ratio < 0.2: 
        radii = enlarge_radii(radii, r_max) # 75%分位拉伸到 r_max，75%分位以下按同比例拉伸
        cover_ratio = get_cover_ratio(Xs, radii, labeled_indices) # 重新计算覆盖率
    
    # 所有未被人工标注样本覆盖的点，用于后续计算静态分数（注意不是候选池）
    uncovered_indices = get_uncovered_indices(Xs, radii, labeled_indices)
    Xs_uncovered = [Xs[i] for i in uncovered_indices]

    # 所有未标注点作为候选
    candidate_indices = set(range(len(Xs))) - set(labeled_indices)

    # 计算每个候选点的预估半径，被人工样本覆盖的点，半径为 0
    candidate_radii = {
        i: 0 if cover_by_labeled(Xs[i], radii, labeled_indices, r_max)
            else get_candidate_radius(Xs[i], Xs, model)
                for i in candidate_indices }

    # 计算每个候选点的静态分数 index -> static_score
    static_scores = get_static_scores(candidate_indices, candidate_radii, Xs_uncovered, model, cover_ratio)

    # 选中的样本索引，最终返回
    selected_indices = []
    # 移除候选的索引，不能被选中（包括已选择的点，及其覆盖的点）
    discarded_indices = set()
    # 贪心选择
    while len(selected_indices) < budget:
        # 计算每个点在 Xs_uncovered 中覆盖的样本数据。半径 0 的点只覆盖自身
        cover_numbers = get_cover_numbers(candidate_indices, candidate_radii, Xs_uncovered)

        # 计算每个点的分数 = 静态分数 + 覆盖得分
        scores = {i: static_scores[i] + np.log(cover_numbers[i]) for i in candidate_indices}
        # 选择 discarded_indices 外 分数最高的点的索引，输出
        max_id = argmax(scores, discarded_indices)
        selected_indices.append(max_id)
        # max_id 覆盖的点（包括自身），分数设为 -1，不再被取出
        for i in cover_indices(Xs[max_id], Xs, candidate_radii[max_id]):
            discarded_indices.add(i)
    return selected_indices

# 计算人工标注点的半径
def get_radius(
    X, # 人工标注样本
    y, # 人工标签
    Xs, # 全量数据
    model, # 当前训练的模型
):
    # 其他超参数
    diff_min = 0.0 # diff 的下界
    h_min = 1e-6 # h 的下界
    lipschitz_min = 1e-12 # Lipschitz 的下界
    knn_distance_eps = 1e-12 # knn 忽略距离小于此的点
    epsilon = 0.01

    logits_x = model(X) # 模型最后一层输出的原始数值
    probs = softmax(logits_x) # 概率分布，例如 (0.7, 0.3)
    eta = one_hot(y) # 将人工标签转为独热分布，例如 (1.0, 0.0)
    diff = L2_distance(probs, eta) # 模型预测和人工标签的距离
    diff = max(diff, diff_min) # 限制 diff 的下界

    H = diag(probs) - probs * probs.T # 概率协方差矩阵
    h = spectral_norm(H) # 谱范数
    h = max(h, h_min) # 限制 h 的下界

    neighbors = KNN(Xs, X, k=50) # 找到 50 个最近邻点
    lipschitz_values = []
    for N in neighbors:
        distance = L2_distance(X, N) # 样本距离
        if distance < knn_distance_eps:
            continue
        logits_n = model(N)
        logits_diff = L2_distance(logits_x, logits_n) # logits 距离
        lipschitz_n = logits_diff / distance # logits 距离 / 样本距离
        lipschitz_values.append(lipschitz_n) #
    L = percentile(lipschitz_values, 90) # 取 90% 分位数，作为局部 Lipschitz 放大系数
    L = max(L, lipschitz_min) # 限制 Lipschitz 的下界

    # 半径公式
    r = (-diff + sqrt(diff**2 + 2 * h * epsilon)) / (h * L)

    # 计算这个点的理论最大距离（和 GPT 详细确认）
    sqrt_term = sqrt(diff_min**2 + 2 * h_min * epsilon)
    r_max = 2 * epsilon / (L * (diff_min + sqrt_term))

    r = min(r, r_max) # 限制半径最大值
    return r


# 计算未标记点（候选点）的估计半径
def get_candidate_radius(
    X, # 当前样本数据
    Xs, # 全量数据
    model, # 当前训练的模型
):
    # 计算过程和上面的 get_radius 一模一样
    # 只是不计算 diff，用 diff_min 代替 diff
    pass


# 计算未标记点的静态分数
def get_static_scores(
    candidate_indices, # 候选样本索引
    candidate_radii, # 候选样本的估计半径
    Xs_uncovered, # 全量未被人工标注样本覆盖的点
    model, # 当前训练的模型
    cover_ratio, # 覆盖率
):
    # 计算每个候选样本在 Xs_uncovered 中覆盖点的数目（半径为 0 的点只覆盖自身）
    cover_numbers = {i: get_cover_number(Xs[i], Xs_uncovered, candidate_radii[i]) for i in candidate_indices}
    log_cover_numbers = {i: log(cover_numbers[i]) for i in candidate_indices} # 转为对数

    # 计算每个候选样本的模型预测不确定程度
    uncertainties = {}
    for i in candidate_indices:
        logits_x = model(Xs[i]) # 模型最后一层输出的原始数值
        probs = softmax(logits_x) # 概率分布，例如 (0.7, 0.3)
        margin = max(probs) - second_max(probs) # 最大类别概率 - 第二大类别概率
        uncertainty = -log(margin) # 不确定性
        uncertainties[i] = uncertainty

    # 计算 static 分数的权重
    # IQR 指的是 75%分位数 - 25%分位数
    # 用户也可以指定 alpha，则直接使用指定值，例如 alpha=0.1
    alpha = IQR(log_cover_numbers) / IQR(uncertainties)

    # 计算静态分数 alpha * cover_ratio * uncertainty，前两项固定，最后一项是点自己的不确定程度
    static_scores = {i: alpha * cover_ratio * uncertainties[i] for i in candidate_indices}
    return static_scores
