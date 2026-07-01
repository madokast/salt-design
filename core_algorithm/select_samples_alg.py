

# 第一轮选择
def first_round_select(
    Xs, # 全部无标签数据，embedding 向量格式
    r, # 半径，首轮运行时，所有数据点用人工指定的半径
    budget, # 预算，最大人工标注的样本数目
):
    # 选中的样本索引，最终返回
    selected_indices = []  
    # 移除候选的索引，不能被选中（包括以选择的点，及其覆盖的点）
    discarded_indices = set()
    
    # 贪心选择
    while len(selected_indices) < budget:
        # 计算每个点在半径 r 球内覆盖 Xs 中点的数目，作为分数
        scores = [cover_number(X, Xs, r) for X in Xs]
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
    radii = [get_radius(Xs[i], labele_dict[i], model, r_max) for i in labeled_indices]

    # 计算总的覆盖率
    cover_ratio = get_cover_ratio(Xs, radii, labeled_indices)
    
    # 覆盖率低于20%，则扩大半径，然后重新计算覆盖率
    if cover_ratio < 0.2: 
        radii = enlarge_radii(radii, r_max) # 75%分位拉伸到 r_max，75%分位以下按同比例拉伸
        cover_ratio = get_cover_ratio(Xs, radii, labeled_indices) # 重新计算覆盖率
    
    # 所有未标注点作为候选
    candidate_indices = set(range(len(Xs))) - set(labeled_indices)

    # 计算每个候选点的静态分数 index -> static_score
    static_scores = {get_static_score(Xs[i], cover_ratio, model):
         i for i in candidate_indices}

    # 选中的样本索引，最终返回
    selected_indices = []
    # 移除候选的索引，不能被选中（包括以选择的点，及其覆盖的点）
    discarded_indices = set()
    # 贪心选择
    while len(selected_indices) < budget:
        # 计算每个候选点的预估半径，被人工样本覆盖的点，半径为 0；半径最大限制为 r_max
        candidate_radii = {
            i: 0 if cover_by_labeled(Xs[i], radii, labeled_indices, r_max)
                else get_candidate_radius(Xs[i], model)
                    for i in candidate_indices }
        
        # 计算每个点覆盖的样本数据 dict。半径 0 的点只覆盖自身
        cover_numbers = get_cover_numbers(Xs, candidate_radii, r_max)

        # 计算每个点的分数 index -> score。静态分数 + 覆盖得分
        scores = {i: static_scores[i] + np.log(cover_numbers[i]) for i in candidate_indices}
        # 选择 discarded_indices外 分数最高的点的索引，输出
        max_id = argmax(scores, discarded_indices)
        selected_indices.append(max_id)
        # max_id 覆盖的点（包括自身），分数设为 -1，不再被取出
        for i in cover_indices(Xs[max_id], Xs, candidate_radii[max_id]):
            discarded_indices.add(i)
    return selected_indices

