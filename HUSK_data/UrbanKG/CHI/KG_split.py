# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch

# 设置Python的随机种子
random.seed(42)

# 设置NumPy的随机种子
np.random.seed(42)

# 设置PyTorch的随机种子
torch.manual_seed(42)

# 如果使用GPU（CUDA），还需要设置CUDA的种子
torch.cuda.manual_seed(42)


def load_id_dict(entity2id_file, relation2id_file):
    """
    读取实体和关系的编号文件，返回两个字典：
    entity2id: {实体字符串: 实体id}
    relation2id: {关系字符串: 关系id}
    """
    entity2id = {}
    relation2id = {}

    # 读取 entity2id
    with open(entity2id_file, "r", encoding="utf-8") as f_ent:
        for line in f_ent:
            line = line.strip()
            if not line:
                continue
            ent_str, ent_id = line.split()
            entity2id[ent_str] = int(ent_id)

    # 读取 relation2id
    with open(relation2id_file, "r", encoding="utf-8") as f_rel:
        for line in f_rel:
            line = line.strip()
            if not line:
                continue
            rel_str, rel_id = line.split()
            relation2id[rel_str] = int(rel_id)

    return entity2id, relation2id


def load_kg_triples(kg_file):
    """
    从UrbanKG_NYC.txt中读取三元组，返回列表[(head, relation, tail), ...]
    """
    triples = []
    with open(kg_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 假设每行格式: head  rel  tail
            head, rel, tail = line.split()
            triples.append((head, rel, tail))
    return triples


def convert_triples_to_id(triples, entity2id, relation2id):
    """
    将(实体字符串, 关系字符串, 实体字符串)的三元组列表转换成
    (实体id, 关系id, 实体id)的列表。
    """
    triple_ids = []
    for (h, r, t) in triples:
        # 有可能出现无法在字典中找到的情况，这里假设必然能找到
        h_id = entity2id[h]
        r_id = relation2id[r]
        t_id = entity2id[t]
        triple_ids.append((h_id, r_id, t_id))
    return triple_ids


def write_triple_ids(triple_ids, output_file):
    """
    将(实体id, 关系id, 实体id)的列表写入文件，每行 head_id rel_id tail_id
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for (h_id, r_id, t_id) in triple_ids:
            f.write(f"{h_id}\t{r_id}\t{t_id}\n")


def generate_test_valid_if_not_exist(triples, entity2id, relation2id,
                                     test_file="test2id_NYC.txt",
                                     valid_file="valid2id_NYC.txt",
                                     chosen_relations=None):
    """
    如果 test_file 和 valid_file 不存在，则按照指定逻辑生成。

    参数chosen_relations:
    - None: 从所有三元组随机抽取
    - 非None: 仅从与 chosen_relations 中任一relation匹配的三元组里抽取

    注意：
    - 这里的5%与5%是针对“选定子集”来做的随机抽取。
    - 如果chosen_relations非常小，可能拿不到足量数据。此时就随机取5%做test，5%做valid，实际数量不足也会照常生成。
    """
    # 如果test和valid都存在，就直接读取
    if os.path.exists(test_file) and os.path.exists(valid_file):
        print("检测到已有测试集与验证集文件，跳过生成。")
        # 读取已有的test, valid
        test_data = []
        valid_data = []
        with open(test_file, "r", encoding="utf-8") as f_test:
            for line in f_test:
                line = line.strip()
                if not line:
                    continue
                h_id, r_id, t_id = line.split()
                test_data.append((int(h_id), int(r_id), int(t_id)))
        with open(valid_file, "r", encoding="utf-8") as f_valid:
            for line in f_valid:
                line = line.strip()
                if not line:
                    continue
                h_id, r_id, t_id = line.split()
                valid_data.append((int(h_id), int(r_id), int(t_id)))
        return test_data, valid_data
    else:
        # 需要重新生成 test、valid
        if chosen_relations is None:
            # 全部三元组做抽样
            eligible_triples = triples
        else:
            # 仅使用指定关系的三元组
            chosen_set = set(chosen_relations)  # 如果是单个relation，可写 [chosen_relations]
            eligible_triples = [(h, r, t) for (h, r, t) in triples if r in chosen_set]

        # 打乱
        random.shuffle(eligible_triples)

        # 计算切分位置
        total_count = len(eligible_triples)
        test_size = int(total_count * 0.05)
        valid_size = int(total_count * 0.05)

        test_triples = eligible_triples[:test_size]
        valid_triples = eligible_triples[test_size: test_size + valid_size]

        # 转成id
        test_ids = convert_triples_to_id(test_triples, entity2id, relation2id)
        valid_ids = convert_triples_to_id(valid_triples, entity2id, relation2id)

        # 写文件
        write_triple_ids(test_ids, test_file)
        write_triple_ids(valid_ids, valid_file)

        print(f"已生成 {test_file} 和 {valid_file}。")
        print(f"test 数量: {len(test_triples)}  valid 数量: {len(valid_triples)}")

        return test_ids, valid_ids


def create_train(triples, test_data, valid_data, entity2id, relation2id,
                 train_file="train2id_NYC.txt"):
    """
    根据test_data、valid_data生成train，并考虑以下特殊规则：
    1. 如果test+valid不止一种relation，train去掉test+valid即可；
    2. 如果test+valid只有一种relation:
       - 若是PHPC，train仅排除test+valid本身；
       - 若是PLA，除了排除test+valid本身，还要排除所有包含test+valid中出现过的POI实体，且关系 ∈ {PBB, PLF, PLR} 的三元组。
         注意：只排除 test+valid 中出现的 POI，如果一个POI没有在test+valid出现，就不管。
    最后将生成的train三元组写到文件(仅id形式)。
    """
    # 将 test_data, valid_data 转回 (h_str, r_str, t_str) 好做统计
    # 因为我们需要识别POI实体（字符串开头是否是"POI/"？）等信息
    # 这里我们做一个 逆字典 映射 id->实体/关系字符串
    rev_entity = {v: k for k, v in entity2id.items()}
    rev_relation = {v: k for k, v in relation2id.items()}

    # 把原始 triples 做一个 set 方便排除
    # triples: [(h_str, r_str, t_str), ...]
    triple_set = set(triples)

    # 先构造 test+valid 的原始字符串三元组
    # test_data, valid_data 目前是 [(h_id, r_id, t_id), ...]
    def to_str_triple(h_id, r_id, t_id):
        return (rev_entity[h_id], rev_relation[r_id], rev_entity[t_id])

    test_str_triples = [to_str_triple(*x) for x in test_data]
    valid_str_triples = [to_str_triple(*x) for x in valid_data]

    # 所有 test+valid 三元组
    all_test_valid_str = test_str_triples + valid_str_triples

    # 先收集 test+valid 的所有关系
    rel_set = set([r for (_, r, _) in all_test_valid_str])

    # 构造一个排除用的 set
    test_valid_set = set(all_test_valid_str)

    # 如果 test+valid中只有一种关系
    if len(rel_set) == 1:
        only_rel = list(rel_set)[0]  # 取出那唯一的关系

        if only_rel == "PHPC":
            # train 不能包含 test+valid 的三元组
            # 其它关系不做特殊处理
            pass  # 无需额外操作

        elif only_rel == "PLA":
            # 除了去掉 test+valid，本身，还需去掉:
            #   含有 test+valid 中出现过的 POI 实体 并且 relation ∈ {PBB, PLF, PLR}
            poi_entities = set()
            for (h, _, t) in all_test_valid_str:
                # 如果实体是 "POI/xxx" 则加入 set
                if h.startswith("POI/"):
                    poi_entities.add(h)
                if t.startswith("POI/"):
                    poi_entities.add(t)

            exclude_relations = {"PBB", "PLF", "PLR"} # 在这里修改删除掉的关系

            # 从 triple_set 中再排除这部分
            # 只要三元组的关系 ∈ exclude_relations，且 (h 或 t 在 poi_entities)，则排除
            to_remove = set()
            for (h, r, t) in triple_set:
                if r in exclude_relations:
                    if h in poi_entities or t in poi_entities:
                        to_remove.add((h, r, t))

            # 统一在外面做差集
            test_valid_set = test_valid_set.union(to_remove)

        elif only_rel == "PLR":
            pass

        elif only_rel == "PLF":
            # 除了去掉 test+valid，本身，还需去掉:
            #   含有 test+valid 中出现过的 POI 实体 并且 relation ∈ {PBB, PLF, PLR}
            poi_entities = set()
            for (h, _, t) in all_test_valid_str:
                # 如果实体是 "POI/xxx" 则加入 set
                if h.startswith("POI/"):
                    poi_entities.add(h)
                if t.startswith("POI/"):
                    poi_entities.add(t)

            exclude_relations = {"PBB", "PLA"} # 在这里修改删除掉的关系

            # 从 triple_set 中再排除这部分
            # 只要三元组的关系 ∈ exclude_relations，且 (h 或 t 在 poi_entities)，则排除
            to_remove = set()
            for (h, r, t) in triple_set:
                if r in exclude_relations:
                    if h in poi_entities or t in poi_entities:
                        to_remove.add((h, r, t))

            # 统一在外面做差集
            test_valid_set = test_valid_set.union(to_remove)

        else:
            # 如果后续扩展了别的关系的特殊逻辑，可以在这里加
            # 当前仅实现PHPC, PLA两个分支
            pass

    # 最终可进入train的三元组 = triple_set - test_valid_set
    train_str_triples = list(triple_set - test_valid_set)

    # 将 train_str_triples 转为 id
    train_ids = convert_triples_to_id(train_str_triples, entity2id, relation2id)

    # 写入文件
    write_triple_ids(train_ids, train_file)
    print(f"已生成 {train_file}，其中包含 {len(train_ids)} 条三元组。")

    # 返回方便后续做统计
    return train_ids


def split_kg(
        kg_file="UrbanKG_NYC.txt",
        entity2id_file="entity2id_NYC.txt",
        relation2id_file="relation2id_NYC.txt",
        chosen_relations=None
):
    """
    主函数：
    1. 读取或生成 test、valid
    2. 生成 train（并考虑特殊逻辑）
    3. 输出最终结果
    """
    # 读取 id 映射
    entity2id, relation2id = load_id_dict(entity2id_file, relation2id_file)

    # 读取全部原始三元组
    all_triples = load_kg_triples(kg_file)  # [(h, r, t), ...]

    # 生成或读取 test、valid
    test_data, valid_data = generate_test_valid_if_not_exist(
        triples=all_triples,
        entity2id=entity2id,
        relation2id=relation2id,
        test_file="test2id_NYC.txt",
        valid_file="valid2id_NYC.txt",
        chosen_relations=chosen_relations
    )

    # 生成 train
    create_train(
        triples=all_triples,
        test_data=test_data,
        valid_data=valid_data,
        entity2id=entity2id,
        relation2id=relation2id,
        train_file="train2id_NYC.txt"
    )
    print("数据拆分完成！")


if __name__ == "__main__":
    """
    使用示例：
    1）不传relation参数，随机抽取：
       python split_kg.py

    2）只让test和valid包含PLA关系：
       python split_kg.py --rel PLA

    可根据需要自行扩展。
    """
    # 示例中直接调用，无额外传参（不指定relation）
    split_kg(
        kg_file="UrbanKG_CHI_PLR_withFZ_FHPC.txt",
        entity2id_file="entity2id_CHI.txt",
        relation2id_file="relation2id_CHI.txt"# ,
        # chosen_relations=["PLR"]  # 或者传 ["PLA"] / ["PHPC"] 等 ###########################################################
    )
