"""Knowledge Graph dataset pre-processing functions."""

import collections
import os
import pdb
import pickle

import numpy as np
import torch
import random

# 设置Python的随机种子
random.seed(42)

# 设置NumPy的随机种子
np.random.seed(42)

# 设置PyTorch的随机种子
torch.manual_seed(42)

# 如果使用GPU（CUDA），还需要设置CUDA的种子
torch.cuda.manual_seed(42)

# 不同参数的文件夹
DATA_PATH = "../data_PLA"
DATA_NAME = "NYC"
def get_idx(path):
    """Map entities and relations to unique ids.

    Args:
      path: path to directory with raw dataset files (tab-separated train/valid/test triples)

    Returns:
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids
    """
    entities, relations = set(), set()
    for split in ["train", "valid", "test"]:
        with open(os.path.join(path, split), "r") as lines:
            for line in lines:
                lhs, rel, rhs = line.strip().split("\t")
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel)
    ent2idx = {x: i for (i, x) in enumerate(sorted(entities))}
    rel2idx = {x: i for (i, x) in enumerate(sorted(relations))}

    return ent2idx, rel2idx


def to_np_array(dataset_file, ent2idx, rel2idx):
    """Map raw dataset file to numpy array with unique ids.

    Args:
      dataset_file: Path to file containing raw triples in a split
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids

    Returns:
      Numpy array of size n_examples x 3 mapping the raw dataset file to ids
    """
    examples = []
    with open(dataset_file, "r") as lines:
        for line in lines:
            lhs, rel, rhs = line.strip().split("\t")
            try:
                # pdb.set_trace()
                examples.append([ent2idx[lhs], rel2idx[rel], ent2idx[rhs]])
            except ValueError:
                continue
    return np.array(examples).astype("int64")


def get_filters(examples, n_relations):
    """Create filtering lists for evaluation.

    Args:
      examples: Numpy array of size n_examples x 3 containing KG triples
      n_relations: Int indicating the total number of relations in the KG

    Returns:
      lhs_final: Dictionary mapping queries (entity, relation) to filtered entities for left-hand-side prediction
      rhs_final: Dictionary mapping queries (entity, relation) to filtered entities for right-hand-side prediction
    """
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)
    for lhs, rel, rhs in examples:
        rhs_filters[(lhs, rel)].add(rhs)
        lhs_filters[(rhs, rel + n_relations)].add(lhs)
    lhs_final = {}
    rhs_final = {}
    for k, v in lhs_filters.items():
        lhs_final[k] = sorted(list(v))
    for k, v in rhs_filters.items():
        rhs_final[k] = sorted(list(v))
    return lhs_final, rhs_final


def process_dataset(path, dataset_name):
    """Map entities and relations to ids and saves corresponding pickle arrays.

    Args:
      path: Path to dataset directory

    Returns:
      examples: Dictionary mapping splits to with Numpy array containing corresponding KG triples.
      filters: Dictionary containing filters for lhs and rhs predictions.
    """
    # ent2idx, rel2idx = get_idx(dataset_path)

    ent2idxf = open(path+"/entity2id_NYC.txt", "r", encoding="utf-8").readlines()
    ent2idx = {}
    for line in ent2idxf:
        line = line.strip()
        k, v = line.split(" ")
        ent2idx[v] = int(v)

    rel2idxf = open(path+"/relation2id_NYC.txt", "r",
                    encoding="utf-8").readlines()
    rel2idx = {}
    for line in rel2idxf:
        line = line.strip()
        k, v = line.split(" ")
        rel2idx[v] = int(v)

    entity_idx = list(ent2idx.keys())
    relations_idx = list(rel2idx.keys())
    for i in range(len(entity_idx)):
        entity_idx[i] = int(entity_idx[i])
    for i in range(len(relations_idx)):
        relations_idx[i] = int(relations_idx[i])
    entiy_id_embeddings = np.array(entity_idx)
    relations_id_embeddings = np.array(relations_idx)

    # The index between UrbanKG id and embedding
    np.savetxt(path + "/relations_idx_embeddings.csv", relations_id_embeddings, encoding="utf-8", delimiter=",")
    np.savetxt(path + "/entity_idx_embedding.csv", entiy_id_embeddings, encoding="utf-8", delimiter=",")

    examples = {}
    splits = ["train", "valid", "test"]
    for split in splits:
        dataset_file = os.path.join(path, split)
        examples[split] = to_np_array(dataset_file, ent2idx, rel2idx)
    all_examples = np.concatenate([examples[split] for split in splits], axis=0)
    lhs_skip, rhs_skip = get_filters(all_examples, len(rel2idx))
    filters = {"lhs": lhs_skip, "rhs": rhs_skip}
    return examples, filters


if __name__ == "__main__":
    data_path = DATA_PATH
    dataset_name = DATA_NAME
    dataset_path = os.path.join(data_path, dataset_name)
    dataset_examples, dataset_filters = process_dataset(dataset_path, dataset_name)
    for dataset_split in ["train", "valid", "test"]:
        save_path = os.path.join(dataset_path, dataset_split + ".pickle")
        with open(save_path, "wb") as save_file:
            pickle.dump(dataset_examples[dataset_split], save_file)
    with open(os.path.join(dataset_path, "to_skip.pickle"), "wb") as save_file:
        pickle.dump(dataset_filters, save_file)
