import argparse
import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.optim
import pandas as pd
import numpy as np
import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
import torch
import random
import numpy as np

# 设置Python的随机种子
random.seed(42)

# 设置NumPy的随机种子
np.random.seed(42)

# 设置PyTorch的随机种子
torch.manual_seed(42)

# 如果使用GPU（CUDA），还需要设置CUDA的种子
torch.cuda.manual_seed(42)

DATA_PATH = 'data'

parser = argparse.ArgumentParser(
    description="Urban Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="NYC", choices=["NYC", "CHI"],
    help="Urban Knowledge Graph dataset"
)
parser.add_argument(
    "--model", default="GIE", choices=all_models, help='"TransE", "CP", "MurE", "RotE", "RefE", "AttE",'
                                                       '"ComplEx", "RotatE",'
                                                       '"RotH", "RefH", "AttH"'
                                                       '"GIE'
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adam",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=150, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=3, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--rank", default=32, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--batch_size", default=4120, type=int, help="Batch size"
)
parser.add_argument(
    "--learning_rate", default=1e-3, type=float, help="Learning rate"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"],
    help="Bias type (none for no bias)"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)


def get_embeddings(args):
    # create model
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    model = getattr(models, args.model)(args)
    model.load_state_dict(torch.load(
        # 在这里改
        os.path.join("logs/06_01/NYC/GIE_00_23_59", "model.pt")))
    entity_embeddings = model.entity.weight.detach().numpy()

    idx = pd.read_csv(DATA_PATH + '/' + args.dataset + "/entity_idx_embedding.csv", header=None)
    entity_idx = np.array(idx)

    # import pdb
    # pdb.set_trace()
    entity_final_embedddings = np.zeros([entity_embeddings.shape[0]+6, entity_embeddings.shape[1]])
    for i in range(entity_embeddings.shape[0]):
        import pdb
        try:
            entity_final_embedddings[int(entity_idx[i])] = entity_embeddings[i]
        except:
            pdb.set_trace()


    return entity_final_embedddings


def get_region_embeddings(grid_KG_id_path, entity_final_embedddings, save_path):

    grid = pd.read_csv(grid_KG_id_path)
    grid_KG_id = grid[["region_id", "KG_id"]].values
    # grid_embeddings = np.zeros([260, 32])
    grid_embeddings = np.zeros([grid_KG_id.shape[0], 32])

    for i in range(grid_embeddings.shape[0]):
        grid_embeddings[i] = entity_final_embedddings[int(grid_KG_id[i][1])]

    print(grid_embeddings)
    np.save(save_path, grid_embeddings)

def get_POI_embedding(grid_KG_id_path, entity_final_embedddings, save_path):
    poi = pd.read_csv(grid_KG_id_path)
    poi_KG_id = poi[["poi_id", "KG_id", "Region_id"]].values
    # 修改1600
    # poi_embeddings = np.zeros([62450, 33])
    poi_embeddings = np.zeros([poi_KG_id.shape[0], 33])  # 动态根据行数创建
    for i in range(poi_embeddings.shape[0]):
        poi_embeddings[i] [0:32] = entity_final_embedddings[int(poi_KG_id[i][1])]
        poi_embeddings[i] [32] =  int(poi_KG_id[i][2])

    print(poi_embeddings)
    np.save(save_path, poi_embeddings)

def get_Road_embedding(grid_KG_id_path, entity_final_embedddings, save_path):
    road = pd.read_csv(grid_KG_id_path)
    road_KG_id = road[["road_id", "KG_id", "Region_id"]].values
    # 修改2500
    # road_embeddings = np.zeros([110919, 33])
    road_embeddings = np.zeros([road_KG_id.shape[0], 33])  # 动态根据行数创建
    for i in range(road_embeddings.shape[0]):
        road_embeddings[i] [0:32] = entity_final_embedddings[int(road_KG_id[i][1])]
        road_embeddings[i] [32] =  int(road_KG_id[i][2])

    print(road_embeddings)
    np.save(save_path, road_embeddings)

# import pdb
# pdb.set_trace()
# entity_final_embedddings, pca_final_embeddings, tsne_dim2_embeddings = get_embeddings(parser.parse_args())
entity_final_embedddings = get_embeddings(parser.parse_args())
get_region_embeddings("areaid2KGid.csv", entity_final_embedddings,
                      'embedding/NYC_region_embedding.npy')

get_POI_embedding("POIid2KGid.csv", entity_final_embedddings,
                      'embedding/NYC_POI_embedding.npy')

get_Road_embedding("roadid2KGid.csv", entity_final_embedddings,
                      'embedding/NYC_Road_embedding.npy')

