import csv
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

# region
# ——————————————————————————————————————————————————————————————————————————
urban_file = '../UrbanKG_data/UrbanKG/NYC/UrbanKG_NYC_PLR_withFZ_FHPC.txt'
entity_file = 'data/NYC/entity2id_NYC.txt'
output_file = 'areaid2KGid.csv'

with open(entity_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    # 写入列名
    writer.writerow(['region_id', 'KG_id'])

    for line in f_in:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # 假设文件中每一行至少有两列，第一列为字符串，第二列为id
        if len(parts) >= 2:
            entity, entity_id = parts[0], parts[1]
            # 检查第一列是否包含 "area"
            if 'Area' in entity:
                writer.writerow([entity, entity_id])


# POI
output_file = 'POIid2KGid.csv'

# 创建一个字典，用于从urban_file中根据poi_id获取region_id
poi_to_region = {}
with open(urban_file, 'r', encoding='utf-8') as f_urban:
    for line in f_urban:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        # 假设行格式为：poi_id PLA region_id，例如：POI/1175120 PLA Area/27
        if len(parts) == 3:
            poi_id, label, region_full = parts
            # 我们仅对第二列为"PLA"的行处理
            if label == 'PLA':
                region_id = region_full.split('/')[-1]
                poi_to_region[poi_id] = region_id

with open(entity_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    # 写入列名
    writer.writerow(['poi_id', 'KG_id', 'Region_id'])

    for line in f_in:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # 假设entity_file中格式为: poi_id KG_id
        if len(parts) >= 2:
            poi_id, KG_id = parts[0], parts[1]
            # 检查第一列是否包含 "POI"
            if 'POI' in poi_id:
                # 在poi_to_region中查找对应的region_id
                if poi_id in poi_to_region:
                    region_id = poi_to_region[poi_id]
                    writer.writerow([poi_id, KG_id, region_id])

# road
output_file = 'roadid2KGid.csv'

# 创建一个字典，用于从urban_file中根据road_id获取region_id
road_to_region = {}
with open(urban_file, 'r', encoding='utf-8') as f_urban:
    for line in f_urban:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        # 假设行格式为：road_id PLA region_id，例如：POI/1175120 PLA Area/27
        if len(parts) == 3:
            road_id, label, region_full = parts
            # 我们仅对第二列为"PLA"的行处理
            if label == 'RLA':
                region_id = region_full.split('/')[-1]
                road_to_region[road_id] = region_id

with open(entity_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    # 写入列名
    writer.writerow(['road_id', 'KG_id', 'Region_id'])

    for line in f_in:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # 假设entity_file中格式为: road_id KG_id
        if len(parts) >= 2:
            road_id, KG_id = parts[0], parts[1]
            # 检查第一列是否包含 "Road"
            if 'Road' in road_id:
                # 在road_to_region中查找对应的region_id
                if road_id in road_to_region:
                    region_id = road_to_region[road_id]
                    writer.writerow([road_id, KG_id, region_id])