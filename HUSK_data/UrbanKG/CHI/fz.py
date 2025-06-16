import os
import re
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# ==================
# 参数配置（统一放在最开头方便修改）
# ==================
kg_path = "UrbanKG_CHI.txt"
poi_csv_path = "/home/gwan700/UUKG_wgj/UUKG-main/UrbanKG_data/Processed_data/CHI/CHI_poi.csv"
output_dir = "/home/gwan700/UUKG_wgj/UUKG-main/UrbanKG_data/Processed_data/CHI/"
eps = 0.01
min_samples = 3
alpha = 0.5
beta = 0.5

def parse_kg_file(kg_path):
    poi_to_area = {}
    area_to_pois = {}
    poi_to_pc = {}

    with open(kg_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) != 3:
                continue

            subj, pred, obj = parts[0], parts[1], parts[2]

            if pred == "PLA" and subj.startswith("POI/") and obj.startswith("Area/"):
                poi_id = subj.replace("POI/", "")
                area_id = obj.replace("Area/", "")
                poi_to_area[poi_id] = area_id
                if area_id not in area_to_pois:
                    area_to_pois[area_id] = []
                area_to_pois[area_id].append(poi_id)

            if pred == "PHPC" and subj.startswith("POI/") and obj.startswith("PC/"):
                poi_id = subj.replace("POI/", "")
                pc_val = obj.replace("PC/", "")
                if poi_id not in poi_to_pc:
                    poi_to_pc[poi_id] = []
                poi_to_pc[poi_id].append(pc_val)

    return poi_to_area, area_to_pois, poi_to_pc

def load_poi_coordinates(csv_path):
    df = pd.read_csv(csv_path, dtype={'poi_id': str})
    poi_to_coords = {}
    for idx, row in df.iterrows():
        poi_id = row['poi_id']
        lat = float(row['lat'])
        lng = float(row['lng'])
        poi_to_coords[poi_id] = (lat, lng)
    return poi_to_coords

def one_hot_encode_pc(poi_to_pc):
    all_pcs = set()
    for poi_id, pcs in poi_to_pc.items():
        for pc in pcs:
            all_pcs.add(pc)
    pc_list = sorted(list(all_pcs))
    pc_to_index = {pc: i for i, pc in enumerate(pc_list)}
    poi_to_pc_vec = {}
    for poi_id, pcs in poi_to_pc.items():
        vec = np.zeros(len(pc_list), dtype=np.float32)
        for pc in pcs:
            idx = pc_to_index[pc]
            vec[idx] = 1.0
        poi_to_pc_vec[poi_id] = vec
    return pc_list, poi_to_pc_vec

def custom_distance(poi1, poi2, poi_to_coords, poi_to_pc_vec, alpha=1.0, beta=1.0):
    lat1, lng1 = poi_to_coords[poi1]
    lat2, lng2 = poi_to_coords[poi2]
    geo_dist = np.sqrt((lat1 - lat2) ** 2 + (lng1 - lng2) ** 2)

    vec1 = poi_to_pc_vec.get(poi1, None)
    vec2 = poi_to_pc_vec.get(poi2, None)
    if vec1 is None or vec2 is None:
        cos_sim = 0.0
    else:
        numerator = np.dot(vec1, vec2)
        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denominator == 0:
            cos_sim = 0.0
        else:
            cos_sim = numerator / denominator

    dist = alpha * geo_dist + beta * (1.0 - cos_sim)
    return dist

def create_distance_matrix(pois, poi_to_coords, poi_to_pc_vec, alpha=1.0, beta=1.0):
    n = len(pois)
    D = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            d = custom_distance(pois[i], pois[j], poi_to_coords, poi_to_pc_vec, alpha=alpha, beta=beta)
            D[i, j] = d
            D[j, i] = d
    return D

def cluster_area_pois(area_to_pois, poi_to_coords, poi_to_pc_vec, eps=0.01, min_samples=3, alpha=1.0, beta=1.0):
    normal_clusters = []
    noise_points = []
    global_cluster_id = 1

    for area_id, pois in area_to_pois.items():
        if len(pois) == 0:
            continue
        elif len(pois) == 1:
            normal_clusters.append((global_cluster_id, area_id, [pois[0]]))
            global_cluster_id += 1
            continue

        dist_matrix = create_distance_matrix(pois, poi_to_coords, poi_to_pc_vec, alpha, beta)

        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = db.fit_predict(dist_matrix)

        cluster_map = {}
        for poi_idx, lbl in enumerate(labels):
            poi_id = pois[poi_idx]
            if lbl not in cluster_map:
                cluster_map[lbl] = []
            cluster_map[lbl].append(poi_id)

        for lbl, poi_list in cluster_map.items():
            if lbl == -1:
                noise_points.append((area_id, poi_list))
            else:
                normal_clusters.append((global_cluster_id, area_id, poi_list))
                global_cluster_id += 1

    return normal_clusters, noise_points

def save_cluster_results(normal_clusters, noise_points, cluster_output_path, noise_output_path):
    cluster_rows = []
    for (cluster_id, area_id, pois) in normal_clusters:
        cluster_rows.append((cluster_id, area_id, ",".join(pois)))
    df_cluster = pd.DataFrame(cluster_rows, columns=["functional_zone_id", "area_id", "poi_ids"])
    df_cluster.to_csv(cluster_output_path, index=False, encoding='utf-8')

    noise_rows = []
    for (area_id, pois) in noise_points:
        noise_rows.append((-1, area_id, ",".join(pois)))
    df_noise = pd.DataFrame(noise_rows, columns=["functional_zone_id", "area_id", "poi_ids"])
    df_noise.to_csv(noise_output_path, index=False, encoding='utf-8')

def main():
    print("1) 读取KG文件...")
    poi_to_area, area_to_pois, poi_to_pc = parse_kg_file(kg_path)

    print("2) 读取POI坐标...")
    poi_to_coords = load_poi_coordinates(poi_csv_path)

    print("3) 独热编码POI的PC信息...")
    pc_list, poi_to_pc_vec = one_hot_encode_pc(poi_to_pc)

    print("4) 开始聚类 (alpha=0.5, beta=0.5, eps=0.01, min_samples=3)...")

    normal_clusters, noise_points = cluster_area_pois(
        area_to_pois, poi_to_coords, poi_to_pc_vec,
        eps=eps, min_samples=min_samples,
        alpha=alpha, beta=beta
    )

    normal_count = len(normal_clusters)
    noise_count = sum(len(pois) for _, pois in noise_points)

    cluster_output_path = os.path.join(output_dir, f"cluster_result_alpha{alpha:.2f}_beta{beta:.2f}.csv")
    noise_output_path = os.path.join(output_dir, f"noise_points_alpha{alpha:.2f}_beta{beta:.2f}.csv")

    print(f"聚类完成 => 正常簇数: {normal_count}, 噪点POI总数: {noise_count}")

    save_cluster_results(normal_clusters, noise_points, cluster_output_path, noise_output_path)
    print(f"正常簇已输出到: {cluster_output_path}")
    print(f"噪点已输出到: {noise_output_path}")

if __name__ == "__main__":
    main()
