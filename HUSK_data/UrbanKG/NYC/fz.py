import os
import re
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


# ==============
# 1. 读取KG文件
# ==============

def parse_kg_file(kg_path):
    """
    解析 UrbanKG_NYC.txt 文件，获取：
    1) poi -> area 的映射（来自 "POI/x PLA Area/y"）
    2) poi -> pc   的类别（来自 "POI/x PHPC PC/xxx"）
    返回:
       poi_to_area: {poi_id: area_id}
       area_to_pois: {area_id: [poi_id1, poi_id2, ...]}
       poi_to_pc: {poi_id: [pc1, pc2, ...]} # 一个POI可能存在多个PC
    """
    poi_to_area = {}
    area_to_pois = {}
    poi_to_pc = {}

    with open(kg_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 为空或注释行则跳过
            if not line or line.startswith('#'):
                continue

            # 根据空格分割
            parts = line.split()
            if len(parts) != 3:
                continue

            subj, pred, obj = parts[0], parts[1], parts[2]

            # 解析 POI -> Area 关系 (PLA)
            if pred == "PLA" and subj.startswith("POI/") and obj.startswith("Area/"):
                poi_id = subj.replace("POI/", "")
                area_id = obj.replace("Area/", "")
                poi_to_area[poi_id] = area_id
                if area_id not in area_to_pois:
                    area_to_pois[area_id] = []
                area_to_pois[area_id].append(poi_id)

            # 解析 POI -> PC 关系 (PHPC)
            if pred == "PHPC" and subj.startswith("POI/") and obj.startswith("PC/"):
                poi_id = subj.replace("POI/", "")
                pc_val = obj.replace("PC/", "")
                if poi_id not in poi_to_pc:
                    poi_to_pc[poi_id] = []
                poi_to_pc[poi_id].append(pc_val)

    return poi_to_area, area_to_pois, poi_to_pc


# =================
# 2. 读取POI坐标信息
# =================

def load_poi_coordinates(csv_path):
    """
    从 NYC_poi.csv 中读取 poi_id, lat, lng,
    返回 {poi_id: (lat, lng)}
    """
    df = pd.read_csv(csv_path, dtype={'poi_id': str})  # 确保poi_id以字符串处理
    poi_to_coords = {}
    for idx, row in df.iterrows():
        poi_id = row['poi_id']
        lat = float(row['lat'])
        lng = float(row['lng'])
        poi_to_coords[poi_id] = (lat, lng)
    return poi_to_coords


# ======================
# 3. 独热编码PC(类别)信息
# ======================

def one_hot_encode_pc(poi_to_pc):
    """
    给所有 POI 的 PC 做独热编码。
    poi_to_pc: {poi_id: [pc1, pc2, ...]}
    返回:
      pc_list: 排序后的去重类别列表
      poi_to_pc_vec: {poi_id: np.array([...])} 每个POI对应的独热向量
    """
    # 1) 收集所有出现的类别
    all_pcs = set()
    for poi_id, pcs in poi_to_pc.items():
        for pc in pcs:
            all_pcs.add(pc)
    pc_list = sorted(list(all_pcs))  # 排序后保证可重复使用

    # 2) 构建 {pc: index}
    pc_to_index = {pc: i for i, pc in enumerate(pc_list)}

    # 3) 为每个POI构建独热向量
    poi_to_pc_vec = {}
    for poi_id, pcs in poi_to_pc.items():
        vec = np.zeros(len(pc_list), dtype=np.float32)
        for pc in pcs:
            idx = pc_to_index[pc]
            vec[idx] = 1.0
        poi_to_pc_vec[poi_id] = vec

    return pc_list, poi_to_pc_vec


# =====================
# 4. 定义自定义距离函数
# =====================

def custom_distance(poi1, poi2,
                    poi_to_coords,
                    poi_to_pc_vec,
                    alpha=1.0,
                    beta=1.0):
    """
    计算两个POI之间的综合距离：
    - 地理距离(欧氏)  +  (1 - 类别向量相似度)
    其中相似度这里以余弦相似度为例
    最终距离 = alpha * geo_dist + beta * (1 - cos_sim)
    """
    # 1) 地理欧氏距离
    lat1, lng1 = poi_to_coords[poi1]
    lat2, lng2 = poi_to_coords[poi2]
    geo_dist = np.sqrt((lat1 - lat2) ** 2 + (lng1 - lng2) ** 2)

    # 2) 类别向量余弦相似度
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
    """
    为给定的一批 POI 构造两两之间的距离矩阵 (N x N)，
    之后可直接传给 DBSCAN 的 metric='precomputed' 用。
    """
    n = len(pois)
    D = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            d = custom_distance(pois[i], pois[j], poi_to_coords, poi_to_pc_vec,
                                alpha=alpha, beta=beta)
            D[i, j] = d
            D[j, i] = d
    return D


# ==========================
# 5. 在每个Area下进行DBSCAN
# ==========================

def cluster_area_pois(area_to_pois, poi_to_coords, poi_to_pc_vec,
                      eps=0.01, min_samples=3,
                      alpha=1.0, beta=1.0):
    """
    对每个 Area 下的所有 POI 进行 DBSCAN 聚类。
    返回:
      normal_clusters: [(global_cluster_id, area_id, [poi_ids...]), ...]
      noise_points: [(area_id, [poi_ids...]), ...]
    说明：
      - normal_clusters 中包含所有非 -1 簇的信息，并在全局范围内从1开始编号
      - noise_points 中记录噪点 (label = -1)，记录其所在的 area_id 和 poi 列表
    """

    normal_clusters = []
    noise_points = []

    global_cluster_id = 1  # 全局簇编号从1开始

    # 依次处理每个 area
    for area_id, pois in area_to_pois.items():
        if len(pois) == 0:
            continue
        elif len(pois) == 1:
            # 只有一个POI，直接视作一个聚类簇
            normal_clusters.append((global_cluster_id, area_id, [pois[0]]))
            global_cluster_id += 1
            continue

        # 构造距离矩阵
        dist_matrix = create_distance_matrix(pois, poi_to_coords, poi_to_pc_vec, alpha, beta)

        # DBSCAN, 以“预计算距离矩阵”的方式
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = db.fit_predict(dist_matrix)

        # 收集结果
        cluster_map = {}
        for poi_idx, lbl in enumerate(labels):
            poi_id = pois[poi_idx]
            if lbl not in cluster_map:
                cluster_map[lbl] = []
            cluster_map[lbl].append(poi_id)

        # 处理每个聚类标签
        for lbl, poi_list in cluster_map.items():
            if lbl == -1:
                # 噪点
                noise_points.append((area_id, poi_list))
            else:
                # 正常簇
                normal_clusters.append((global_cluster_id, area_id, poi_list))
                global_cluster_id += 1

    return normal_clusters, noise_points


def save_cluster_results(normal_clusters, noise_points,
                         cluster_output_path, noise_output_path):
    """
    将聚类结果分别写入两个文件:
      1) 正常簇 (cluster_output_path):
         列: functional_zone_id, area_id, poi_ids
      2) 噪点 (noise_output_path):
         列: functional_zone_id, area_id, poi_ids
         其中 functional_zone_id 可统一为 -1
    """

    # 1) 写正常簇
    cluster_rows = []
    for (cluster_id, area_id, pois) in normal_clusters:
        cluster_rows.append((cluster_id, area_id, ",".join(pois)))
    df_cluster = pd.DataFrame(cluster_rows,
                              columns=["functional_zone_id", "area_id", "poi_ids"])
    df_cluster.to_csv(cluster_output_path, index=False, encoding='utf-8')

    # 2) 写噪点
    noise_rows = []
    for (area_id, pois) in noise_points:
        # 噪点ID 统一写成 -1
        noise_rows.append((-1, area_id, ",".join(pois)))
    df_noise = pd.DataFrame(noise_rows,
                            columns=["functional_zone_id", "area_id", "poi_ids"])
    df_noise.to_csv(noise_output_path, index=False, encoding='utf-8')


def main():
    # ================
    # 1) 路径参数
    # ================
    kg_path = "/home/qwan857/UUKG_wgj/UUKG-main/UrbanKG_data/UrbanKG/NYC/UrbanKG_NYC.txt"
    poi_csv_path = "/home/qwan857/UUKG_wgj/UUKG-main/UrbanKG_data/Processed_data/NYC/NYC_poi.csv"
    output_dir = "/home/qwan857/UUKG_wgj/UUKG-main/UrbanKG_data/Processed_data/NYC/"

    # ================
    # 2) 读取与预处理
    # ================
    print("1) 读取KG文件...")
    poi_to_area, area_to_pois, poi_to_pc = parse_kg_file(kg_path)

    print("2) 读取POI坐标...")
    poi_to_coords = load_poi_coordinates(poi_csv_path)

    print("3) 独热编码POI的PC信息...")
    pc_list, poi_to_pc_vec = one_hot_encode_pc(poi_to_pc)

    # ================================
    # 3) 聚类参数设置
    # ================================
    eps = 0.01       # DBSCAN eps参数
    min_samples = 3  # DBSCAN 最小样本数

    # ================================
    # 4) 循环测试不同的 alpha:beta 比例
    # ================================
    ratio_list = [0.0, 0.25, 0.5, 0.75, 1.0]  # alpha : beta 比例
    print("4) 测试不同的 (alpha : beta) 比例 ...")
    print("注意：这里 alpha + beta = 1，仅测试比例，且包含 alpha=0 / beta=0")

    for alpha in ratio_list:
        beta = 1 - alpha

        # 在每个 Area 下进行聚类
        normal_clusters, noise_points = cluster_area_pois(
            area_to_pois, poi_to_coords, poi_to_pc_vec,
            eps=eps, min_samples=min_samples,
            alpha=alpha, beta=beta
        )

        # 统计正常簇数量和噪点 POI 总数
        normal_count = len(normal_clusters)
        noise_count = sum(len(pois) for _, pois in noise_points)

        # 输出文件路径
        cluster_output_path = os.path.join(
            output_dir, f"cluster_result_alpha{alpha:.2f}_beta{beta:.2f}.csv"
        )
        noise_output_path = os.path.join(
            output_dir, f"noise_points_alpha{alpha:.2f}_beta{beta:.2f}.csv"
        )

        # 打印结果
        print(f"比例 alpha : beta = {alpha:.2f} : {beta:.2f} => "
              f"正常簇数: {normal_count}, 噪点POI总数: {noise_count}")

        # 保存聚类结果到文件
        save_cluster_results(normal_clusters, noise_points, cluster_output_path, noise_output_path)
        print(f"正常簇已输出到: {cluster_output_path}")
        print(f"噪点已输出到: {noise_output_path}")

    # ================================
    # 5) 测试 alpha=beta=1:1 时 DBSCAN 超参数变化
    # ================================
    print("\n5) 测试 alpha=beta=1:1 时 DBSCAN 超参数的影响 ...")
    for eps in [0.005, 0.01, 0.05]:  # 测试不同的eps值
        for min_samples in [3, 5, 10]:  # 测试不同的min_samples值
            # 在 alpha=1, beta=0 时聚类
            normal_clusters, noise_points = cluster_area_pois(
                area_to_pois, poi_to_coords, poi_to_pc_vec,
                eps=eps, min_samples=min_samples,
                alpha=1.0, beta=0.0
            )

            # 统计正常簇数量和噪点 POI 总数
            normal_count = len(normal_clusters)
            noise_count = sum(len(pois) for _, pois in noise_points)

            # 输出文件路径
            cluster_output_path = os.path.join(
                output_dir, f"cluster_result_eps{eps:.4f}_min_samples{min_samples}_alpha1_beta0.csv"
            )
            noise_output_path = os.path.join(
                output_dir, f"noise_points_eps{eps:.4f}_min_samples{min_samples}_alpha1_beta0.csv"
            )

            # 打印结果
            print(f"eps = {eps}, min_samples = {min_samples} => "
                  f"正常簇数: {normal_count}, 噪点POI总数: {noise_count}")

            # 保存聚类结果到文件
            save_cluster_results(normal_clusters, noise_points, cluster_output_path, noise_output_path)
            print(f"正常簇已输出到: {cluster_output_path}")
            print(f"噪点已输出到: {noise_output_path}")


if __name__ == "__main__":
    main()
