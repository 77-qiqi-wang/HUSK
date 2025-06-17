import os
import pandas as pd


def main():
    # ========== 0. 路径设定 ==========
    base_dir = "/home/gwan700/UUKG_wgj/UUKG-main/UrbanKG_data/UrbanKG/NYC"
    processed_dir = "/home/gwan700/UUKG_wgj/UUKG-main/UrbanKG_data/Processed_data/NYC"

    # 原始文件
    kg_file = os.path.join(base_dir, "UrbanKG_NYC_PLR.txt")
    entity2id_file = os.path.join(base_dir, "entity2id_NYC.txt")

    #### 测试不同参数文件 ####
    cluster_result_file = os.path.join(processed_dir, "cluster_result_alpha0.50_beta0.50.csv")

    # 输出文件
    kg_withfz_file = os.path.join(base_dir, "UrbanKG_NYC_PLR_withFZ.txt")
    entity2id_withfz_file = os.path.join(base_dir, "entity2id_NYC.txt")

    # ========== 1. 读取 cluster_result.csv ==========
    # 假设 cluster_result.csv 的列为 ["functional_zone_id", "area_id", "poi_ids"]
    df_cluster = pd.read_csv(cluster_result_file, dtype=str)
    # 注意：functional_zone_id、area_id、poi_ids 都可能是字符串形式
    # poi_ids 是用逗号分隔的一串 poi_id

    # ========== 2. 读取并复制原 UrbanKG_NYC.txt 的内容 ==========
    # 稍后我们在这个文件末尾追加新的三元组
    with open(kg_file, 'r', encoding='utf-8') as f:
        kg_lines = f.readlines()

    # ========== 3. 读取 entity2id_NYC.txt，并找出当前最大编号 ==========
    entity_list = []
    max_id = -1

    with open(entity2id_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 每一行形如: Road/88151 140601
            parts = line.split()
            if len(parts) != 2:
                continue
            entity_name, entity_id = parts
            entity_id = int(entity_id)
            entity_list.append((entity_name, entity_id))
            if entity_id > max_id:
                max_id = entity_id

    # 此时 max_id 就是源文件中最后一个编号 (用户示例: 140601)
    # 接下来为所有出现的FZ实体依次从 max_id+1 开始编号

    # ========== 4. 为所有出现的 FZ/编号 分配新的 entity_id ==========
    # 先收集所有 FZ/编号
    # 注意：在 cluster_result.csv 里，"functional_zone_id" 可能是全局编号，也可能是别的形式
    # 这里假设就是 1, 2, 3...  或者 area_0, area_1, ...

    unique_fz_ids = df_cluster["functional_zone_id"].unique().tolist()  # 可能是字符串

    # 生成 { "1": 140602, "2": 140603, ... } 的映射
    fz_to_eid = {}
    next_id = max_id + 1

    for fz_id in unique_fz_ids:
        # 构造 FZ/<fz_id> 的名字
        fz_name = f"FZ/{fz_id}"
        fz_to_eid[fz_id] = next_id
        entity_list.append((fz_name, next_id))
        next_id += 1

    # ========== 5. 在 UrbanKG_NYC 原文末尾，追加三元组 ==========
    new_kg_lines = []

    for idx, row in df_cluster.iterrows():
        fz_id_str = row["functional_zone_id"]  # 例如 "1"
        area_id_str = row["area_id"]  # 例如 "224"
        poi_ids_str = row["poi_ids"]  # 例如 "497,498,499"

        # 追加一行: FZ/<fz_id> FLA Area/<area_id>
        new_kg_lines.append(f"FZ/{fz_id_str} FLA Area/{area_id_str}\n")

        # 对每个 poi_id, 追加: POI/<poi_id> PLF FZ/<fz_id>
        poi_list = poi_ids_str.split(",")
        for poi_id in poi_list:
            poi_id = poi_id.strip()
            new_kg_lines.append(f"POI/{poi_id} PLF FZ/{fz_id_str}\n")

    # ========== 6. 写回新的 KG 文件 ==========
    with open(kg_withfz_file, 'w', encoding='utf-8') as f:
        # 先写原始行
        for line in kg_lines:
            f.write(line)
        # 再追加新行
        for line in new_kg_lines:
            f.write(line)

    # ========== 7. 写回新的 entity2id_NYC_withFZ 文件 ==========
    with open(entity2id_withfz_file, 'w', encoding='utf-8') as f:
        for name, eid in entity_list:
            f.write(f"{name} {eid}\n")

    print("===== 已完成 FZ 节点和三元组的追加，输出文件如下：=====")
    print(f"新知识图谱文件: {kg_withfz_file}")
    print(f"新实体编号文件: {entity2id_withfz_file}")


if __name__ == "__main__":
    main()
