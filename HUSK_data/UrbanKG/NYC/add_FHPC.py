def main():
    urbanKG_file = "UrbanKG_NYC_PLR_withFZ.txt"

    # 用来存放 POI -> set(功能区ID) / set(POI类型) / set(路段ID)
    poi_in_zone = {}
    poi_has_pc = {}
    poi_in_road = {}

    # 先读取全部行，仅作解析，不做改写
    with open(urbanKG_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # parts 应该是 [subject, predicate, object] 三段
            if len(parts) != 3:
                continue  # 跳过不符合格式的行

            subj, pred, obj = parts

            if pred == "PLF":
                # 形如: POI/xxx PLF FZ/yyy
                if subj.startswith("POI/") and obj.startswith("FZ/"):
                    poi_id = subj.split("POI/")[1]
                    zone_id = obj.split("FZ/")[1]
                    poi_in_zone.setdefault(poi_id, set()).add(zone_id)

            elif pred == "PHPC":
                # 形如: POI/xxx PHPC PC/yyy
                if subj.startswith("POI/") and obj.startswith("PC/"):
                    poi_id = subj.split("POI/")[1]
                    pc_type = obj.split("PC/")[1]
                    poi_has_pc.setdefault(poi_id, set()).add(pc_type)

            elif pred == "PLR":
                # 形如: POI/xxx PLR Road/yyy
                if subj.startswith("POI/") and obj.startswith("Road/"):
                    poi_id = subj.split("POI/")[1]
                    road_id = obj.split("Road/")[1]
                    poi_in_road.setdefault(poi_id, set()).add(road_id)
            else:
                # 其他关系不处理
                pass

    # -------------------------------------------------------------------------
    # 基于上面解析的结果，生成新的关系 FHPC、RHPC
    # -------------------------------------------------------------------------
    new_relations = set()

    # 为了遍历所有可能的 POI，做一个并集
    all_poi_ids = set(poi_in_zone.keys()) | set(poi_has_pc.keys()) | set(poi_in_road.keys())

    for poi_id in all_poi_ids:
        # 如果该 POI 同时有 zone 和 pc，就生成 FZ/X FHPC PC/Y
        if poi_id in poi_in_zone and poi_id in poi_has_pc:
            zones = poi_in_zone[poi_id]
            pc_types = poi_has_pc[poi_id]
            for z in zones:
                for pc in pc_types:
                    new_relations.add(f"FZ/{z} FHPC PC/{pc}")

        # 如果该 POI 同时有 road 和 pc，就生成 Road/X RHPC PC/Y
        # if poi_id in poi_in_road and poi_id in poi_has_pc:
        #     roads = poi_in_road[poi_id]
        #     pc_types = poi_has_pc[poi_id]
        #     for r in roads:
        #         for pc in pc_types:
        #             new_relations.add(f"Road/{r} RHPC PC/{pc}")

    # new_relations 中的每一条都是类似于 "FZ/1 FHPC PC/residential_area" 或 "Road/2 RHPC PC/hotel" 等。

    # -------------------------------------------------------------------------
    # 将新的关系在文件末尾追加写入，确保不更改原文件中的顺序或内容
    # 同时，由于原文件不含 FHPC、RHPC，故不会与原文件重复
    # 为防止新关系自身重复，我们在上面用 set() 已经去重。
    # 这里可以直接逐条写入
    # -------------------------------------------------------------------------
    with open(urbanKG_file, 'a', encoding='utf-8') as fw:
        for rel in new_relations:
            fw.write(rel + "\n")

    print("处理完成！已将新的 FHPC、RHPC 关系追加至文件末尾。")

if __name__ == "__main__":
    main()
