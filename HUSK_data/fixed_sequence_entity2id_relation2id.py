# -*- coding: utf-8 -*-

def generate_entity_ids(input_file, entity_output):
    """
    从输入的KG文件中提取实体并进行编号，写入到指定文件中。
    """
    entity_dict = {}
    entity_counter = 0

    # 读取输入的KG文件
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过空行
            if not line:
                continue

            # 每行格式：  实体1 关系 实体2
            head, rel, tail = line.split()

            # 处理实体
            if head not in entity_dict:
                entity_dict[head] = entity_counter
                entity_counter += 1
            if tail not in entity_dict:
                entity_dict[tail] = entity_counter
                entity_counter += 1

    # 写出 entity2id_NYC.txt
    with open(entity_output, "w", encoding="utf-8") as f:
        for ent, eid in entity_dict.items():
            f.write(f"{ent} {eid}\n")

    # 打印统计信息
    print(f"实体文件 {entity_output} 生成完毕，共 {len(entity_dict)} 个实体。")


if __name__ == "__main__":
    # input_file = "UrbanKG/NYC/UrbanKG_NYC_PLR_withFZ.txt"
    # entity_output = "UrbanKG/NYC/entity2id_NYC.txt" 
    input_file = "UrbanKG/CHI/UrbanKG_CHI_withFZ.txt"
    entity_output = "UrbanKG/CHI/entity2id_CHI.txt" # 记得改文件夹

    generate_entity_ids(input_file, entity_output)

# _withFZ _withFZ_FHPC _PLR _PLR_RHPC _PLR_withFZ _PLR_withFZ_HPC _PLR_withFZ_RLF