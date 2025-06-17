import csv
import os
from shapely.geometry import Point, shape
from shapely import wkt
from tqdm import tqdm


def read_pois(poi_filename):
    """
    从CSV中读取POI信息，返回列表 [(poi_id, latitude, longitude), ...]
    """
    pois = []
    with open(poi_filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            poi_id = row['poi_id']
            lat = float(row['lat'])
            lon = float(row['lng'])
            pois.append((poi_id, lat, lon))
    return pois


def read_roads(road_filename):
    """
    从CSV中读取Road信息，假设geometry是WKT格式，返回列表 [(link_id, geometry), ...]
    geometry会被转成Shapely对象
    """
    roads = []
    with open(road_filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            link_id = row['link_id']
            # 如果你的geometry是WKT，就用wkt.loads解析
            # 如果你只是存了道路的起止点坐标，需要自行构造LineString或Point等
            geom = wkt.loads(row['geometry'])
            roads.append((link_id, geom))
    return roads


def find_closest_road(poi_point, roads):
    """
    给定一个POI对应的Shapely Point对象，
    以及所有道路信息 (link_id, Shapely几何) 列表，
    计算距离最近道路的 link_id，并返回 (link_id, distance)。
    """
    min_dist = float('inf')
    closest_road_id = None
    for link_id, road_geom in roads:
        dist = poi_point.distance(road_geom)
        if dist < min_dist:
            min_dist = dist
            closest_road_id = link_id
    return closest_road_id, min_dist


def main():
    # 文件路径请自行替换成实际路径
    poi_file = "/home/gwan700/UUKG_wgj/UUKG-main/UrbanKG_data/Processed_data/CHI/CHI_poi.csv"
    road_file = "/home/gwan700/UUKG_wgj/UUKG-main/UrbanKG_data/Processed_data/CHI/CHI_road.csv"
    urbanKG_file = "/home/gwan700/UUKG_wgj/UUKG-main/UrbanKG_data/UrbanKG/CHI/UrbanKG_CHI_PLR.txt"

    # 读取数据
    pois = read_pois(poi_file)
    roads = read_roads(road_file)

    # 把UrbanKG_NYC.txt原有内容先读进来
    # （如果要直接在文件尾部追加，也可以在后面选择 "append" 模式写文件）
    with open(urbanKG_file, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    # 准备一个新的列表来存所有输出行
    new_lines = [line.rstrip('\n') for line in original_lines]

    # 遍历每个POI，找到与其最近的Road
    for (poi_id, lat, lon) in tqdm(pois, desc="Processing POIs"):
        poi_point = Point(lon, lat)  # Point(x, y) = Point(经度, 纬度)
        closest_road_id, dist = find_closest_road(poi_point, roads)
        # 格式：POI/<POI编号> PLR Road/<道路link_id>
        new_line = f"POI/{poi_id} PLR Road/{closest_road_id}"
        new_lines.append(new_line)

    # 将更新后的内容写回文件，也可以写到一个新文件里
    # 下面示例直接覆盖写回
    with open(urbanKG_file, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + "\n")

    print("处理完成，PLR关系已追加到 UrbanKG_CHI_PLR.txt 文件中。")


if __name__ == "__main__":
    main()
