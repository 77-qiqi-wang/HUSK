# -*- coding: utf-8 -*-

############################
# 1. 导入所需的库
############################
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, MultiPolygon, Polygon
import folium

############################
# 2. 读取并处理数据
############################

# (1) 读取 CHI_area.csv
#    - 假设文件中包含 area_id 等字段，重点是有 geometry 列（WKT 格式）
df_area = pd.read_csv('CHI_area.csv')
# 将WKT格式转换成Shapely几何类型
df_area['geometry'] = df_area['geometry'].apply(wkt.loads)
# 转成GeoDataFrame
gdf_area = gpd.GeoDataFrame(df_area, geometry='geometry', crs="EPSG:4326")

# (2) 读取 CHI_poi.csv
#    - 假设包含 poi_id、lat、lng 等列
df_poi = pd.read_csv('CHI_poi.csv')
# 为POI创建几何字段：Point(lng, lat)
df_poi['geometry'] = df_poi.apply(lambda x: Point(x['lng'], x['lat']), axis=1)
gdf_poi = gpd.GeoDataFrame(df_poi, geometry='geometry', crs="EPSG:4326")

# (3) 读取 cluster_result.csv
#    - 假设包含 functional_zone_id, area_id, poi_ids 等列
df_cluster = pd.read_csv('cluster_result_alpha0.50_beta0.50.csv')
# poi_ids 里是逗号分隔的POI ID，需要拆分
df_cluster['poi_ids'] = df_cluster['poi_ids'].apply(lambda x: x.strip('"'))  # 去除多余的引号
df_cluster['poi_ids_list'] = df_cluster['poi_ids'].apply(lambda x: x.split(','))

# 使用 explode 将每个功能区-POI对应关系展开
df_exploded = df_cluster.explode('poi_ids_list')
df_exploded = df_exploded.rename(columns={'poi_ids_list': 'poi_id'})
# 转成整型（确保 poi_id 类型正确）
df_exploded['poi_id'] = df_exploded['poi_id'].astype(int)

# 将功能区信息与POI信息合并
df_merged = df_exploded.merge(df_poi, on='poi_id', how='left')
gdf_merged = gpd.GeoDataFrame(df_merged, geometry='geometry', crs="EPSG:4326")

############################
# 3. 创建Folium地图
############################
# 定位到芝加哥市附近，经纬度可依据实际情况做修改
m = folium.Map(location=[41.8781, -87.6298], zoom_start=12)

############################
# 4. 绘制区域多边形
############################
# 针对每一个区域，如果是Polygon，直接取exterior；
# 如果是MultiPolygon，需要遍历内部每个Polygon。
for _, area in gdf_area.iterrows():
    geom = area['geometry']
    if geom.geom_type == 'MultiPolygon':
        # 如果是MultiPolygon，遍历每个Polygon
        for polygon in geom.geoms:
            # 绘制该Polygon的外轮廓
            coords = list(polygon.exterior.coords)
            folium.Polygon(
                locations=[(y, x) for (x, y) in coords],  # Folium的坐标是(纬度, 经度)顺序
                color='blue',
                weight=1.5,
                fill=True,
                fill_color='blue',
                fill_opacity=0.2
            ).add_to(m)
            # 若有内洞，则遍历interiors
            for interior in polygon.interiors:
                interior_coords = list(interior.coords)
                folium.Polygon(
                    locations=[(y, x) for (x, y) in interior_coords],
                    color='white',
                    weight=1.5,
                    fill=True,
                    fill_color='white',
                    fill_opacity=1.0
                ).add_to(m)
    elif geom.geom_type == 'Polygon':
        # 单一多边形，获取外轮廓
        coords = list(geom.exterior.coords)
        folium.Polygon(
            locations=[(y, x) for (x, y) in coords],
            color='blue',
            weight=1.5,
            fill=True,
            fill_color='blue',
            fill_opacity=0.2
        ).add_to(m)
        # 若有内洞
        for interior in geom.interiors:
            interior_coords = list(interior.coords)
            folium.Polygon(
                locations=[(y, x) for (x, y) in interior_coords],
                color='white',
                weight=1.5,
                fill=True,
                fill_color='white',
                fill_opacity=1.0
            ).add_to(m)
    else:
        # 针对其它几何类型，可根据实际需求进行处理
        pass

############################
# 5. 绘制功能区的POI点
############################
# 不同功能区ID可以用不同颜色
colors = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'pink', 'gray', 'brown']

for fz_id, group in gdf_merged.groupby('functional_zone_id'):
    # 若功能区ID大于colors数量，可取模循环
    c = colors[fz_id % len(colors)]
    for _, poi in group.iterrows():
        lat, lng = poi['lat'], poi['lng']
        folium.CircleMarker(
            location=[lat, lng],
            radius=5,
            color=c,
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

############################
# 6. 保存地图并查看
############################
m.save('functional_zones_with_pois_map.html')

print("地图绘制完成，请打开 functional_zones_with_pois_map.html 查看。")
