### Guidance on data usage and processing

All raw data is stored in the **'./Meta_data'** folder. You can process these files using the `preprocess_meta_data_nyc.py` or `preprocess_meta_data_chi.py` scripts to perform necessary cleaning and alignment. Once processed, the cleaned data will be saved into the **'./Processed_data'** folder.

Next, you can generate the basic urban knowledge graphs by running the `construct_UrbanKG_XXX.py` script. After that, under the `UrbanKG` directory, both the **'./CHI'** and **'./NYC'** folders contain corresponding scripts for knowledge graph enhancement. You may sequentially run `fz.py`, `add_fz.py`, `add_PLR.py`, and `add_FHPC.py` to build the enhanced HUSK.

In addition, the `fixed_sequence_entity2id_relation2id.py` script assigns unique IDs to entities and relations in the HUSK. Then, `KG_split.py` is used to generate the training, validation, and test sets required for downstream POI-level tasks.

The file information in each directory is as follows:

```
./Meta_data    Raw data set: administrative division data, POI and road network data
./Processed_data   Preprocessed data and clustering results of Functional Zones
./UrbanKG    Various versions of HUSK, containing multiple types of entities and diverse relations
```

The following types of atomic files are defined:

| filename                              | content                                 | example                       |
| ------------------------------------- | --------------------------------------- | ----------------------------- |
| entity2id_XXX.txt                     | entity_name, entity_id                  | FZ/1256 237542                |
| relation2id_XXX.txt                   | relation_name, relation_id              | PLR 13                        |
| train                                 | entity_id, relation_id, entity_id       | 187868	12	236285        |
| valid                                 | entity_id, relation_id, entity_id       | 19586	10	236262         |
| test                                  | entity_id, relation_id, entity_id       | 137618	5	140317         |
| triplet.txt                           | entity_id, relation_id, entity_id       | 48034   12 168303             |
| UrbanKG_XXX.txt                       | entity_name, relation_name, entity_name | POI/442 PLA Area/13           |
| UrbanKG_XXX_PLR_withFZ_FHPC.txt       | entity_name, relation_name, entity_name | FZ/707 FHPC PC/parking_area   |
| cluster_result_alpha0.50_beta0.50.csv | functional_zone_id, area_id, poi_ids    | 7,211,"418980,621514,1074344" |

### The Preparation for POI-level Tasks
To perform POI-level tasks, the corresponding training, validation, and test sets are required. The `KG_split.py` script provides configurable options for switching between different tasks—simply modify the relevant sections of the code and run the script.
