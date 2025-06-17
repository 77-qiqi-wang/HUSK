#  Knowledge-enhanced Urban Spatial Task 

## Directory Structure

- |-raw_data/:    Store preprocessed atomic files.
- |-libcity/:    Project code root directory.
  - |-config/:   The ConfigParser class is defined here, which supports command line and config file to modify our default parameters. 
  - |-data/:   The Dataset class is stored in a subfolder of this folder according to different tasks. 
  - |-model/:    Model classes are stored in subfolders of this folder according to the tasks they belong to. 
  - |-evaluator/:    A task corresponds to a dedicated evaluator.
  - |-executor/:    Each task provides a standard Executor, and the model can also have its own exclusive Executor.
  - |-pipeline/:     Store user-oriented pipeline functions, which are responsible for running through the entire framework process.
  - |cache/:    Store the cache. Specifically, data preprocessing results, model training results, and evaluation results will be cached.
  - |-tmp/:    Store temporary files such as checkpoint generated during training.
  - |-utils/:    Store some general utility functions.
- |-log/:    Store log information during training.

The directory structure could help readers better understand our code framework.


## Datasets

The dataset is stored in the **./raw_data** folder.

The following types of atomic files are defined:

| filename    | content                                  | example                                  |
| ----------- | ---------------------------------------- | ---------------------------------------- |
| xxx.geo     | Store geographic entity attribute information. | geo_id, type, coordinates                |
| xxx.rel     | Store the relationship information between entities, such as areas. | rel_id, type, origin_id, destination_id  |
| xxx.dyna    | Store traffic condition information.     | dyna_id, type, time, entity_id, location_id |
| config.json | Used to supplement the description of the above table information. |                                          |

## Quick to Usage

The script `run.py` used for training and evaluating a single model is provided in the root directory of the framework, and a series of command line parameters are provided to allow users to adjust the running parameter configuration.

When run the `run.py`, you must specify the following three parameters, namely `task`, `dataset` and `model`. For example:

```bash
python run.py --task traffic_state_pred --model STPGCN --dataset CHICrime20210112
```

This script will run the STPGCN model on the CHICrime20210112 dataset for traffic state prediction task under the default configuration.

**How to fuse UrbanKG embedding?**

To fuse UrbanKG embedding, we directly concatenate the embedding with USTP feature for input. You can mannualy modify it in the **./data/dataset/traffic_state_dataset.py**.

## References

Some parts of the code were forked from the original LibCity [1] implementation and adapted from the UUKG project [2], available at: https://github.com/LibCity/Bigscity-LibCity-Docs-zh_CN and https://github.com/usail-hkust/UUKG/

[1] Wang, Jingyuan, et al. "Libcity: An open library for traffic prediction." *Proceedings of the 29th International Conference on Advances in Geographic Information Systems*. 2021.

[2] Ning, Yu, et al. "UUKG: Unified Urban Knowledge Graph Dataset for Urban Spatiotemporal Prediction." *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 2023.
