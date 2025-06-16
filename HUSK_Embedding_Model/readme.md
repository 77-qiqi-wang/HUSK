# POI-level Tasks and UrbanKG Embedding 

Before conducting POI-level tasks, please ensure that the corresponding training, validation, and test sets for the selected task have been generated and copied to the **`./data`** directory. You can also modify the file paths directly in `run.py` if needed.

## Usage

To train and evaluate a UrbanKG embedding model for the link prediction task, use the run.py script:

```bash
usage: run.py [-h] [--dataset {NYC, CHI}]
              [--model {TransE, RotH, ...}]
              [--regularizer {N3,N2}] [--reg REG]
              [--optimizer {Adagrad,Adam,SGD,SparseAdam,RSGD,RAdam}]
              [--max_epochs MAX_EPOCHS] [--patience PATIENCE] [--valid VALID]
              [--rank RANK] [--batch_size BATCH_SIZE]
              [--neg_sample_size NEG_SAMPLE_SIZE] [--dropout DROPOUT]
              [--init_size INIT_SIZE] [--learning_rate LEARNING_RATE]
              [--gamma GAMMA] [--bias {constant,learn,none}]
              [--dtype {single,double}] [--double_neg] [--debug] [--multi_c]

```
## How to get the embedding

We establish an index mapping between entities and their learned embeddings, which is stored in **`./data/entity_idx_embedding.csv`**. To obtain the learned UrbanKG embeddings, run **`id2id.py`** followed by **`get_embedding.py`**. The resulting embeddings will be saved in the **`./embedding`** folder in `.npy` format.

## References

Some of the code was forked from the original AttH implementation which can be found at: [https://github.com/HazyResearch/KGEmb](https://github.com/HazyResearch/KGEmb)

