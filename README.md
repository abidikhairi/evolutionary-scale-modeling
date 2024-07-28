# Evolutionary Scale Modeling: Unofficial implementation

## ESM 1
Esm1 is a transformer encoder model trained with masked language modeling on a large scale protein sequence database (uniref 50).

This is a user-friendly implementation, that is meant for research purposes (i.e, not intended for deployment).

## TODO
- [ ] Contanct prediction map
- [ ] Finetuning on different tasks
- [ ] Upload pretrained models on github (needs LFS)
- [ ] Release training datasets on huggingface
- [ ] Notebook to help users reproduce results
- [ ] Add `setup.py`

## Repository Files:
- `esm`: Directory that contains main source code
    - `data.py`: Data processing/transformation pipelines
    - `constants.py`: Amino acids and special tokens
    - `rotary_embedding.py`: Rotary embedding implementation
    - `schedulers.py`: Learning rate schedulers implementation (See ESM paper)
    - `tokenization.py`: Protein tokenizer implementation (using _transformers_ python package)
    - `multihead_attention.py`: Multihead attention implementation (not optimized)
    - `modules.py`: Transformers building blocks
    - `mask_utils.py`: Utilities to create attention masks.
    - `model`: Models implementations goes here
    - `model/esm1.py`: ESM1 implementation
    - `model/modeling_outputs.py`: Python classes to hold structured data (dictionaries, layer outputs, _etc_) 
    - `training`: Training utilities
    -  `training/lightning.py`: Pytorch lightning training code
    - `training/model_config.py`: Pre defined hyperparameters for different model sizes
-  `pretrain_lm.py`: code for pre-training the model on protein sequence  
