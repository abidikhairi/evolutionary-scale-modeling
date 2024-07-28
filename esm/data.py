from typing import Tuple
from esm.tokenization import EsmTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset


def _esm_train_test_split(ds: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
    ds = ds.train_test_split(test_size=0.2)
    train_data = ds['train']
    
    ds_chunk = ds['test'].train_test_split(test_size=0.5)
    valid_data = ds_chunk['train']
    test_data = ds_chunk['test']
    
    return train_data, valid_data, test_data


def prepare_data_for_pretraining(
    input_file: str,
    tokenizer: EsmTokenizer,
    batch_size: int = 4,
    num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = Dataset.from_csv(input_file)

    dataset = dataset.map(lambda examples: tokenizer(examples['Sequence'], padding=True, return_tensors='pt'), batch_size=1024, batched=True) \
        .remove_columns('Sequence')
    
    train_data, valid_data, test_data = _esm_train_test_split(dataset)
    
    train_data = train_data.select(range(100000)).with_format('torch')
    valid_data = valid_data.select(range(500)).with_format('torch')
    test_data = test_data.select(range(500)).with_format('torch')
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15, return_tensors='pt')    

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=data_collator, num_workers=num_workers)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader
