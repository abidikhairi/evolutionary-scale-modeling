from pytorch_lightning import Trainer, callbacks, loggers
import torch
from esm import TransformerTrainer, EsmTokenizer
from esm.training import ModelNames, ModelsConfig, load_model_config
from esm.data import prepare_data_for_pretraining


def main():
    tokenizer = EsmTokenizer()
    
    model_name = ModelNames.esm1
    model_config = load_model_config(ModelsConfig.esm1_sm, tokenizer)
    
    model = TransformerTrainer(
        model_name=model_name,
        **model_config
    )
    
    train_loader, valid_loader, test_loader = prepare_data_for_pretraining(
        input_file='data/path/to/sequences.csv', # NOTE: contains one column (protein sequence) with no header
        tokenizer=tokenizer,
        batch_size=32,
        num_workers=4
    )

    trainer = Trainer(
        accelerator='gpu',
        max_epochs=1,
        enable_model_summary=True,
        log_every_n_steps=50,
        logger=[
            loggers.CSVLogger('data/logs', name='esm1_debug', flush_logs_every_n_steps=50),
            loggers.WandbLogger(name='esm1_debug_uniprot', save_dir='data', project='LitReview')
        ],
        
        callbacks=[
            callbacks.LearningRateMonitor(logging_interval='step')
        ]
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    trainer.test(model, test_loader)
    
    trainer.save_checkpoint(filepath='data/models/esm1_debug.ckpt', weights_only=True)
    
    torch.save(model.model.state_dict(), 'data/models/esm1_debug.pt')
    

if __name__ == '__main__':
    main()
