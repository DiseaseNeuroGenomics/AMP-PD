from typing import List
import numpy as np
import torch
import pickle
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
from data import DataModule
from config import dataset_cfg, model_cfg, task_cfg, trainer_cfg
from task import StandardLoss
from networks import FeedforwardNetwork, load_model
from pytorch_lightning.loggers import CSVLogger
import gc

torch.set_float32_matmul_precision('medium')


def main(train_idx: List[int], test_idx: List[int], cell_class: str, brain_region: str):

    # Set seed
    pl.seed_everything(42)

    for k, v in task_cfg.items():
        print(f"{k}: {v}")

    dataset_cfg["train_idx"] = train_idx
    dataset_cfg["test_idx"] = test_idx

    dataset_cfg["cell_restrictions"] = {"derived_class2_Dec2024": cell_class, }

    max_epochs = 30
    # Set up data module
    dm = DataModule(**dataset_cfg)
    dm.setup("train")

    task_cfg["cell_properties"] = model_cfg["cell_properties"] = dm.cell_properties
    model_cfg["n_input"] = dm.train_dataset.n_genes
    model_cfg["batch_properties"] = dm.batch_properties
    network = FeedforwardNetwork(**model_cfg)

    task = StandardLoss(network=network, **task_cfg)
    for n, p in network.named_parameters():
        print(n, p.size())

    gc.collect()

    logger = CSVLogger("logs", name=dataset_cfg["log_path"])

    trainer = pl.Trainer(
        enable_checkpointing=False,
        accelerator='gpu',
        devices=trainer_cfg["n_devices"],
        max_epochs=max_epochs,
        gradient_clip_val=trainer_cfg["grad_clip_value"],
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
        precision=trainer_cfg["precision"],
        strategy=DDPStrategy(find_unused_parameters=True) if trainer_cfg["n_devices"] > 1 else "auto",
        logger=logger,
    )
    trainer.fit(task, dm)




if __name__ == "__main__":


    splits = pickle.load(open(dataset_cfg["train_test_splits_path"], "rb"))

    for cell_class in ["Myeloid"]:


        #for brain_region in ['DMNX', 'GPI', 'PMC', 'PFC', 'PVC']:
        for split_num in range(20):


            print(f"Cell class: {cell_class}, split number: {split_num}")

            gc.collect()
            train_idx = splits[split_num]["train_idx"]
            test_idx = splits[split_num]["test_idx"]
            #idx = train_idx + test_idx
            main(train_idx, test_idx, cell_class, brain_region=None)