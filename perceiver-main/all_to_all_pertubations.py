
import torch
import pickle
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from datasets import DataModule
from networks import Exceiver, GatedMLP, load_model
from tasks import AllToAllPerutbation
from config_tf import dataset_cfg, task_cfg, model_cfg, trainer_cfg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"GPU is available {torch.cuda.is_available()}")
torch.set_float32_matmul_precision('medium')



def main(train_idx, test_idx):

    # Set seed
    pl.seed_everything(45)

    for i in range(0, 4):


        if i == 0:
            dataset_cfg["cell_restrictions"] = {"cell_subtype": ["Micro"], "BRAAK": [0, 1, 2], "CERAD": [1],
                                                "Dementia": [0, ], "source": ["RUSH", "MSSM"]}
        elif i == 1:
            dataset_cfg["cell_restrictions"] = {"cell_subtype": ["Micro"], "BRAAK": [4, 5, 6], "CERAD": [3, 4],
                                                "Dementia": [0.5, 1], "source": ["RUSH", "MSSM"]}

        if i == 2:
            dataset_cfg["cell_restrictions"] = {"cell_type": ["IN"], "BRAAK": [0, 1, 2], "CERAD": [1],
                                                "Dementia": [0, ], "source": ["RUSH", "MSSM"]}
        elif i == 3:
            dataset_cfg["cell_restrictions"] = {"cell_type": ["IN"], "BRAAK": [4, 5, 6], "CERAD": [3, 4],
                                                "Dementia": [0.5, 1], "source": ["RUSH", "MSSM"]}

        #dataset_cfg["cell_restrictions"] = None
        dataset_cfg["train_idx"] = train_idx
        dataset_cfg["test_idx"] = test_idx
        model_cfg["RDA"] = dataset_cfg["RDA"]

        # Set up data module
        dm = DataModule(**dataset_cfg)
        dm.setup("train")
        task_cfg["gene_names"] = dm.train_dataset.gene_names
        task_cfg["n_bins"] = dataset_cfg["n_bins"]


        # Set up data module
        dm = DataModule(**dataset_cfg)

        # Transfer information from Dataset
        model_cfg["seq_len"] = dm.n_genes
        model_cfg["cell_properties"] = dm.cell_properties
        task_cfg["cell_properties"] = dm.cell_properties
        model_cfg["n_bins"] = dataset_cfg["n_bins"]
        task_cfg["learning_rate"] = 0.0

        # Create network
        # model = Exceiver(**model_cfg)
        model = GatedMLP(**model_cfg)

        model = load_model(model_cfg["model_save_path"], model)
        model.eval()

        task = AllToAllPerutbation(network=model, task_cfg=task_cfg)

        trainer = pl.Trainer(
            enable_checkpointing=False,
            accelerator='gpu',
            devices=trainer_cfg["n_devices"],
            max_epochs=60,
            gradient_clip_val=trainer_cfg["grad_clip_value"],
            accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
            precision=trainer_cfg["precision"],
            strategy=DDPStrategy(find_unused_parameters=True) if trainer_cfg["n_devices"] > 1 else "auto",
            limit_train_batches=1,
            limit_val_batches=100,
        )

        trainer.fit(task, dm)


if __name__ == "__main__":

    meta = pickle.load(open(dataset_cfg["metadata_path"], "rb"))
    train_idx = np.arange(len(meta["obs"]["experiment"]))
    test_idx = np.arange(len(meta["obs"]["experiment"]))
    del meta


    main(train_idx, test_idx)
