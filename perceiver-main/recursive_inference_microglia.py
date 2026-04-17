
import torch
import pickle
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from datasets import DataModule
from networks import Exceiver, GatedMLP, load_model
from tasks import RecursiveInference
from config_recursive_microglia import dataset_cfg, task_cfg, model_cfg, trainer_cfg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"GPU is available {torch.cuda.is_available()}")
torch.set_float32_matmul_precision('medium')
#torch.backends.cuda.sdp_kernel(True)
#torch.backends.cuda.enable_flash_sdp(True)
#torch.backends.cuda.enable_math_sdp(True)

def check_train_test_set(cfg):
    train = pickle.load(open(cfg["train_metadata_path"], "rb"))
    test = pickle.load(open(cfg["test_metadata_path"], "rb"))
    train_ids = set(train["obs"]["SubID"])
    test_ids = set(test["obs"]["SubID"])
    print(f"Number of subjects in train set: {len(train_ids)}, and test set: {len(test_ids)}")
    train_test_inter = train_ids.intersection(test_ids)
    print(f"Number of train users in test set: {len(train_test_inter)}")


def main(train_idx, test_idx, perturb_genes):

    # Set seed
    pl.seed_everything(45)

    # check_train_test_set(dataset_cfg)


    dataset_cfg["train_idx"] = train_idx
    dataset_cfg["test_idx"] = test_idx
    #dataset_cfg["RDA"] = False

    #dataset_cfg["cell_restrictions"] = {"cell_type": "K562", "perturbation": "control"}
    #dataset_cfg["cell_restrictions"] = {"cell_type": "RPE", "perturbation": "control"}
    task_cfg["perturb_genes"] = perturb_genes

    #dataset_cfg["cell_restrictions"] = {"class": cell_class, }

    # Set up data module
    dm = DataModule(**dataset_cfg)
    dm.setup("train")
    task_cfg["gene_names"] = dm.train_dataset.gene_names
    task_cfg["cell_idx"] = dm.train_dataset.cell_idx

    # Set up data module
    dm = DataModule(**dataset_cfg)

    # Transfer information from Dataset
    model_cfg["seq_len"] = dm.n_genes
    model_cfg["cell_properties"] = dm.cell_properties
    model_cfg["n_bins"] = dataset_cfg["n_bins"]
    #model_cfg["RDA"] = True
    task_cfg["cell_properties"] = dm.cell_properties

    # Create network
    # model = Exceiver(**model_cfg)
    model = GatedMLP(**model_cfg)

    if model_cfg["model_save_path"] is not None:
        model = load_model(model_cfg["model_save_path"], model)

    #for n, p in model.named_parameters():
    #    print(n, p.size())

    task = RecursiveInference(network=model, task_cfg=task_cfg)

    trainer = pl.Trainer(
        enable_checkpointing=False,
        accelerator='gpu',
        devices=trainer_cfg["n_devices"],
        max_epochs=1,
        gradient_clip_val=trainer_cfg["grad_clip_value"],
        accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
        precision=trainer_cfg["precision"],
        strategy=DDPStrategy(find_unused_parameters=True) if trainer_cfg["n_devices"] > 1 else "auto",
        val_check_interval=1,
        limit_val_batches=50,
    )

    trainer.fit(task, dm)


if __name__ == "__main__":

    meta = pickle.load(open(dataset_cfg["metadata_path"], "rb"))

    n_samples = len(meta["obs"]["barcode"])
    print(n_samples)

    idx = np.where((np.array(meta["obs"]["mg_subtype"]) == "Homeo_CECR2"))
    perturb_genes = ["TNFRSF1A", "IL1R1", "IFNGR1", "TREM2", "DPYD", "MITF", "GPNMB", "APOE", "IL6ST", "SPP1", "PTPRG", "SAMD4A", "HIF1A", "ACSL1", "CD163"]
    perturb_genes = ["IL15", "CX3CR1",  "FRMD4A", "OXR1", "ELMO1", "HSPA1A", "TNFRSF1A", "IL1R1", "IFNGR1", "TREM2", "DPYD", "MITF", "GPNMB",
                     "APOE", "IL6ST", "SPP1", "PTPRG", "SAMD4A", "HIF1A", "ACSL1", "CD163", "HSPA1A"]

    del meta

    main(idx, idx, perturb_genes)