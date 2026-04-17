
import torch
import pickle
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.strategies.ddp import DDPStrategy
from datasets import DataModule
from networks import Exceiver, load_model
from tasks import MSELoss
from config import test_dataset_cfg, task_cfg, model_cfg, trainer_cfg, test_cfg
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"GPU is available {torch.cuda.is_available()}")
torch.set_float32_matmul_precision('medium')

# Set seed
pl.seed_everything(2299)

def cell_sweep():

    classes = ['Astro', 'EN', 'Endo', 'IN', 'Immune', 'Mural', 'OPC', 'Oligo']
    subclasses = [
        'Astro', 'EN_L2_3_IT', 'EN_L3_5_IT_1', 'EN_L3_5_IT_2', 'EN_L3_5_IT_3',
        'EN_L5_6_NP', 'EN_L5_ET', 'EN_L6B', 'EN_L6_CT', 'EN_L6_IT_1', 'EN_L6_IT_2',
        'EN_NF', 'Endo', 'IN_ADARB2', 'IN_LAMP5_LHX6', 'IN_LAMP5_RELN', 'IN_PVALB',
        'IN_PVALB_CHC', 'IN_SST', 'IN_VIP', 'Immune', 'Micro', 'OPC', 'Oligo', 'PC',
        'PVM', 'SMC', 'VLMC',
    ]

    for c in classes[:4]:

        cell_prop = ("class", [c])
        print(f"Cell prop {c}")


        test_dataset_cfg["subset_data_info"] = cell_prop
        # Set up data module
        dm = DataModule(**test_dataset_cfg)

        dm.setup(None)

        # Transfer information from Dataset
        model_cfg["seq_len"] = dm.train_dataset.n_genes
        model_cfg["cell_properties"] = dm.train_dataset.cell_properties
        task_cfg["cell_properties"] = dm.train_dataset.cell_properties
        task_cfg["balance_classes"] = False
        task_cfg["learning_rate"] = 0.0
        task_cfg["weight_decay"] = 0.0
        model_cfg["rank_order"] = test_dataset_cfg["rank_order"]

        # Create network
        model = Exceiver(**model_cfg)
        model.eval()

        #model = torch.load(test_cfg["ckpt_path"])
        model = load_model(test_cfg["ckpt_path"], model)

        task = MSELoss(
            network=model,
            task_cfg=task_cfg,
        )


        trainer = pl.Trainer(
            enable_checkpointing=False,
            accelerator='gpu',
            devices=trainer_cfg["n_devices"],
            max_epochs=1,
            gradient_clip_val=trainer_cfg["grad_clip_value"],
            accumulate_grad_batches=trainer_cfg["accumulate_grad_batches"],
            precision=trainer_cfg["precision"],
            strategy=DDPStrategy(find_unused_parameters=True) if trainer_cfg["n_devices"] > 1 else "auto",
            limit_train_batches=1,
            limit_val_batches=1000,
        )

        trainer.fit(task, dm)
        analyze_results(trainer)


def softmax(x):
    y = np.exp(x)
    return y / np.sum(y)

def convert(x, k, dim):
    print(k, x["pred_"+k].shape)

    x0 = np.reshape(x[k], (-1,))
    idx = np.where((x0 >= -99))[0]
    x0 = x0[idx]

    if x["pred_"+k].ndim > 2:
        x1 = np.reshape(x["pred_"+k], (-1, x["pred_"+k].shape[-1]))
        x1 = x1[idx, :]
        un0 = np.arange(dim)
    else:
        x1 = np.reshape(x["pred_"+k], (-1,))
        x1 = x1[idx]
        un0 = np.unique(x0)

    s = np.zeros((dim, dim))
    for y0, y1 in zip(x0, x1):
        #p = softmax(y1)
        #s[y0, :] += p
        # i0 = np.where(y0 == un0)[0]
        if x["pred_" + k].ndim > 2:
            i1 = np.where(np.argmax(y1) == un0)[0]
            j = y0
        else:
            i1 = np.argmin(np.abs(un0 - y1))
            j = np.argmin(np.abs(un0 - y0))

        s[j, i1] += 1.0
    #s /= (1e-9 + np.sum(s, axis=1, keepdims=True))
    return s, len(x0)

def analyze_results(save_fn, trainer):
    v = trainer.logger.version
    fn = f"{trainer.log_dir}/lightning_logs/version_{v}/test_results.pkl"
    x = pickle.load(open(fn, "rb"))

    targets = {"AD": 2, "Dementia": 2, "ApoE_gt": 5, "BRAAK_AD": 7, "CERAD": 4, "class": 8}

    f, axs = plt.subplots(2, 3, figsize=(15, 8))

    for n, k in enumerate(targets.keys()):
        s, n_samples = convert(x, k, targets[k])
        acc = np.sum(np.diag(s)) / np.sum(s)
        ax = axs[n // 3, n % 3]
        pcm = ax.imshow(s, aspect="auto")
        f.colorbar(pcm, ax=ax)
        ax.set_ylabel("Target", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_title(f"{k} \n Accuracy={acc:1.3f} N={n_samples}", fontsize=15)

    plt.suptitle(f"{cell_prop[0]} = {cell_prop[1][0]}", fontsize=20)
    plt.tight_layout()
    save_fn = f"{cell_prop[0]}_{cell_prop[1][0]}_v116_mssm_test_set"
    fig_fn = f"{trainer.log_dir}/figures/{save_fn}.png"
    plt.savefig(fig_fn)


if __name__ == "__main__":

    cell_sweep()

