
"""List of cell properties to predict, with values indicating (possibly restricted) list of target outputs.
None indicates continuous vales"""

cell_properties = None

embedding_strategy = "continuous"
n_bins = 32

base_dir = "/home/masse/work/GRN"
loss = "ZINB"

dataset_cfg = {
    "data_path": "/home/masse/work/GRN/data/data_perturb_small_v4.dat",
    "metadata_path": "/home/masse/work/GRN/data/metadata_perturb_small_v4.pkl",
    #"data_path": "/home/masse/work/GRN/data/data_replogle_k562_v4.dat",
    #"metadata_path": "/home/masse/work/GRN/data/metadata_replogle_k562_v4.pkl",
    "cell_properties": cell_properties,
    "batch_size": 32,
    "num_workers": 14,
    "n_targets": None,
    "n_input": 3_500,
    "remove_sex_chrom": False,
    "protein_coding_only": False,
    "cell_prop_same_ids": False,
    "embedding_strategy": embedding_strategy,
    "n_bins": n_bins,
    "RDA": False,
}


trainer_cfg = {
    "accumulate_grad_batches": 1,
    "precision": "bf16-mixed",
    "grad_clip_value": 1.0,
    "n_devices": 1,
}

task_cfg = {
    "learning_rate": 0e-4,
    "weight_decay": 0.00,
    "warmup_steps": 25_000,
    "decay_steps": 25_000,
    "decay": 0.999986,
    "balance_classes": True,
    "gene_weight": 1.0,
    "cell_prop_weight": 0.0,
    "loss": loss,
    "min_count_gene_expression": 1.0,
    "perturb_val": -1.0, # in counts
    "perturb_knockdown": True,
}

model_cfg = {
    "seq_dim": 512,
    "query_len": 128,
    "query_dim": 512,
    "n_layers": 36,
    "n_heads": 4,
    "dim_feedforward": 1024,
    "dropout": 0.0,
    "embedding_strategy": embedding_strategy,
    "linear_embedding": True,
    "n_bins": n_bins,
    "RDA": False,
    "second_layer_RDA": False,
    "loss": loss,
    "output_pi": True,
    "gene_val_emb_input_dim": 128,
    #"model_save_path": f"{base_dir}/saved_models/version_73/checkpoints/epoch=1-step=207116.ckpt",
    #"model_save_path": f"/home/masse/work/GRN/saved_models/version_147/checkpoints/epoch=0-step=80000.ckpt" # good

    #"model_save_path": f"/home/masse/work/GRN/saved_models/version_171/checkpoints/epoch=0-step=30000.ckpt"
    #"model_save_path": f"/home/masse/work/GRN/saved_models/version_241/checkpoints/epoch=1-step=366409.ckpt"
    #"model_save_path": f"/home/masse/work/GRN/saved_models/version_183/checkpoints/epoch=1-step=228671.ckpt"
    #"model_save_path": f"/home/masse/work/GRN/saved_models/version_194/checkpoints/epoch=0-step=35000.ckpt"
    #"model_save_path": f"/home/masse/work/GRN/saved_models/version_243/checkpoints/epoch=0-step=150000.ckpt"
    "model_save_path": f"/home/masse/work/GRN/saved_models/version_241/checkpoints/epoch=1-step=366409.ckpt"
}

