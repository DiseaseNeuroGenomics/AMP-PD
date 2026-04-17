
"""List of cell properties to predict, with values indicating (possibly restricted) list of target outputs.
None indicates continuous vales"""

cell_properties = None

embedding_strategy = "continuous"
n_bins = 32

base_dir = "/home/masse/work/GRN"
loss = "ZINB"

dataset_cfg = {
    "data_path": "/home/masse/work/GRN/data/data_psychad_micro_v4.dat",
    "metadata_path": "/home/masse/work/GRN/data/metadata_psychad_micro_v4.pkl",
    #"data_path": "/home/masse/work/GRN/data/data_replogle_k562_v4.dat",
    #"metadata_path": "/home/masse/work/GRN/data/metadata_replogle_k562_v4.pkl",
    "cell_properties": cell_properties,
    "batch_size": 64,
    "num_workers": 14,
    "n_targets": None,
    "n_input": 2500,
    "remove_sex_chrom": False,
    "protein_coding_only": False,
    "cell_prop_same_ids": False,
    "embedding_strategy": embedding_strategy,
    "n_bins": n_bins,
    "RDA": True,
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
}

model_cfg = {
    "seq_dim": 512,
    "query_len": 128,
    "query_dim": 512,
    "n_layers": 36,
    "n_heads": 8,
    "dim_feedforward": 1024,
    "dropout": 0.0,
    "embedding_strategy": embedding_strategy,
    "linear_embedding": False,
    "n_bins": n_bins,
    "RDA": True,
    "second_layer_RDA": True,
    "loss": loss,
    "output_pi": True,
    #"model_save_path": f"{base_dir}/saved_models/version_73/checkpoints/epoch=1-step=207116.ckpt",
    #"model_save_path": f"/home/masse/work/GRN/saved_models/version_147/checkpoints/epoch=0-step=80000.ckpt" # good
    "model_save_path": f"/home/masse/work/GRN/saved_models/version_164/checkpoints/epoch=0-step=40000.ckpt" # also good, basd on ver 147

}

