
cell_properties = {
    #"path_braak_lb_condensed_v2": {"discrete": True, "values": [0,1,2], "stop_grad": False},
    "path_braak_lb_condensed": {"discrete": True, "values": [0,1,2,3], "stop_grad": False},
    "path_braak_lb": {"discrete": False, "values": [0,1,2,3,4,5,6], "stop_grad": True},
}

batch_properties = None

dataset_cfg = {
    "data_path": "../data/PD_myeloid/data.dat",
    "metadata_path": "../data/PD_myeloid/metadata.pkl",
    "train_test_splits_path": "../data/PD_myeloid/train_test_20splits.pkl",
    "log_path": "AMPPD_PD", # location where the model logs (including inference) are stored
    "cell_properties": cell_properties,
    "batch_size": 256,
    "num_workers": 14,
    "batch_properties": batch_properties,
    "remove_sex_chrom": True,
    "protein_coding_only": True,
    "top_k_genes": 8_000,
    "group_balancing": "pd_condensed", #"pd_condensed_brain_region", #"pd_condensed_brain_region",
}

model_cfg = {
    "n_layers": 3,
    "n_hidden": 2048,
    "dropout_rate": 0.5,
    "input_dropout_rate": 0.0,
    "cell_decoder_hidden_layer": False,
    "layer_norm": True,
    "log_first": True,
}

task_cfg = {
    "learning_rate": 2e-2,
    "warmup_steps": 2000.0,
    "weight_decay": 0.05,
    "l1_lambda": 0.00,
    "balance_classes": False,
    "batch_properties": batch_properties,
    "save_gene_vals": False,
    "focal_loss": False,
    "focal_loss_gamma": 2.0,
    "adam_betas": [0.0, 0.0],
    "adam_eps": 1,

}

trainer_cfg = {
    "n_devices": 1,
    "grad_clip_value": 1.0,
    "accumulate_grad_batches": 1.0,
    "precision": "32-true",
}