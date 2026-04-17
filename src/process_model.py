from typing import List, Optional
import pickle
import copy
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import scipy.stats as stats
from sklearn.decomposition import PCA
import scipy.optimize as optimize


def explained_var(x_pred, x_real):
    return 1 - np.nanvar(x_pred - x_real) / np.nanvar(x_real)

def classification_score(x_pred, x_real):
    s0 = np.sum((x_real == 0) * (x_pred < 0.5)) / np.sum(x_real == 0)
    s1 = np.sum((x_real == 1) * (x_pred >= 0.5)) / np.sum(x_real == 1)
    return (s0 + s1) / 2

def softmax(x, axis=-1):
    y = np.exp(x)
    return y / np.sum(y, axis=axis, keepdims=True)


class ModelResults:

    def __init__(
        self,
        data_fn: str = "/home/masse/work/data/pd/data.dat",
        meta_fn: str = "/home/masse/work/data/pd/metadata.pkl",
        obs_list: List[str] = ["path_braak_lb"],
        n_bins: int = 100,
        normalize_gene_counts: bool = False,
        log_gene_counts: bool = False,
        add_gene_scores: bool = False,
        add_all_gene_vals: bool = False,
    ):

        self.data_fn = data_fn
        self.meta = pickle.load(open(meta_fn, "rb"))

        self.obs_list = obs_list

        self.gene_idx = self.meta["var"]["protein_coding"]
        self.n_genes = len(self.meta["var"]["gene_name"])
        self.gene_names = self.meta["var"]["gene_name"][self.gene_idx]
        self.latent_dim = np.sum(self.gene_idx)
        self.n_bins = n_bins

        self.normalize_gene_counts = normalize_gene_counts
        self.log_gene_counts = log_gene_counts
        self.add_gene_scores = add_gene_scores
        self.add_all_gene_vals = add_all_gene_vals

        self.extra_obs = []

        self.obs_from_metadata = [
            "SubID", "sex", "age", "Brain_bank", "path_braak_lb",
            "brain_reg_w_gt", 'derived_class2_Dec2024', 'derived_subclass2_Dec2024',
            'derived_subtype2_Dec2024', "RIN", "n_counts", "n_genes", "barcode"
        ]

        self.donor_stats = [
            "Brain_bank", "age", "path_braak_lb",
        ]
        for p in obs_list:
            self.donor_stats.append(p)
            self.donor_stats.append(f"pred_{p}")

        self.donor_stats = np.unique(self.donor_stats)

        self.brain_regions = ['DMNX', 'GPI', 'PFC', 'PMC', 'PVC']


        gene_names = self.meta["var"]["gene_name"]
        self.gene_mask = np.ones((len(gene_names),))


    def create_data(
        self,
        model_fns,
        subclass=None,
        cell_class=None,
        brain_region=None,
        model_average=False,
    ):

        if model_average:
            adata = self.create_base_anndata_repeats(
                model_fns,
                subclass=subclass,
                cell_class=cell_class,
                brain_region=brain_region,
            )
        else:
            adata = self.create_base_anndata(
                model_fns, subclass=subclass,
                cell_class=cell_class,
                brain_region=brain_region,
            )

        # save directories where data came from
        adata.uns["model_fns"] = model_fns

        # print out which cell subclasses are present to confirm we're processing right data
        subclasses = adata.obs.derived_subclass2_Dec2024.unique()
        print(f"Subclasses present: {subclasses}")
        adata = self.add_unstructured_data(adata)
        adata = self.add_donor_stats(adata, add_gene_scores=self.add_gene_scores)
        if self.add_gene_scores:
            adata = self.add_gene_scores_real(adata)

        return adata

    def normalize_data(self, x):

        x = np.float32(x)
        if self.normalize_gene_counts:
            x = 10_000 * x / np.sum(x)
        if self.log_gene_counts:
            x = np.log1p(x)

        return x

    @staticmethod
    def subset_data_by_subclass(adata, subclass):

        return adata[adata.obs.subclass == subclass]

    @staticmethod
    def weight_probs(prob, k):


        if k == "path_braak_lb_condensed_v2":
            w = np.array([0, 1.5, 4.5])
        elif k == "path_braak_lb_condensed":
            w = np.array([0, 1.5, 3.5, 5.5])
        else:
            w = np.array([0, 1])

        if prob.ndim == 2:
            w = w[None, :]
        elif prob.ndim == 3:
            w = w[None, None, :]
            prob = softmax(prob, -1)

        return np.sum(prob * w, axis=-1)

    def add_obs_from_metadata(self, adata):

        for k in self.obs_from_metadata:
            x = []
            for n in adata.obs["cell_idx"]:
                x.append(self.meta["obs"][k][n])
            adata.obs[k] = x

        return adata

    def create_single_anndata(self, z):

        n = z[self.obs_list[0]].shape[0]
        n_latent = self.latent_dim if self.add_all_gene_vals else 1

        latent = np.zeros((n, n_latent), dtype=np.uint8)
        mask = np.reshape(z["cell_mask"], (-1, z["cell_mask"].shape[-1]))

        a = ad.AnnData(latent)
        a.obs["split_num"] = z["split_num"]

        if self.add_all_gene_vals:
            a.var["gene_name"] = self.meta["var"]["gene_name"]
        for m, k in enumerate(self.obs_list):
            try:
                a.obs[k] = z[k]
            except:
                print(f"{k} not found. Skipping.")
                continue

            idx = np.where(mask[:, m] == 0)[0]
            a.obs[k][idx] = np.nan

            if z[f"pred_{k}"].ndim == 2:
                probs = z[f"pred_{k}"]
                a.obs[f"pred_{k}"] = self.weight_probs(probs, k)
                a.uns[f"full_pred_{k}"] = probs

            else:
                a.obs[f"pred_{k}"] = z[f"pred_{k}"]


        a.obs["cell_idx"] = z["cell_idx"]
        for k in self.extra_obs:
            a.obs[k] = np.array(self.meta["obs"][k])[z["cell_idx"]]
            if not isinstance(a.obs[k][0], str):
                idx = np.where(a.obs[k] < -99)[0]
                a.obs[k][idx] = np.nan

        a = self.add_obs_from_metadata(a)

        return a

    def concat_arrays(self, fns):

        for n, fn in enumerate(fns):
            z = pickle.load(open(fn, "rb"))
            if n == 0:
                x = copy.deepcopy(z)
                x["split_num"] = n * np.ones(len(x[self.obs_list[0]]))
            else:
                z["split_num"] = n * np.ones(len(z[self.obs_list[0]]))
                for k in self.obs_list:
                    x[k] = np.concatenate((x[k], z[k]), axis=0)
                    x[f"pred_{k}"] = np.concatenate((x[f"pred_{k}"], z[f"pred_{k}"]), axis=0)

                for k in self.extra_obs + ["cell_idx"]:
                    x[k] = np.concatenate((x[k], z[k]), axis=0)

                x["split_num"] = np.concatenate((x["split_num"], z["split_num"]), axis=0)

        return x

    def create_base_anndata_repeats(self, model_fns, cell_class=None, subclass=None, brain_region=None):

        x = []
        n_models = len(model_fns)
        for fns in model_fns:
            x0 = self.concat_arrays(fns)
            idx = np.argsort(x0["cell_idx"])

            for k in self.obs_list:
                x0[k] = x0[k][idx]
                x0[f"pred_{k}"] = x0[f"pred_{k}"][idx]

            for k in self.extra_obs + ["cell_idx"]:
                x0[k] = x0[k][idx]

            x.append(x0)

        x_new = copy.deepcopy(x0)

        for k in self.obs_list:
            x_new[k] = 0
            x_new[f"pred_{k}"] = 0
            for n in range(n_models):
                x_new[k] += x[n][k] / n_models
                x_new[f"pred_{k}"] += x[n][f"pred_{k}"] / n_models


        if "donor_px_r" in x0.keys():
            for k in x0["donor_px_r"].keys():
                x_new["donor_px_r"][k] = 0
                for n in range(n_models):
                    x_new["donor_px_r"][k] += x[n]["donor_px_r"][k] / n_models

        adata = self.create_single_anndata(x_new)

        idx = None
        if subclass is not None:
            if isinstance(subclass, str):
                idx = adata.obs.subclass == subclass
            elif isinstance(subclass, list):
                idx = adata.obs.subclass.isin(subclass)
        elif cell_class is not None:
            idx = adata.obs["class"] == cell_class
        elif brain_region is not None:
            idx = adata.obs["brain_reg_w_gt"] == brain_region

        if idx is not None:
            adata = adata[idx]
            for k in adata.uns.keys():
                if f"pred_" in k:
                    adata.uns[k] = adata.uns[k][idx]

        return adata

    def create_base_anndata(self, model_fns, cell_class=None, subclass=None, brain_region=None):

        for n, fn in enumerate(model_fns):
            z = pickle.load(open(fn, "rb"))
            a = self.create_single_anndata(z)

            a.obs["split_num"] = n
            if n == 0:
                adata = a.copy()
            else:
                uns = adata.uns
                adata = ad.concat((adata, a), axis=0)
                adata.uns = uns
                if "donor_px_r" in a.uns.keys():
                    for k in a.uns["donor_px_r"].keys():
                        adata.uns["donor_px_r"][k] = a.uns["donor_px_r"][k]

        if subclass is not None:
            if isinstance(subclass, str):
                adata = adata[adata.obs.subclass == subclass]
            elif isinstance(subclass, list):
                adata = adata[adata.obs.subclass.isin(subclass)]
        if cell_class is not None:
            adata = adata[adata.obs["class"] == cell_class]
        if brain_region is not None:
            adata = adata[adata.obs["brain_reg_w_gt"] == brain_region]

        return adata

    def get_cell_index(self, model_fns):

        cell_idx = []
        cell_class = []
        cell_subclass = []

        for n, fn in enumerate(model_fns):
            z = pickle.load(open(fn, "rb"))
            cell_idx += z["cell_idx"].tolist()
            cell_class += self.meta["obs"]["class"][z["cell_idx"]].tolist()
            cell_subclass += self.meta["obs"]["subclass"][z["cell_idx"]].tolist()

        # assuming cell class is the same
        index = {cell_class[0]: cell_idx}
        for sc in np.unique(cell_subclass):
            idx = np.where(np.array(cell_subclass) == sc)[0]
            index[sc] = np.array(cell_idx)[idx]

        return index, np.unique(cell_class), np.unique(cell_subclass)


    def add_gene_scores_real(self, adata):

        """not used in manuscript, but can be helpful for visualization of changes in gene expression"""

        percentiles = {}
        for k in self.obs_list:
            percentiles[k] = np.percentile(adata.obs[f"pred_{k}"], np.arange(0, 100, 100/self.n_bins))
        adata.uns["percentiles"] = percentiles

        conds = [None]
        score_names = [""]

        all_gene_vals = []

        for cond, score_name in zip(conds, score_names):

            gene_vals_cond = []

            if cond is None:
                a = adata.copy()
            else:
                for k, v in cond.items():
                    a = adata[adata.obs[k] == v]

            counts = {k: 1e-6 + np.zeros(self.n_bins, dtype=np.float32) for k in self.obs_list}
            scores = {k: np.zeros((self.n_genes, self.n_bins), dtype=np.float32) for k in self.obs_list}

            for n, i in enumerate(a.obs["cell_idx"]):

                data = np.memmap(
                    self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                ).astype(np.uint8).astype(np.float32)

                data = self.normalize_data(data)

                if self.add_all_gene_vals and cond is None:
                    all_gene_vals.append(data[self.gene_idx])

                data = data[self.gene_idx]
                # gene_vals_cond.append(data)

                for j, k in enumerate(self.obs_list):
                    pred_obs = a.obs[f"pred_{k}"][n]
                    if np.isnan(pred_obs) or pred_obs < -99:
                        continue
                    else:
                        # bin_idx = np.argmin((v - percentiles[k]) ** 2)
                        bin_idx = np.where(pred_obs >= adata.uns["percentiles"][k])[0][-1]

                        if bin_idx == 0:
                            counts[k][bin_idx] += 2
                            scores[k][:, bin_idx] += 2 * data
                            counts[k][bin_idx+1] += 1
                            scores[k][:, bin_idx+1] += data
                        elif bin_idx == self.n_bins - 1:
                            counts[k][bin_idx] += 2
                            scores[k][:, bin_idx] += 2 * data
                            counts[k][bin_idx-1] += 1
                            scores[k][:, bin_idx-1] += data
                        else:
                            counts[k][bin_idx] += 1
                            scores[k][:, bin_idx] += data
                            counts[k][bin_idx - 1] += 1
                            scores[k][:, bin_idx - 1] += data
                            counts[k][bin_idx + 1] += 1
                            scores[k][:, bin_idx + 1] += data


            for k in scores.keys():
                print("COUNTS", k, counts[k][:5])
                scores[k] /= counts[k][None, :]

            adata.uns["traj" + score_name] = scores
            adata.uns["gene_counts" + score_name] = counts

            if self.add_all_gene_vals:
                adata.X = np.stack(all_gene_vals, axis=0)

        return adata

    def add_unstructured_data(self, adata):

        for k in self.obs_list:
            adata.uns[k] = []

        adata.uns["donors"] = []
        for subid in np.unique(adata.obs["SubID"]):
            adata.uns["donors"].append(subid)
            idx = np.where(np.array(adata.obs["SubID"].values) == subid)[0][0]
            for k in self.obs_list:
                adata.uns[k].append(adata.obs[k].values[idx])

        return adata

    def add_donor_stats_region(self, adata, add_gene_scores=True):

        """ not used in manuscript"""

        n_donors = len(adata.uns["donors"])
        n_regions = len(self.brain_regions)

        if add_gene_scores:
            adata.uns[f"donor_gene_means"] = np.zeros((n_donors, n_regions, self.n_genes), dtype=np.float32)

        adata.uns[f"donor_cell_count"] = np.zeros((n_donors, n_regions), dtype=np.float32)
        for k in self.donor_stats:
            adata.uns[f"donor_{k}"] = np.zeros((n_donors, n_regions), dtype=np.float32)

        for m, subid in enumerate(adata.uns["donors"]):
            a0 = adata[adata.obs["SubID"] == subid]

            for m0, br in enumerate(self.brain_regions):
                a = a0[a0.obs["brain_reg_w_gt"] == br]
                if len(a) == 0:
                    continue

                count = 1e-6

                for n, i in enumerate(a.obs["cell_idx"]):
                    if add_gene_scores:
                        data = np.memmap(
                            self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                        ).astype(np.float32)
                        data = self.normalize_data(data)
                        adata.uns[f"donor_gene_means"][m, m0, :] += data

                    count += 1

                    for k in self.donor_stats:
                        if "pred_" in k:
                            if np.isnan(a.obs[f"{k}"][n]):
                                print(k, subid, a.obs[f"{k}"][n])
                                continue
                            else:
                                adata.uns[f"donor_{k}"][m, m0] += a.obs[f"{k}"][n]
                        elif "Sex" in k:
                            adata.uns[f"donor_{k}"][m, m0] = float(a.obs[f"{k}"][n]=="Male")
                        elif "Brain_bank" in k:
                            adata.uns[f"donor_{k}"][m, m0] = float(a.obs[f"{k}"][n]=="MSSM")
                        else:
                            adata.uns[f"donor_{k}"][m, m0] = a.obs[f"{k}"][n]

                adata.uns[f"donor_cell_count"][m, m0] = count
                if add_gene_scores:
                    adata.uns[f"donor_gene_means"][m, m0, :] /= count

                for k in self.donor_stats:
                    if "pred_" in k:
                        adata.uns[f"donor_{k}"][m, m0] /= count

        return adata

    def add_donor_stats(self, adata, add_gene_scores=True):

        n_donors = len(adata.uns["donors"])
        if add_gene_scores:
            adata.uns[f"donor_gene_means"] = np.zeros((n_donors, self.n_genes), dtype=np.float32)

        adata.uns[f"donor_cell_count"] = np.zeros(n_donors, dtype=np.float32)

        for k in self.donor_stats:
            adata.uns[f"donor_{k}"] = np.zeros((n_donors), dtype=np.float32)

        for m, subid in enumerate(adata.uns["donors"]):
            a = adata[adata.obs["SubID"] == subid]
            count = 1e-6

            for n, i in enumerate(a.obs["cell_idx"]):
                if add_gene_scores:
                    data = np.memmap(
                        self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                    ).astype(np.float32)
                    data = self.normalize_data(data)
                    adata.uns[f"donor_gene_means"][m, :] += data

                count += 1

                for k in self.donor_stats:
                    if "pred_" in k:
                        if np.isnan(a.obs[f"{k}"][n]):
                            print(k, subid, a.obs[f"{k}"][n])
                            continue
                        else:
                            adata.uns[f"donor_{k}"][m] += a.obs[f"{k}"][n]
                    elif "Sex" in k:
                        adata.uns[f"donor_{k}"][m] = float(a.obs[f"{k}"][n]=="Male")
                    elif "Brain_bank" in k:
                        adata.uns[f"donor_{k}"][m] = float(a.obs[f"{k}"][n]=="MSSM")
                    else:
                        adata.uns[f"donor_{k}"][m] = a.obs[f"{k}"][n]

            adata.uns[f"donor_cell_count"][m] = count
            if add_gene_scores:
                adata.uns[f"donor_gene_means"][m, :] /= count

            for k in self.donor_stats:
                if "pred_" in k:
                    adata.uns[f"donor_{k}"][m] /= count

        return adata


    @staticmethod
    def corr_2d(x, y):
        x -= np.mean(x, axis=0, keepdims=True)
        y -= np.mean(y, axis=0, keepdims=True)
        x /= np.linalg.norm(x, axis=0, keepdims=True)
        y /= np.linalg.norm(y, axis=0, keepdims=True)
        print(x.shape, y.shape)
        return x.T @ y


    def add_donor_gene_pca(self, adata):

        n_components = 20
        pca = PCA(n_components=n_components)

        gene_idx = self.important_genes["gene_idx"]
        gene_names = self.important_genes["gene_names"]
        adata.uns["gene_corr_names"] = gene_names

        n_donors = len(adata.uns["donors"])
        n_genes = len(gene_idx)
        adata.uns[f"donor_pca_explained_var"] = np.zeros((n_donors, n_components), dtype=np.float32)
        adata.uns[f"donor_pca_explained_var_ratio"] = np.zeros((n_donors, n_components), dtype=np.float32)
        adata.uns[f"donor_pca_components"] = np.zeros((n_donors, n_components, n_genes), dtype=np.float32)
        adata.uns[f"donor_pca_var"] = np.zeros((n_donors, n_genes), dtype=np.float32)
        adata.uns[f"donor_pca_noise_var"] = np.zeros((n_donors), dtype=np.float32)
        adata.uns[f"donor_pca_counts"] = np.zeros(n_donors, dtype=np.float32)

        for m, subid in enumerate(adata.uns["donors"]):
            print(m, len(adata.uns["donors"]), subid)
            a = adata[adata.obs["SubID"] == subid]
            gene_counts = []

            for n, i in enumerate(a.obs["cell_idx"]):
                data = np.memmap(
                    self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
                ).astype(np.float32)

                data = self.normalize_data(data)
                gene_counts.append(data[gene_idx])

            gene_counts = np.stack(gene_counts, axis=0)
            #gene_counts = np.log1p(gene_counts)
            gene_counts -= np.mean(gene_counts, axis=0, keepdims=True)
            gene_counts /= (0.1 + np.std(gene_counts, axis=0, keepdims=True))
            adata.uns[f"donor_pca_counts"][m] = gene_counts.shape[0]

            if gene_counts.shape[0] < 20:
                continue

            pca.fit(gene_counts)
            adata.uns[f"donor_pca_explained_var"][m, :] = pca.explained_variance_
            adata.uns[f"donor_pca_explained_var_ratio"][m, :] = pca.explained_variance_ratio_
            adata.uns[f"donor_pca_components"][m, :, :] = pca.components_
            #adata.uns[f"donor_pca_cov"][m, :, :] = pca.get_covariance()
            adata.uns[f"donor_pca_var"][m, :] = np.var(gene_counts, axis=0)
            adata.uns[f"donor_pca_noise_var"][m] = pca.noise_variance_


        return adata




class GeneSignature:

    def __init__(
            self,
            gene_signatures,
            data_fn: str = "/home/masse/work/data/mssm_rush/data.dat",
            meta_fn: str = "/home/masse/work/data/mssm_rush/metadata_slim.pkl",
    ):
        self.gene_signatures = gene_signatures
        meta = pickle.load(open(meta_fn, "rb"))
        self.gene_names = meta["var"]["gene_name"]
        self.n_genes = len(self.gene_names)
        del meta
        self.data_fn = data_fn

        genes = []
        for v in self.gene_signatures.values():
            genes += v
        self.genes = np.unique(genes)
        self.gene_idx = []
        for g in self.genes:
            j = np.where(np.array(self.gene_names) == g)[0][0]
            self.gene_idx.append(j)
        self.gene_idx = np.array(self.gene_idx)

    def create_adata(self, adata):

        n_genes_signature = len(self.gene_idx)
        n = adata.shape[0]
        adata_new = ad.AnnData(np.zeros((n, n_genes_signature), dtype=np.float32))
        adata_new.var["gene_name"] = list(self.genes)
        adata_new.var.index = adata_new.var["gene_name"]
        adata_new.obs = adata.obs.copy()
        adata_new.uns = adata.uns.copy()
        del adata

        for n, i in enumerate(adata_new.obs["cell_idx"]):
            data = np.memmap(
                self.data_fn, dtype='uint8', mode='r', shape=(self.n_genes,), offset=i * self.n_genes,
            ).astype(np.float32)
            adata_new.X[n, :] = data[self.gene_idx]

        return adata_new



def curve(x, a0, a1):
    return a0 + a1 * x

def fit_single(x, y):

    n_time, n_genes = y.shape
    slopes = np.zeros((n_genes), dtype=np.float32)
    resid = np.zeros((n_time, n_genes), dtype=np.float32)
    for n in range(n_genes):
        curve_coefs, _ = optimize.curve_fit(curve, x, y[:, n])
        slopes[n] = curve_coefs[1]
        y_hat = curve(x, *curve_coefs)
        resid[:, n] = y[:, n] - y_hat

    return slopes, resid, curve_coefs, y_hat


bad_path_words = [
    "female",
    "bone",
    "retina",
    "ureteric",
    "ear",
    "skin",
    "hair",
    "cardiac",
    "metanephros",
    "outflow",
    "sound",
    "chondrocy",
    "eye",
    "vocalization",
    "social",
    "aorta",
    "pancreas",
    "digestive",
    "cochlea",
    "optic",
    "megakaryocyte",
    "embryo",
    "ossifi",
    "anterior/posterior",
    "animal",
    "cartilage",
    "cocaine",
    "sperm",
    "blastocyst",
    "fat ",
    "mammary",
    "substantia nigra",
    "mesenchymal",
    "estrous",
    "hindbrain",
    "forebrain",
    "brain",
    "locomotory",
    "acrosome",
    "ethanol",
    "nicotine",
    "cadmium",
    "ovarian",
    "melanocyte",
    "lead",
    "thyroid",
    "dexamethasone",
    "bacterium",
    "motor",
    "lung",
    "oocyte",
    "liver",
    "odontogenesis",
    "epidermis",
    "endodermal",
    "pulmonary",
    "decidualization",
    "response to heat",
    "pituitary",
    "tissue homeo",
    "keratinocyte",
    "keratinization",
    "osteoblast",
    "epidermal",
    "sensory organ",
    "layer formation",
    "endocardial",
    "organism development",
    "embryo",
    #"killing cells",
    "protozoan",
    "smell",
    "tumor cell",
    "hemopoiesis",
    "sensory perception",
    "genitalia",
    "urine",
    "xenobiotic",
    "muscle",
]


def check_path_term(path, bad_path_words):
    for p in bad_path_words:
        if p in path:
            return True
    return False
