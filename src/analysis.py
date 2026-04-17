import os, sys
import copy
import pickle
import json
import pandas as pd
import scanpy as sc
import numpy as np

import path_enrichment_utils

import scipy.stats as stats
import scipy.optimize as optimize
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams['pdf.fonttype'] = 42


def curve(x, a0, a1):
    """Simple linear function. Used in fit_single"""
    return a0 + a1 * x


def fit_single(x, y):

    """Wrapper to fit linear function across multiple genes"""
    n_time, n_genes = y.shape
    time = np.arange(0, n_time)
    slopes = np.zeros((n_genes), dtype=np.float32)
    ex_var_gene = np.zeros((n_genes), dtype=np.float32)
    resid = np.zeros((n_time, n_genes), dtype=np.float32)
    for n in range(n_genes):
        curve_coefs, _ = optimize.curve_fit(curve, x, y[:, n])
        slopes[n] = curve_coefs[1]
        #y_hat = curve(x, *curve_coefs)
        #resid[:, n] = y[:, n] - y_hat
        #ex_var_gene[n] = np.mean(resid[:, n]**2) / np.var(y[:, n])

    #err = np.sum(np.mean(resid ** 2, axis=0))
    #var = np.sum(np.var(y, axis=0))
    #ex_var = 1 - err / var

    return slopes


def filter_vals_interp(x, y, sigma, x_interp=None, max_range=10_000, n_time_pts=1000):
    # x and y need to be sorted according to x

    if x_interp is None:
        x_interp = np.linspace(np.min(x), np.max(x), n_time_pts)

    N = y.shape[1]
    T = len(x_interp)

    x_filtered = np.zeros(T)
    y_filtered = np.zeros((T, N))
    w = np.zeros_like(x)[:, None]

    for t in range(len(x_interp)):
        d = (x_interp[t] - x) ** 2
        w[:, 0] = np.exp(-d / (2 * sigma ** 2))

        t_max = np.argmax(w[:, 0])
        t0 = np.maximum(0, t_max - max_range)
        t1 = np.minimum(len(x) - 1, t_max + max_range)
        w[:t0, 0] = 0.0
        w[t1:, 0] = 0.0
        idx = w[:, 0] > 1e-3
        w[idx, 0] /= np.sum(w[idx, 0])

        x_filtered[t] = np.sum(w[idx, 0] * x[idx])
        y_filtered[t, :] = w[idx, :].T @ y[idx, :]

    return x_filtered, y_filtered


def zenith_wrapper(
        data,
        gene_names,
        save_fn = f"../zenith/PD_myeloid_zenith_input.csv",
):

    time_pts = [[n, n + 50] for n in range(0, 950, 20)]
    braak_mid_pts = [np.mean(data["braak_filtered"][t]) for t in time_pts]

    slopes = calculate_slopes(time_pts, data["braak_filtered"], data["genes_filtered_full"])
    slope_dict = {f"Braak{n}": slopes[n, :] for n in range(len(time_pts))}

    genes = copy.deepcopy(list(gene_names))
    for n, g in enumerate(genes):
        if n == 0:
            continue
        if g in genes[:n]:
            genes[n] = g + "-1"
    output_zenith(slope_dict, save_fn, genes)

    return braak_mid_pts


def output_zenith(slope_dict, save_fn, gene_names):

    data = {"genes": []}
    for k in slope_dict.keys():
        data[k] = []

    for n, g in enumerate(gene_names):

        data["genes"].append(g)
        for k, v in slope_dict.items():
            data[k].append(v[n])

    df = pd.DataFrame(data=data)
    df.to_csv(save_fn)


def calculate_slopes(time_pts, pred_braak_traj, genes_braak_traj):
    n_time, n_genes = genes_braak_traj.shape

    m = len(time_pts)
    slopes = np.zeros((m, n_genes))

    for i, break_point in enumerate(time_pts):
        t0 = break_point[0]
        t1 = break_point[1]
        slopes[i, :] = fit_single(pred_braak_traj[t0:t1], genes_braak_traj[t0:t1, :])

    return slopes

def violin_plot_data(braak, pred_braak):
    idx = np.argsort(braak)
    braak = braak[idx]
    pred_braak = pred_braak[idx]
    vals = []
    for b in np.unique(braak):
        idx = braak == b
        vals.append(pred_braak[idx])

    return vals


def plot_hotspot_heatmap(
        df,
        gene_names,
        braak_filtered,
        genes_filtered,
        clusters=None,
        normalized=True,
        save_fig_fn=None,
):

    f, ax = plt.subplots(1, 2, figsize=(6, 3.25))

    x = (braak_filtered - braak_filtered[0]) / (braak_filtered[-1] - braak_filtered[0])

    if clusters is None:
        clusters = np.sort(df.Module.unique())
    y_full = []
    y_mean = []
    y_se = []
    raw_clusters = []
    cuts = [0]
    for c in clusters:
        raw_clusters.append([])

        df1 = df[df.Module == c]
        cuts.append(cuts[-1] + len(df1))
        y = []
        for g in df1.featurekey.values:

            if not g in gene_names:
                continue

            j = np.where(gene_names == g)[0]
            if len(j) == 0:
                continue
            j = j[0]
            raw_clusters[-1].append(genes_filtered[:, j])
            if normalized:
                z = genes_filtered[:, j] - np.min(genes_filtered[:, j])
                z /= np.max(z)
            else:
                z = genes_filtered[:, j]
            y_full.append(z)
            y.append(z)

        y = np.stack(y, axis=0)
        y_mean.append(np.mean(y, axis=0))
        # y_se.append(np.std(y, axis=0) / np.sqrt(y.shape[0]))
        y_se.append(np.std(y, axis=0))

    y_full = np.stack(y_full)
    cuts = cuts[1:-1]
    ax[0].imshow(y_full, aspect='auto', cmap="cividis")
    ax[0].hlines(cuts, 0, 999, 'r')
    # ax.legend(fontsize=8)
    ax[0].set_xlabel("Braak pseudotime")
    ax[0].set_ylabel("Normalized expression")
    ax[0].set_xticks([0, 999], [0, 1])
    ax[0].set_yticks([])

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='4%', pad=0.02)
    norm = matplotlib.colors.Normalize(vmin=np.min(y_full), vmax=np.max(y_full))
    cbar = f.colorbar(
        cm.ScalarMappable(norm=norm, cmap="cividis"), cax=cax, orientation='vertical',
    )

    N = len(clusters)
    for n in range(N):
        u0 = y_mean[n] - y_se[n]
        u1 = y_mean[n] + y_se[n]
        # ax.fill_between(x, u0, u1, facecolor="gray")
        ax[1].plot(x, y_mean[n], label=clusters[n], linewidth=2)
    ax[1].legend(fontsize=6)
    ax[1].set_ylabel("Normalized expression")
    ax[1].set_xlabel("Braak pseudotime")
    ax[1].set_xticks([0, 1], [0, 1])
    ax[1].set_yticks([0, 1], [0, 1])
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0, 1])

    # ax.set_title("Hotspot pseudotime")
    plt.tight_layout()
    plt.savefig("/home/masse/work/pd/figures/hotspot0.pdf")
    plt.show()

    f, ax = plt.subplots(N, 1, figsize=(3, 3.35), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for n in range(N):
        u0 = y_mean[n] - y_se[n]
        u1 = y_mean[n] + y_se[n]
        ax[n].fill_between(x, u0, u1, facecolor="gray")
        ax[n].plot(x, y_mean[n], 'k-', label=f"Module {clusters[n]}")
        ax[n].set_xlim([0, 1])
        ax[n].set_ylim([0, 1])
        ax[n].legend(fontsize=6)
        # ax[n].set_title(f"Module {clusters[n]}")
        # ax[n].set_ylabel("Norm")

        if n == N - 1:
            ax[n].set_xlabel("Braak pseudotime")
            ax[n].set_xticks([0, 1], [0, 1])
            ax[n].set_yticks([0, 1], [0, 1])
        else:
            ax[n].set_xticks([])
            ax[n].set_yticks([])

    plt.tight_layout()
    if save_fig_fn is not None:
        plt.savefig(save_fig_fn)
    plt.show()

    return x, y_mean, y_se, raw_clusters


def plot_pls_results(data, save_fig_fn=None):
    f, ax = plt.subplots(1, 3, figsize=(8, 3))

    vel = np.diff(data["genes_filtered_pls"], axis=0)
    vel /= np.sqrt(np.sum(vel ** 2, axis=1, keepdims=True))

    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(vel)
    clusters = kmeans.labels_
    transitions = []
    cols = ["m", "k"]

    for k in range(3):
        idx = clusters == k
        transitions.append(np.where(idx)[0][-1])
    transitions = np.array(transitions)
    transitions = transitions[transitions < 900]
    print(transitions)

    count = 0

    for i in range(2):
        for j in range(i + 1, 3):
            ax[count].scatter(data["gene_means_pls"][:, i], data["gene_means_pls"][:, j], c=data["pred_braak"], s=5)
            ax[count].scatter(
                data["genes_filtered_pls"][:, i], data["genes_filtered_pls"][:, j], c=data["braak_filtered"], s=10
            )
            ax[count].scatter(data["genes_filtered_pls"][transitions[0], i],
                              data["genes_filtered_pls"][transitions[0], j], c=cols[0], s=25)
            ax[count].scatter(data["genes_filtered_pls"][transitions[1], i],
                              data["genes_filtered_pls"][transitions[1], j], c=cols[1], s=25)
            ax[count].set_xlabel(f"Latent var {i}")
            ax[count].set_ylabel(f"Latent var {j}")
            count += 1

    plt.tight_layout()
    if save_fig_fn is not None:
        plt.savefig(save_fig_fn)
    plt.show()


def donor_cell_types(
        adata,
        n_min_cells=20,
        sigma=np.sqrt(0.1),
        braak_feature_name="pred_path_braak_lb_condensed",
):
    data = {
        "pred_braak": [],
        "subclasses": ["Homeo", "Adapt", "DAM", "Prolif", "PVM"],
        "subtypes": [s for s in adata.obs.derived_subtype2_Dec2024.unique() if s != "NA"],
        "subclass_pct": [],
        "subtype_pct": [],
    }
    for n, donor in enumerate(adata.uns["donors"]):
        if adata.uns["donor_cell_count"][n] < n_min_cells:
            continue

        data["pred_braak"].append(adata.uns[f"donor_{braak_feature_name}"][n])
        a = adata[adata.obs.SubID == donor]

        subclass_pct = []
        for s in data["subclasses"]:
            temp = np.array([s in st for st in a.obs.derived_subtype2_Dec2024.values])
            subclass_pct.append(np.mean(temp))
        data["subclass_pct"].append(subclass_pct)

        subtype_pct = []
        for s in data["subtypes"]:
            temp = np.array([s in st for st in a.obs.derived_subtype2_Dec2024.values])
            subtype_pct.append(np.mean(temp))
        data["subtype_pct"].append(subtype_pct)

    data["pred_braak"] = np.stack(data["pred_braak"])
    data["subclass_pct"] = np.stack(data["subclass_pct"])
    data["subtype_pct"] = np.stack(data["subtype_pct"])

    data["pred_braak_filtered"], data["subclass_pct_filtered"] = filter_vals_interp(
        data["pred_braak"], data["subclass_pct"], sigma, x_interp=None, max_range=10_000, n_time_pts=1000
    )
    data["pred_braak_filtered"], data["subtype_pct_filtered"] = filter_vals_interp(
        data["pred_braak"], data["subtype_pct"], sigma, x_interp=None, max_range=10_000, n_time_pts=1000
    )

    return data


def find_transitions(data, n_time_pts = 1000, n_clusters = 3):
    vel = np.diff(data["genes_filtered_pls"], axis=0)
    vel /= np.sqrt(np.sum(vel ** 2, axis=1, keepdims=True))

    kmeans = KMeans(n_clusters = n_clusters, random_state=0, n_init="auto").fit(vel)
    clusters = kmeans.labels_
    transitions = []

    for k in range(n_clusters):
        idx = clusters == k
        transitions.append(np.where(idx)[0][-1])
    transitions = np.array(transitions)
    transitions = transitions[transitions < n_time_pts * 0.99]

    return transitions, clusters


def process_data(
        model_fn=None,  # model_fn (h5ad file) or data must specified
        data=None,
        n_min_cells=20,
        sigma=0.33,
        shuffle_braak=False,
        n_time_pts=1000,
        braak_feature_name="pred_path_braak_lb_condensed",
        zenith_save_fn="../zenith/PD_myeloid_zenith_input.csv",
        gene_names = None,
):
    assert not (model_fn is None and data is None)

    if data is None:
        adata = sc.read_h5ad(model_fn, "r")
        # print(adata.uns["model_fns"])
        idx = adata.uns["donor_cell_count"] >= n_min_cells

        if "donor_path_braak_lb" in adata.uns.keys():
            k0 = "donor_path_braak_lb"
            k1 = f"donor_{braak_feature_name}"
            k2 = "path_braak_lb"
            k3 = braak_feature_name
            compute_cell_types = True
        else:
            k0 = "donor_BRAAK_AD"
            k1 = "donor_pred_BRAAK_AD"
            k2 = "BRAAK_AD"
            k3 = "pred_BRAAK_AD"
            compute_cell_types = False

        data = {
            "pred_braak": adata.uns[k1][idx],
            "braak": adata.uns[k0][idx],
            "gene_means": adata.uns["donor_gene_means"][idx, :],
        }
        idx = np.argsort(data["pred_braak"])
        data["pred_braak"] = data["pred_braak"][idx]
        data["braak"] = data["braak"][idx]
        data["gene_means"] = data["gene_means"][idx, :]

    data["pred_braak_violin"] = violin_plot_data(
        adata.obs[k2].values,
        adata.obs[k3].values,
    )
    data["r_cell"], _ = stats.pearsonr(adata.obs[k2].values, adata.obs[k3].values)
    data["r_donor"], _ = stats.pearsonr(data["braak"], data["pred_braak"])

    if compute_cell_types:
        data["cell_types"] = donor_cell_types(adata, n_min_cells=n_min_cells, sigma=sigma)

    pls = PLSRegression(n_components=10, scale=False, max_iter=1000, tol=1e-07)

    y = data["pred_braak_full"] if "pred_braak_full" in data else data["pred_braak"]

    if shuffle_braak:
        np.random.shuffle(y)

    pls.fit(data["gene_means"], y)

    data["gene_means_pls"] = pls.transform(data["gene_means"])

    # pls.fit(data["gene_means"], data["braak"])
    # data["gene_means_pls"] = pls.transform(data["gene_means"])

    data["braak_filtered"], data["genes_filtered_pls"] = filter_vals_interp(
        data["pred_braak"], data["gene_means_pls"], sigma, x_interp=None, max_range=10_000, n_time_pts=n_time_pts,
    )

    data["transitions"], data["clusters"] = find_transitions(data)

    _, data["genes_filtered_full"] = filter_vals_interp(
        data["pred_braak"], data["gene_means"], sigma, x_interp=None, max_range=10_000, n_time_pts=n_time_pts,
    )

    if gene_names is not None:
        data["braak_mid_pts"] = zenith_wrapper(data, gene_names, save_fn = zenith_save_fn)

    return data

def plot_basics(data, save_fig_fn = None):

    f, ax = plt.subplots(3, 2, figsize=(6.25, 7))

    ax[0, 1].violinplot(data["pred_braak_violin"], np.arange(7))
    ax[0, 1].plot(data["braak"], data["pred_braak"],'k.', markersize=3)
    ax[0, 1].set_title(f"R-cell: {data["r_cell"]:1.3f}  R-donor: {data["r_donor"]:1.3f}")
    ax[0, 1].set_xlabel("Braak")
    ax[0, 1].set_ylabel("Predicted Braak")

    m0 = np.min(data["pred_braak"])
    m1 = np.max(data["pred_braak"])
    braak_norm = (data["pred_braak"] - m0) / (m1 - m0)
    braak_norm_filtered = (data["braak_filtered"] - m0) / (m1 - m0)

    i = 0
    tr = data["transitions"]
    cols = ["m", "c"]
    for j in range(2):
        ax[1, j].scatter(data["gene_means_pls"][:, i], data["gene_means_pls"][:, j+1], c=braak_norm, s=8)
        ax[1, j].scatter(
            data["genes_filtered_pls"][:, i], data["genes_filtered_pls"][:, j+1], c=braak_norm_filtered, s=10
        )
        ax[1, j].scatter(data["genes_filtered_pls"][tr[0], i], data["genes_filtered_pls"][tr[0], j+1], c=cols[0], s=25)
        ax[1, j].scatter(data["genes_filtered_pls"][tr[1], i], data["genes_filtered_pls"][tr[1], j+1], c=cols[1], s=25)
        ax[1, j].set_xlabel(f"Latent var {i}")
        ax[1, j].set_ylabel(f"Latent var {j+1}")

    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes('right', size='4%', pad=0.02)
    norm = matplotlib.colors.Normalize(vmin=np.min(braak_norm), vmax=np.max(braak_norm))
    cbar = f.colorbar(
        cm.ScalarMappable(norm=norm, cmap="cividis"), cax=cax, orientation='vertical',
    )

    x = data["cell_types"]["pred_braak_filtered"]
    x -= np.min(x)
    x /= np.max(x)
    y = data["cell_types"]["subclass_pct_filtered"]
    y -= np.min(y, axis=0, keepdims=True)
    y /= np.max(y, axis=0, keepdims=True)
    ax[2, 0].plot([x[tr[0]], x[tr[0]]], [0, 1], 'k--')
    ax[2, 0].plot([x[tr[1]], x[tr[1]]], [0, 1], 'k--')
    for n, s in enumerate(data["cell_types"]["subclasses"]):
        if s == "PVM":
            continue
        line = "-" if n % 2 == 0 else "--"
        ax[2, 0].plot(x, y[:, n], line, label=s, linewidth=2)
    ax[2, 0].legend(fontsize=7)
    ax[2, 0].set_xlabel("Normalized predicted Braak")
    ax[2, 0].set_ylabel("Normalized proportion")
    ax[2, 0].set_xticks([0, 1], [0, 1])
    ax[2, 0].set_yticks([0, 1], [0, 1])
    #ax[2, 0].set_xlim([0, 1])
    #ax[2, 0].set_ylim([0, 1])

    y = data["cell_types"]["subtype_pct_filtered"]
    y -= np.min(y, axis=0, keepdims=True)
    y /= np.max(y, axis=0, keepdims=True)
    subtypes = ["Mg_Homeo_CECR2", "Mg_Homeo_PICALM", "Mg_Adapt_TMEM163", "Mg_Adapt_HSPA1A", "Mg_Adapt_CCL3"]
    ax[2, 1].plot([x[tr[0]], x[tr[0]]], [0, 1], 'k--')
    ax[2, 1].plot([x[tr[1]], x[tr[1]]], [0, 1], 'k--')
    for n, s in enumerate(subtypes):
        line = "-" if n % 2 == 0 else "--"
        ax[2, 1].plot(x, y[:, n], line, label=s, linewidth=2)
    ax[2, 1].legend(fontsize=7)
    ax[2, 1].set_xticks([0, 1], [0, 1])
    ax[2, 1].set_yticks([0, 1], [0, 1])
    #ax[2, 1].set_xlim([0, 1])
    #ax[2, 1].set_ylim([0, 1])

    plt.tight_layout()
    if save_fig_fn is not None:
        plt.savefig(save_fig_fn)
    plt.show()

def plot_hotspot_overaly(
        clusters_pd,
        clusters_ad,
        x_pd, x_ad,
        clusters = [2, 3, 7, 8, 10],
        save_fig_fn = None,
):

    y_pd = []
    y_ad = []
    y_pd_se = []
    y_ad_se = []

    for n in range(5):
        temp_pd = []
        temp_ad = []
        for y0, y1 in zip(clusters_pd[n], clusters_ad[n]):
            y_concat = np.concatenate((y0, y1), axis=-1)
            m0 = np.min(y_concat)
            m1 = np.max(y_concat) - m0
            temp_pd.append((y0 - m0) / m1)
            temp_ad.append((y1 - m0) / m1)
            # temp_pd.append(y0-m0)
            # temp_ad.append(y1-m0)
        temp_pd = np.stack(temp_pd)
        temp_ad = np.stack(temp_ad)
        y_pd.append(np.mean(temp_pd, axis=0))
        y_ad.append(np.mean(temp_ad, axis=0))
        y_pd_se.append(np.std(temp_pd, axis=0))
        y_ad_se.append(np.std(temp_ad, axis=0))

    N = len(clusters)
    f, ax = plt.subplots(5, 1, figsize=(4, 8), sharex=True, sharey=False)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    y_mean = [y_pd, y_ad]
    y_se = [y_pd_se, y_ad_se]
    x = [x_pd, x_ad]
    cols = ["g", "m"]
    mod = ["PD", "AD"]

    for n in range(N):
        for i in range(2):
            u0 = y_mean[i][n] - y_se[i][n]
            u1 = y_mean[i][n] + y_se[i][n]
            ax[n].fill_between(x[i], u0, u1, facecolor=cols[i], alpha=0.33)
            ax[n].plot(x[i], y_mean[i][n], '-', color=cols[i], label=f"{mod[i]} - Mod. {clusters[n]}")
        ax[n].set_xlim([0, 1])
        ax[n].set_ylim([0, 1])
        ax[n].legend(fontsize=6)
        # ax[n].set_title(f"Module {clusters[n]}")
        # ax[n].set_ylabel("Norm")

        if n == N - 1:
            ax[n].set_xlabel("Braak pseudotime")
            ax[n].set_ylabel("Normalized expression")
            ax[n].set_xticks([0, 1], [0, 1])
            ax[n].set_yticks([0, 1], [0, 1])
        else:
            ax[n].set_xticks([])
            ax[n].set_yticks([])

    plt.tight_layout()
    if save_fig_fn is not None:
        plt.savefig(save_fig_fn)
    plt.show()


class PathEnrichment:

    def __init__(
            self,
            input_path,
            FDR_cutoff = 0.001,
            n_min_significant_windows = 4,
            min_genes = 10,
            max_genes = 250,
    ):

        self.FDR_cutoff = FDR_cutoff
        self.n_min_significant_windows = n_min_significant_windows
        self.min_genes = min_genes
        self.max_genes = max_genes

        # Load GO BP gene sets from Zenith
        fn = "/home/masse/work/data/annotations/go_bp_pathways.json"
        with open(fn, 'r') as file:
            self.go_bp_gene_sets = json.load(file)

        self._get_input_files(input_path)
        self._load_data()
        self._filter_data()

    def _get_input_files(self, input_path):

        fns = os.listdir(input_path)
        self.fns = [os.path.join(input_path, fn) for fn in fns if not "lock" in fn]

    def _load_data(self):

        dataframes = []
        for fn in self.fns:
            df = pd.read_csv(fn)
            df["fn"] = os.path.split(fn)[-1]
            go_ids = []
            pathways = []
            for g in df["Unnamed: 0"]:
                g1 = f"GO:{g[2:9]}"
                go_ids.append(g1)
                p = g.split(":")[1][1:]
                pathways.append(p)
            df["go_id"] = go_ids
            df["pathway"] = pathways
            dataframes.append(df.copy())

        self.df_full = pd.concat(dataframes, axis=0)
        self.df_full = self.df_full.sort_values(["FDR"])
        self.df_full = self.df_full.reset_index(None)

    def _filter_data(self):

        # save_fn can be used ot save results as csv for rrvgo analysis
        data = {
            "go_id": [],
            "pathway": [],
            "original_pathway": [],
            "FDR": [],
            "direction": [],
            "delta": [],
            "ngenes": [],
            "n_significant" : []
        }
        for g, p, f, d, dl, n, filename in zip(
                self.df_full.go_id.values,
                self.df_full["Unnamed: 0"].values,
                self.df_full.FDR.values,
                self.df_full.Direction.values,
                self.df_full.delta.values,
                self.df_full.NGenes.values,
                self.df_full.fn.values
        ):
            if f > self.FDR_cutoff:
                continue
            if n < self.min_genes or n > self.max_genes:
                continue

            if g in data["go_id"]:

                j = np.where(np.array(data["go_id"]) == g)[0][0]
                if f < data["FDR"][j]:
                    data["FDR"][j] = f
                if f < self.FDR_cutoff:
                    data["n_significant"][j] += 1

            else:
                data["go_id"].append(g)
                data["pathway"].append(p.split(":")[1][1:])
                data["original_pathway"].append(p)
                data["FDR"].append(f)
                data["direction"].append(d)
                data["delta"].append(dl)
                data["ngenes"].append(n)
                data["n_significant"].append(0 if f > self.FDR_cutoff else 1)

        self.df_filtered = pd.DataFrame(data=data)
        self.df_filtered = path_enrichment_utils.remove_bad_terms( self.df_filtered)
        self.df_filtered = self.df_filtered[self.df_filtered.n_significant >= self.n_min_significant_windows]
        self.df_filtered =  self.df_filtered.sort_values(["FDR"])
        self.df_filtered =  self.df_filtered.reset_index(None)

        self.df_filtered = self._drop_similar_pathways(self.df_filtered)

    def save_filtered_data(self, save_fn):
        self.df_filtered.to_csv(save_fn)

    def _drop_similar_pathways(self, df, overalp_threshold = 0.66):

        idx = []
        idx_drop = []
        for n in range(len(df)):
            path0 = df.loc[n].original_pathway
            genes0 = set(self.go_bp_gene_sets[path0])
            fdr0 = df.loc[n].FDR

            include = True

            for i in range(len(df)):
                if i == n:
                    continue
                path1 = df.loc[i].original_pathway
                genes1 = set(self.go_bp_gene_sets[path1])
                fdr1 = df.loc[i].FDR

                if fdr1 > fdr0:
                    continue

                ratio = len(genes0.intersection(genes1)) / len(genes0.union(genes1))
                if ratio > overalp_threshold:
                    include = False

                if path0.split(":")[1] in path1.split(":")[1]:
                    include = False

            if include:
                idx.append(n)
            else:
                idx_drop.append(n)

        df_reduced = df.iloc[idx]
        df_reduced.reset_index(drop=True, inplace=True)
        return df_reduced

    def _generate_data_time_pts(self, pathways):

        df_full = self.df_full[self.df_full.pathway.isin(pathways)]
        fns = df_full.fn.unique()
        n_time_pts = len(fns)
        n_paths = len(pathways)

        self.fdr = np.zeros((n_paths, n_time_pts)) # [pathway, time pts]
        self.z_score = np.zeros((n_paths, n_time_pts))

        for i, p in enumerate(pathways):
            df_rows = df_full[df_full.pathway == p]
            df_rows = df_rows.reset_index()
            for n in range(n_time_pts):
                for j in range(len(df_rows)):
                    if f"Braak_{n+1}_" in df_rows.loc[j].fn:
                        self.fdr[i, n] = - np.sign(df_rows.loc[j].delta) * np.log10(df_rows.loc[j].FDR)
                        self.z_score[i, n] = df_rows.loc[j].delta / df_rows.loc[j].se


    def _generate_dataframes_time_pts(
        self,
        pathways,
        time_pts = None,
        cluster_pathways = False,
        n_interp_pts = 1000,
    ):

        n_paths = len(pathways)
        new_pathways = []
        for p in pathways:
            new_pathways.append(path_enrichment_utils.condense_pathways(p))

        if cluster_pathways:
            Z = linkage(np.reshape(self.z_score, (self.z_score.shape[0], -1)), method='ward')
            s = dendrogram(Z, no_plot=True)
            idx = [s["leaves"][i] for i in range(n_paths)]
            new_pathways = [new_pathways[i] for i in idx]
            self.z_score = self.z_score[idx]
            self.fdr = self.fdr[idx]

        time_pts = self.fdr.shape[1] if time_pts is None else time_pts
        t_new = np.linspace(time_pts[0], time_pts[-1], n_interp_pts)
        n_paths = len(pathways)

        fdr_interp = np.zeros((n_paths, n_interp_pts))
        z_score_interp = np.zeros((n_paths, n_interp_pts))

        for n in range(n_paths):
            fdr_interp[n, :] = np.interp(t_new, time_pts, self.fdr[n, :])
            z_score_interp[n, :] = np.interp(t_new, time_pts, self.z_score[n, :])

        self.df_fdr = pd.DataFrame(fdr_interp, index=new_pathways, columns=t_new)
        self.df_z_score = pd.DataFrame(z_score_interp, index=new_pathways, columns=t_new)
        self.time_pts_interp = t_new


    def generate_sliding_window_figure(
        self,
        pathways,
        time_pts=None,
        pred_braak_traj=None,
        transition_pts=None,
        cluster_pathways=False,
        n_interp_pts=1000,
        figsize=(6.0, 6.0),
        save_fig_fn=None,
    ):

        self._generate_data_time_pts(pathways)
        self._generate_dataframes_time_pts(
            pathways,
            time_pts=time_pts,
            cluster_pathways=cluster_pathways,
            n_interp_pts=n_interp_pts,
        )

        f, ax = plt.subplots(1, 1, figsize=figsize, sharey=True)
        max_val = 10
        sns.heatmap(
            self.df_z_score,
            ax=ax,
            cmap='RdBu_r',
            center=True,
            vmin=-max_val,
            vmax=max_val,
            cbar=False,
            xticklabels=False,
            yticklabels=True,
        )

        if transition_pts is not None and pred_braak_traj is not None:
            t0, t1 = transition_pts[0], transition_pts[1]
            t0 = int(t0 * len(pred_braak_traj))
            t1 = int(t1 * len(pred_braak_traj))
            t0 = np.argmin(np.abs(self.time_pts_interp - pred_braak_traj[t0]))
            t1 = np.argmin(np.abs(self.time_pts_interp - pred_braak_traj[t1]))
            ax.vlines([t0], 0, len(pathways), linewidth=0.5, colors='k')
            ax.vlines([t1], 0, len(pathways), linewidth=0.5, colors='k')

        ax.tick_params(axis='y', labelsize=8)
        ax.set_xlabel("Disease pseudotime")
        plt.tight_layout()
        if save_fig_fn is not None:
            plt.savefig(save_fig_fn)


def format_dataframe_decimals(df, decimals=3):

    def format_value(value):
        if isinstance(value, (int, float)):
            return f"{value:.{decimals}f}"  # Format to specified decimals
        else:
            return value

    formatted_df = df.applymap(format_value)

    # Format column names
    new_columns = []
    for n, col in enumerate(formatted_df.columns):
        # if isinstance(col, (int, float)):
        #    new_columns.append(f"{col:.{decimals}f}")
        # else:
        #    new_columns.append(col)
        new_columns.append(f"Pseudotime {n}")

    formatted_df.columns = new_columns

    return formatted_df

