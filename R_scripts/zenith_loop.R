### Calculates the Zenith pathway enrichment for each time window

library(GSEABase)
library(zenith)

go.gs.bp = get_GeneOntology(onto="BP", to="SYMBOL")

base_dir = "../zenith/"
input_fn = "PD_myeloid_zenith_input.csv"
fn = paste0(base_dir, input_fn)

df = read.csv(fn)
for (j in 1:48){
	res = df[,2+j]
	names(res) = df[,2]
	res.gsa = zenithPR_gsa(statistics=res, ids=names(res), geneSets=go.gs.bp,  progressbar=FALSE, use.ranks = FALSE, n_genes_min = 10,  inter.gene.cor=0.01)
	save_fn = paste0(base_dir, "zenith_output/", "Myeloid_Braak_", j, "_BP.csv")
	print(save_fn)
	write.csv(res.gsa, save_fn)
}

