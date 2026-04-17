import copy
import pandas as pd

"""Miscellaneous helper functions and constants"""

bad_path_words = [
    "limb",
    "blastocyst",
    "gastric",
    "podocyte",
    "cognition",
    "decidualization",
    "acrosome",
    "cardiac",
    "vocalization",
    "auditory stimulus",
    "sensory",
    "learning",
    "memory",
    "walking",
    "behavior",
    "social",
    "locomotor",
    "nervous system",
    "corpus callosum",
    "forebrain",
    "cerebral cortex",
    "hippocamp",
    "cerebellar",
    "cranial",
    "forelimb",
    "hindlimb",
    "startle",
    "prepulse inhibition",
    "dosage",
    "substantia nigra",
    "retina",
    "optic"
    "bone",
    "kidney",
    "glomerulus",
    "heart",
    "ventricular",
    "metanep",
    "nephron",
    "glomerular",
    "of muscle",
    "bone",
    "respiratory",
    "pigmentation",
    "outflow tract septum",
    "placenta",
    "olfactory",
    "aortic",
    "germ layer",
    "mesodermal",
    "epithelial",
    "pulmonary",
    "lung",
    "embronic",
    "embryo",
    "mammary",
    "egg",
    "sperm",
    "cadmium",
    "sarcoplasmic",
    "neuron fate",
    "pregnancy",
    "osteoblast",
    "prostate",
    "hepatocyte",
    "estrous",
    "muscle atrophy",
    "neuromuscular",
    "egg",
    "ovulation",
    "with host",
    "pancreatic",
    " organ ",
    "hematopoietic",
    "epiderm",
    " head ",
    "mesoderm",
    "endoderm",
    "cerebellum",
    "embryonic",
    "ossification",
    "cochlea",
    "digestive",
    "melanocyte",
    "lead ion",
    "coronary",
    "skeletal",
    "developmental growth",
    "metanephric",
    " otic ",
]

gwas_dict = {
    "alzBellenguezNoApoe": "AD_2022_Bellenguez",
    "ms": "MS_2019_IMSGC",
    "pd_without_23andMe": "PD_2019_Nalls",
    "migraines_2021": "Migraines_2021_Donertas",
    "als2021": "ALS_2021_vanRheenen",
    "stroke": "Stroke_2018_Malik",
    "epilepsyFocal": "Epilepsy_2018_ILAECCE",

    "sz3": "SCZ_2022_Trubetskoy",
    "bip2": "BD_2021_Mullins",
    "asd": "ASD_2019_Grove",
    "adhd_ipsych": "ADHD_2023_Demontis",
    "mdd_ipsych": "MDD_2023_AlsBroad",
    "ocd": "OCD_2018_IOCDF_GC",
    "insomn2": "Insomnia_2019_Jansen",
    "alcohilism_2019": "Alcoholism_2019_SanchezRoige",
    "tourette": "Tourettes_2019_Yu",
    "intel": "IQ_2018_Savage",
    "eduAttainment": "Education_2018_Lee",
}

def condense_pathways(pathway):

    pathway = pathway.split()

    for n, p in enumerate(pathway):
        p = copy.deepcopy(p.lower())
        if p == "positive":
            pathway[n] = "pos."
        elif p == "negative":
            pathway[n] = "neg."
        elif p == "regulation":
            pathway[n] = "reg."
        elif p == "response":
            pathway[n] = "resp."
        elif p == "neurotransmitter":
            pathway[n] = "neurotrans."
        elif p == "neurotransmitter":
            pathway[n] = "neurotrans."
        elif p == "modulation":
            pathway[n] = "mod."
        elif p == "differentiation":
            pathway[n] = "diff."
        elif p == "biosynthetic":
            pathway[n] = "biosynth."
        # elif p == "mitochondrial":
        #    pathway[n] = "mito."
        elif p == "nitric-oxide":
            pathway[n] = "NO"
        elif p == "glutamate":
            pathway[n] = "GLU"
        elif p == "glutamatergic":
            pathway[n] = "GLU"
        elif p == "homeostasis":
            pathway[n] = "homeo."
        elif p == "signaling":
            pathway[n] = "sign."
        elif p == "exocytosis":
            pathway[n] = "exocyt."
        elif p == "colony-stimulating":
            pathway[n] = "colony-stim."
        elif p == "derived":
            pathway[n] = "der."
        elif p == "multicellular":
            pathway[n] = "multicell."
        elif p == "presentation":
            pathway[n] = "present."
        elif p == "organismal-level":
            pathway[n] = "org.-level"
        elif p == "proliferation":
            pathway[n] = "prolif."
        elif p == "homeostasis":
            pathway[n] = "homeo."
        elif p == "t-helper":
            pathway[n] = "T-help."
            # elif p == "immune":
            #    pathway[n] = "imm.."
        elif p == "helper":
            pathway[n] = "help."
        elif p == "ligand-gated":
            pathway[n] = "lig.-gated"
        elif p == "macrophage":
            pathway[n] = "macrophg."
        elif p == "associated":
            pathway[n] = "ass."
        elif p == "transport":
            pathway[n] = "trans."
        elif p == "synthesis":
            pathway[n] = "synth."
        elif p == "contraction":
            pathway[n] = "contract."
        elif p == "migration":
            pathway[n] = "migrat."
        elif p == "processing":
            pathway[n] = "proc."
        elif p == "exogenous":
            pathway[n] = "exon."
        elif p == "gamma-aminobutyric acid":
            pathway[n] = "GABA"

        for n in range(len(pathway) - 1):
            if pathway[n] == "calcium" and pathway[n + 1] == "ion":
                pathway[n] = "Ca2+"
                pathway[n + 1] = ""
            elif pathway[n] == "iron" and pathway[n + 1] == "ion":
                pathway[n] = "Fe2+"
                pathway[n + 1] = ""
            elif pathway[n] == "manganese" and pathway[n + 1] == "ion":
                pathway[n] = "Mn2+"
                pathway[n + 1] = ""
            elif pathway[n] == "calcium" and "ion-" in pathway[n + 1]:
                pathway[n] = "Ca2+ " + pathway[n + 1][4:]
                pathway[n + 1] = ""
        for n in range(len(pathway) - 2):
            if pathway[n] == "mhc" and pathway[n + 1] == "class" and pathway[n + 2] == "ii":
                pathway[n] = "MHC-II"
                pathway[n + 1] = ""
                pathway[n + 2] = ""

        pathway = " ".join(pathway)
        pathway = pathway.split()
        return " ".join(pathway)


def remove_bad_terms(df):
    idx = []
    for i in range(len(df)):
        include = True
        for w in bad_path_words:
            if w in df.loc[i]["pathway"]:
                include = False
        idx.append(include)

    return df[idx]