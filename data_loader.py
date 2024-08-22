import numpy as np
from nilearn.datasets import fetch_adhd, fetch_abide_pcp
from nilearn.input_data import NiftiMapsMasker

def load_adhd_data(masker):
    adhd = fetch_adhd(n_subjects=30)
    lst_adhd, lst_label_adhd = [], []
    for func_preproc_file, phenotypic_file in zip(adhd.func, adhd.phenotypic):
        time_series = masker.transform(func_preproc_file)
        lst_adhd.append(time_series)
        lst_label_adhd.append(0)
    return lst_adhd, lst_label_adhd

def load_asd_data(masker):
    autsm = fetch_abide_pcp(n_subjects=30, derivatives=['func_preproc'], quality_checked=True, DX_GROUP=1)
    lst_autsm, lst_label_autsm = [], []
    for func_preproc_file, phenotypic_file in zip(autsm.func_preproc, autsm.phenotypic):
        time_series_aut = masker.transform(func_preproc_file)
        lst_autsm.append(time_series_aut)
        lst_label_autsm.append(1)
    return lst_autsm, lst_label_autsm
