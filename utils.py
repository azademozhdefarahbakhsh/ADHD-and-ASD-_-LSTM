import numpy as np

def pad_and_reshape_data(lst_adhd, lst_label_adhd):
    max_len_image = np.max([len(i) for i in lst_adhd])
    lst_adhd_reshaped = []
    for subject_data in lst_adhd:
        N = max_len_image - len(subject_data)
        padded_array = np.pad(subject_data, ((0, N), (0, 0)), 'constant', constant_values=(0))
        lst_adhd_reshaped.append(padded_array)
    
    x_data = np.array(lst_adhd_reshaped)
    y_data = np.array(lst_label_adhd)
    
    return x_data, y_data
