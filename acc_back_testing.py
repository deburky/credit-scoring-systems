import pandas as pd
import numpy as np

def accuracy_back_test(rating_scale_midpoints, default_vector, pd_vector):
    
    """
    Accuracy back-test
    
    Rank through rating scale based on PDs midpoints
    
    Example: bin_edges = [0.001, 0.02, 0.049, 0.071, 0.1055, 0.2000, 0.4754]
    
    """
    
    i = 0
    acc_report = np.array([])

    for cutoff in rating_scale_midpoints:
     
     # sum of defaults
     sum_dr = sum(default_vector)
    
     # cut-off to convert probabilities into class
     vector_pd = pd.DataFrame(np.where(pd_vector >= cutoff,1,0), index=None, columns=['pd'])
     vector_dr = pd.DataFrame(np.asarray(default_vector), index=None, columns=['default'])
     vectors = pd.merge(vector_pd, vector_dr, left_index=True, right_index=True, how='inner')
     
     # share of defaults in element from total defaults
     acc = vectors[vectors['pd'] == 1]['default'].sum() / sum_dr
     acc_report = np.append(acc_report, acc)
    
    # join midpoints and accuracy (share bad of total bad)
    new_arr = np.transpose(np.vstack((np.transpose(rating_scale_midpoints, axes=None), acc_report)))
    df_cut_off_accuracy = pd.DataFrame(new_arr, columns=['pd_midpoint', 'accuracy'])

    i += 1 
    
    return df_cut_off_accuracy
