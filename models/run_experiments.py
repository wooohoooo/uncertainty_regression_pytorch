import numpy as np
import torch

seed = 42 #424
np.random.seed(seed)
torch.manual_seed(seed)
#NOTE! This only works for non cudnn. gpu needs
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from helpers import showcase_code

import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import pickle

from experimentation import ExperimentAnalyzer
import os
import time
import pandas as pd

files = [x for x in os.listdir('pickles') if x.endswith('pkl')]
df_dict = {}

for file in files:
    with  open('pickles\\'+file, "rb") as f:
        exp = pickle.load(f)



    analyzer = ExperimentAnalyzer(exp)
    df1 = analyzer.analysis()
    print(f'{exp.model_name}')
    print(df1.describe().to_latex())
    analyzer.get_outlier_indices()
    analyzer.plot_outcomes()
    plt.figure()
    df2 = analyzer.analysis()


    print(df2.describe().to_latex())

    analyzer.plot_distribution_of_metrics()

    analyzer.plot_models()

    analyzer.plot_models('cobeau')

    analyzer.plot_models('nlpd')


#     try:
#         analyzer.plot_outlier_models()



#     except Exception as e:
#         print(e)


    
    
