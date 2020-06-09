"""
Preprocess TOX21 data.

TOX21 is multiclass classification on several biochemical "tasks".
    
We select one task, NR-ER, to create a binary classification dataset, 
to be used in this analysis.
"""

import os
import pandas as pd

DATA_URL = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
#Task to select
TASK = "NR-ER"
data_dir = "../data"

df = pd.read_csv(DATA_URL)
dff = df[[TASK, "mol_id", "smiles"]]
dff.dropna(inplace=True)
dff.rename({"NR-ER": "p_np", "mol_id":"name"}, axis=1, inplace=True)
dff["p_np"] = dff["p_np"].astype('int')

out_fn = "tox21_{}_processed.csv".format(TASK)
out_fp = os.path.join(data_dir, out_fn)

dff.to_csv(out_fp, index=False)