# %%
import sys
import pandas as pd 
import os 
# %%
embedding_path = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/"
meta_df = (
    pd.read_csv(embedding_path + "ckd_embedding_full_v3_icd_stage_filter/meta_v3.csv", sep="$")
    .merge(
        pd.read_csv(embedding_path + "CKD-supplemental_pull.rpt", sep="$")
        , on='PatientID', how='left'
    )
)
meta_df['date'] = pd.to_datetime(meta_df['date'])
meta_df['DOB'] = pd.to_datetime(meta_df['DOB'])


meta_df.head()

# %%
print("Number of encounters per patient by max stage")
grp1 = meta_df.groupby(["max_stage", "PatientID"])["META_1"].agg(["nunique"]).groupby("max_stage").describe()
print(grp1)
grp1.to_csv("table_enc_by_stage.csv")
# %%
print("Among progression patients, what's the distribution of diagnoses")
grp2 = meta_df[meta_df["CKD_stage_numeric"] > 0].groupby(["max_stage", "PatientID"])["CKD_stage_numeric"].agg(["min"]).groupby("max_stage").describe()
print(grp2)
grp2.to_csv("table_diag_dist.csv")
# %%
# Table 1 construction
# grouping by Sex, Race, CKD
meta_df["CKD-Conversion"] = (meta_df["max_stage"] > 3.0).astype(int)
# %%

def aggregate(df):
    d = {}
    d['n_enc'] = df.META_1.nunique()
    d['obs_duration'] = (df['date'].max() - df['date'].min()).days
    d['initial_age'] = (df['date'].min() - df['DOB'].min()).days // 365
    d['CKD-Conversion'] = df['CKD-Conversion'].max()
    return pd.Series(d, index=['n_enc', 'obs_duration', 'initial_age', 'CKD-Conversion'])


agg_df = meta_df\
    .groupby(
        [
            "CKD-Conversion", 
            "Sex", 
            "EthnoRacialCategory", 
            "PatientID"], as_index=False)\
    .apply(aggregate)
# %%
# filter out encounters to in person only
from tableone import TableOne
# %%
columns = [
    'initial_age', 
    'n_enc', 
    'obs_duration', 
    'Sex', 
    'CKD-Conversion',
    'EthnoRacialCategory'
]

categorical=['EthnoRacialCategory', 'Sex']
continuous = ['n_enc', 'obs_duration', 'initial_age']

groupby='CKD-Conversion'
# %%
mytable = TableOne(
    agg_df, 
    columns=columns, 
    categorical=categorical, 
    continuous=continuous, 
    groupby=groupby, 
    # nonnormal=nonnormal, 
    # rename=rename, 
    pval=False)

# %%
print(mytable.tabulate(tablefmt = "fancy_grid"))

# %%
mytable.to_csv("table1.csv")
# %%