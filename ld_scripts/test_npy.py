# %%
import os
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path 

# %%
os.listdir("./embeddings_subset_10")
os.listdir("./embeddings_subset_10/Z5052246")

# %%
embedding_path = "./embeddings_subset_10"
metadata_file = "meta_v3_subset_10.csv"

# %%
# for i in os.listdir(embedding_path):
#     print(i)
#     with np.load('Z5052246.npy') as data:
#         # Access contents here
#         pass
# %%
# 
import pandas as pd
import numpy as np
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def merge_embeddings_and_metadata(base_dir: str, df_metadata):
    """
    Loads embeddings from patientID.npy files and merges them with an 
    already loaded metadata DataFrame.

    Assumes the metadata DataFrame contains: PatientID, enc_id, and emb_id,
    where emb_id corresponds to the positional index in the NPY file.
    
    Args:
        base_dir (str): The path to the folder containing patient subdirectories.
        df_metadata (pd.DataFrame): The already loaded metadata DataFrame.

    Returns:
        pd.DataFrame: A single DataFrame with all patient metadata and their 
                      corresponding embeddings.
    """
    
    # 1. Check required columns in the provided DataFrame
    required_cols = ['PatientID', 'enc_id', 'emb_id']
    if not all(col in df_metadata.columns for col in required_cols):
        logger.error(f"Metadata DataFrame must contain columns: {', '.join(required_cols)}")
        return pd.DataFrame()

    all_merged_data = []
    unique_patient_ids = df_metadata['PatientID'].unique()

    # 2. Iterate through unique Patient IDs
    for patient_id in unique_patient_ids:
        patient_id_str = str(patient_id) 
        
        # Construct the path: base_dir/PatientID/PatientID.npy
        patient_folder_path = os.path.join(base_dir, patient_id_str)
        embeddings_path = os.path.join(patient_folder_path, f"{patient_id_str}.npy")

        # Check if the file exists
        if not os.path.isfile(embeddings_path):
            logger.warning(f"Embeddings file not found at {embeddings_path}. Skipping PatientID: {patient_id_str}.")
            continue
            
        # 3. Load the embeddings (assuming no file load errors for simplicity)
        embeddings_array = np.load(embeddings_path)

        # 4. Create a DataFrame for the embeddings and map the positional index
        df_embeddings = pd.DataFrame(embeddings_array)
        
        # The row index of the NPY array is saved as 'emb_id' to match the metadata
        df_embeddings['emb_id'] = df_embeddings.index 
        df_embeddings['PatientID'] = patient_id

        # 5. Filter the metadata for the current patient
        df_patient_metadata = df_metadata[df_metadata['PatientID'] == patient_id].copy()

        # 6. Merge the patient's metadata and embeddings
        # The merge key is ['PatientID', 'emb_id']
        df_merged_patient = pd.merge(
            df_patient_metadata,
            df_embeddings,
            on=['PatientID', 'emb_id'],
            how='inner' 
        )
        
        all_merged_data.append(df_merged_patient)
        # logger.info(f"Successfully merged {len(df_merged_patient)} rows for PatientID: {patient_id_str}")

    # 7. Concatenate all patient data
    if all_merged_data:
        df_final = pd.concat(all_merged_data, ignore_index=True)
        logger.info(f"Final merge complete. Total merged rows: {len(df_final)}")
        return df_final
    else:
        logger.warning("No data was successfully merged.")
        return pd.DataFrame()

#%%
import pandas as pd
import numpy as np
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def merge_embeddings_and_metadata2(base_dir: str, df_metadata: pd.DataFrame):
    """
    Loads embeddings from patientID.npy files and merges them with the metadata 
    DataFrame, storing the full embedding vector in a single column named "embedding".

    Args:
        base_dir (str): The path to the folder containing patient subdirectories.
        df_metadata (pd.DataFrame): The already loaded metadata DataFrame.

    Returns:
        pd.DataFrame: The metadata DataFrame with the new "embedding" column.
    """
    
    # 1. Check required columns
    required_cols = ['PatientID', 'enc_id', 'emb_id']
    if not all(col in df_metadata.columns for col in required_cols):
        logger.error(f"Metadata DataFrame must contain columns: {', '.join(required_cols)}")
        return pd.DataFrame()

    # Define the fixed embedding column name
    EMBEDDING_COL = 'embedding'
    
    # Prepare a list to store the embedding data for efficient merging later
    all_embedding_data = []

    unique_patient_ids = df_metadata['PatientID'].unique()

    # 2. Iterate through unique Patient IDs
    for patient_id in unique_patient_ids:
        patient_id_str = str(patient_id) 
        
        # Construct the path: base_dir/PatientID/PatientID.npy
        patient_folder_path = os.path.join(base_dir, patient_id_str)
        embeddings_path = os.path.join(patient_folder_path, f"{patient_id_str}.npy")

        # Check if the file exists
        if not os.path.isfile(embeddings_path):
            logger.warning(f"Embeddings file not found at {embeddings_path}. Skipping PatientID: {patient_id_str}.")
            continue
            
        # 3. Load the embeddings
        try:
            embeddings_array = np.load(embeddings_path)
        except Exception as e:
            logger.error(f"Error loading embeddings for PatientID {patient_id_str}: {e}")
            continue
        
        # 4. Create a DataFrame for the embeddings with the vector as one column
        df_embeddings = pd.DataFrame({
            # The 'emb_id' column matches the row index of the NPY array
            'emb_id': range(len(embeddings_array)),
            # Store the full vector/array for the new single column "embedding"
            EMBEDDING_COL: list(embeddings_array) # Convert array rows to a list for DataFrame storage
        })
        df_embeddings['PatientID'] = patient_id

        # 5. Collect the embedding data
        all_embedding_data.append(df_embeddings)

    # 6. Concatenate all embedding data into a single DataFrame
    if not all_embedding_data:
        logger.warning("No embeddings were successfully loaded.")
        return df_metadata # Return original metadata if no embeddings found
        
    df_all_embeddings = pd.concat(all_embedding_data, ignore_index=True)

    # 7. Final Merge with Metadata 
    # Merge the original metadata with the new, compiled embedding DataFrame
    df_final = pd.merge(
        df_metadata,
        df_all_embeddings,
        on=['PatientID', 'emb_id'],
        how='left' # Use 'left' merge to keep all metadata rows
    )
    
    logger.info(f"Final merge complete. Added '{EMBEDDING_COL}' column to metadata.")
    return df_final
# %%
print(metadata_file)
metadata = pd.read_csv(os.path.join(embedding_path, metadata_file))

metadata = merge_embeddings_and_metadata2(embedding_path, metadata)
metadata
# %%
# %%

embedding_path = "/opt/data/commonfilesharePHI/jnchiang/projects/OptumCKD/ckd_embedding_full_v3_icd_stage_filter"
metadata_file = "meta_v3_all.csv" 
metadata_full = pd.read_csv(os.path.join(embedding_path, metadata_file), sep='$')
metadata_full.head()
# subset
# %%