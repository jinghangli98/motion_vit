import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pdb
from natsort import natsorted
import glob
from tqdm import tqdm

def generateDataset(df_path, tse_path):
    tses = natsorted(glob.glob(tse_path))
    
    df = pd.read_csv(df_path, delimiter='\t')
    
    for idx, tse in enumerate(tqdm(tses)):    
        study = participant_id = tse.split('/')[5]
        participant_id = tse.split('/')[6]
        session_id = tse.split('/')[7]
        tse_arr = nib.load(tse).get_fdata()
        rating = df[(df['participant_id']==participant_id) & (df['session_id']==session_id)]['tse_rating'].item()
        
        torch.save({'tse':tse_arr, 'rating':rating}, f'data/{study}_{participant_id}_{session_id}.pt')
    
df_path = '/ix1/tibrahim/jil202/studies_BIDS/WPC-7317/derivatives/report/study_report.tsv'
tse_path = '/ix1/tibrahim/jil202/studies_BIDS/WPC-7317/sub-*/ses-*/anat/*hipp_T2w.nii.gz'
generateDataset(df_path, tse_path)
    
# /ix1/tibrahim/jil202/studies_BIDS/GIA_IBR/sub-*/ses-*/anat/*hipp_T2w.nii.gz 186
# /ix1/tibrahim/jil202/studies_BIDS/MAN-IBR/sub-*/ses-*/anat/*hipp_T2w.nii.gz 192
# /ix1/tibrahim/jil202/studies_BIDS/MAR-IBR/sub-*/ses-*/anat/*hipp_T2w.nii.gz 177
# /ix1/tibrahim/jil202/studies_BIDS/WPC-7317/sub-*/ses-*/anat/*hipp_T1w.nii.gz 141
# /ix1/tibrahim/jil202/studies_BIDS/WPC-8521/sub-*/ses*/anat/*hipp_T2w.nii.gz 107
# /ix1/tibrahim/jil202/studies_BIDS/WPC-7268/sub-*/ses*/anat/*hipp_T2w.nii.gz 102
