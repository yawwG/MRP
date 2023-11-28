import os
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
import datetime
import scipy
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import zoom
import SimpleITK as sitk
from datetime import datetime, timedelta
from monai.transforms import (
    AddChanneld,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
)
class ImageBaseDataset(Dataset):
    def __init__(
        self,
        cfg,
        split="train",
        transform=None,
    ):

        self.cfg = cfg
        self.transform = transform
        self.masktransform = transforms.Compose(
            [transforms.ToTensor()])
        self.split = split

        self.size = [125, 360, 360]

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class INBImageDataset(ImageBaseDataset):
    def __init__(self, cfg, args, split="train", transform=None, img_type="Frontal"):

        self.cfg = cfg
        self.df = pd.read_csv(cfg.csv)
        self.m = torch.nn.Upsample(size=(125, 120, 120), mode='trilinear')
        try:
            if args.cross_validation:
                split = "valid"
                print('cross validation fold: ', str(args.fold))
                valid_aidlist = pd.read_csv('/splitcsv/'+'fold_'+str(args.fold)+'.csv')
                self.df =  self.df.loc[~(self.df['aid'] in np.array(valid_aidlist['aid']))]
        except:
            print('')
        
        self.df = self.df.loc[(self.df['split'] == split) & (self.df['multitumor_black'] == 'no')]
        # sample data
        if cfg.data.frac != 1 and split == "train":
             self.df = self.df.sample(frac=cfg.data.frac)
        self.df = self.df.fillna(0)
        self.clip_min = 0
        self.clip_max = 4000
        # replace uncertains
        uncertain_mask = {k: -1 for k in INB_COMPETITION_TASKS}
        # Left(L), Right(R), Posterior(P), Anterior(A), Inferior(I), Superior(S).
        self.orientation = Orientationd(keys=["i1", "i2"], axcodes="LPS")
        self.rand_affine = RandAffined(
                            keys=["i1", "i2"],
                            mode=("bilinear", "bilinear"),
                            prob=0.5,
                            # spatial_size=(125,260,260),
                            spatial_size=(125, 120, 120),
                            translate_range=(2,15,15),
                            rotate_range=(np.pi / 180, np.pi / 4, np.pi / 4),
                            # scale_range=(0.10, 0.10, 0.10),
                            padding_mode="border",
                        )
        self.imgpath = '/imgpath/'

        super(INBImageDataset, self).__init__(cfg, split, transform)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = str(row["aid"])
        date1 = str(int(row['time1']))
        date2 = str(int(row['time2']))
        y = int(row["pcr"])

        mri_img_path = img_path

        x_idx1 = self.imgpath + mri_img_path + "_" + date2 + "_sinwas.nii.gz"  # sip
        try:
            img1 = load_nii(x_idx1)
        except:
            img1 = np.zeros((1, 125, 360, 360))
            print(x_idx1,"not exist!")

        img1 = (np.clip(img1, self.clip_min, self.clip_max) - self.clip_min) / (self.clip_max - self.clip_min)
        img1 = np.expand_dims(img1, axis=0)
        x_idx2 = self.imgpath  + mri_img_path + "_" + date2 + "_sinwas.nii.gz"

        try:
            img2 = load_nii(x_idx2)
        except:
            img2 = np.zeros((1, 125, 360, 360))
            print(x_idx2, "not exist!")
        img2 = (np.clip(img2,  self.clip_min,  self.clip_max) -  self.clip_min) / ( self.clip_max -  self.clip_min)

        img2 = np.expand_dims(img2, axis=0)
        if img1.shape != (1, 125, 360, 360):
            print(mri_img_path + "_" + date1 + "_sinwas.nii.gz date1 shape error! " + str(img1.shape))
            img1 = np.zeros((1, 125, 360, 360))

        if img2.shape != (1, 125, 360, 360):
            print(mri_img_path + "_" + date2 + "_sinwas.nii.gz date1 shape error! " + str(img2.shape))
            img2 = np.zeros((1, 125, 360, 360))

#input
        c_stage = int(row["clinical_stage1"])
        morph = int(row["Morphology_code"])
        tumor_behavior = int(row["Tumor_behaviour_code"])
        tumor_type = int(row["tumor_type"])
        N = int(row["TRTU_Klinische_N_code"])
        M = int(row["TRTU_Klinische_M_code"])
        ER = int(row["ER_code1"])
        PR = int(row["PR_code1"])
        HER2 = int(row["HER2_code1"])
        ki67 = int(row["ki_67"])
        molecular = int(row["molecular_type_code"])
        radiotherapie = int(row["neo_radiotherapie"])
        chemotherapie = int(row["neo_chemotherapie"])
        immunotherapie = int(row["neo_immunotherapie"])
        hormon = int(row["neo_hormone"])
        image_state1 = int(row["image_state_1"])
        image_state2 = int(row["image_state_2"])
        # image_state1 = -1
        # image_state2 = -1
#predict
        location = int(row["Location_code1"])
        lateral = int(row["Lateral_code"])
        clinical_size = int(row["clinical_size"])
        multifocal = int(row["multifocal_code"])
        insitu = int(row["insitu_code"])
        differentiation = int(row["Differentiation_code"])
        T = int(row["TRTU_Klinische_T_code"])

        post_T = int(row["TRTU_Post_chir_T_code"])
        post_N = int(row["TRTU_Post_chir_N_code"])

#personal info
        age = int(row["age"])
        sex = int(row["sex"])
        weight = int(row["weight"])
        brac1 = int(row["brac1"])
        brac2 = int(row["brac2"])
        chek2 = int(row["chek2"])
        tq53 = int(row["tp53"])
        menarche = int(row["menarche"])
        menopause = int(row["menopause"])

        # time_interval = int(row["days_between_t1_t2"])
        x1 = self.m(torch.unsqueeze(torch.from_numpy(img1), dim=0))
        x2 = self.m(torch.unsqueeze(torch.from_numpy(img2),dim=0))
        if self.split == 'train':
            if self.cfg.data.transform:
                data_dicts = {"i1": x1[0], "i2": x2[0]}
                data_dicts = self.orientation(data_dicts)
                affined_data_dict = self.rand_affine(data_dicts)
                x1, x2 = affined_data_dict["i1"], affined_data_dict["i2"]
                x1 = torch.unsqueeze(x1, dim=0)
                x2 = torch.unsqueeze(x2, dim=0)

        ignore_clinicalsize = 1
        ignore_T = 1
        ignore_postT = 1
        ignore_postN = 1
        ignore_N = 1
        ignore_M = 1
        ignore_tumortype = 1
        ignore_age = 1
        ignore_weight = 1
        ignore_menarche = 1

        if self.cfg.mip:
            x_ = torch.cat((x1[0].half().float(), x2[0].half().float()), dim=0)

            return x_, y, x_idx1, x_idx2, image_state1, image_state2, c_stage, morph, tumor_behavior, tumor_type, N, M, molecular, ER, PR, HER2, ki67, radiotherapie, chemotherapie, immunotherapie, hormon, location, lateral, clinical_size, multifocal, insitu, differentiation, T, ignore_clinicalsize, ignore_T, post_T, post_N, age, weight, sex, brac1,brac2,chek2,tq53, menarche, menopause, ignore_N, ignore_M, ignore_postT, ignore_postN, ignore_tumortype, ignore_age, ignore_weight, ignore_menarche
        else:
            return x1[0].half().float(), y, x_idx1, x_idx2, image_state1, image_state2, c_stage, morph, tumor_behavior, tumor_type, N, M, molecular, ER, PR, HER2, ki67, radiotherapie, chemotherapie, immunotherapie, hormon, location, lateral, clinical_size, multifocal, insitu, differentiation, T, ignore_clinicalsize, ignore_T, post_T, post_N, age, weight, sex, brac1,brac2,chek2,tq53,menarche,menopause, ignore_N, ignore_M, ignore_postT, ignore_postN, ignore_tumortype, ignore_age, ignore_weight, ignore_menarche


    def __len__(self):
        return len(self.df)

def load_nii(nii_file):
    itk_img = sitk.ReadImage(nii_file)
    img = sitk.GetArrayFromImage(itk_img)

    return img

def norm(arr):
    """ Set level as intensity of background, and set window as (99%-1%)*2
    arr: [d,w,h]
    """
    level = scipy.stats.mode(arr[:, :5, :].reshape((-1)))[0][0]
    amax = np.percentile(arr, 99)
    amin = np.percentile(arr, 1)
    window = (amax - amin) * 2
    arr = np.clip(arr, level - window / 2, level + window / 2)
    arr = (arr - (level - window / 2)) / window * 2 - 1
    return arr