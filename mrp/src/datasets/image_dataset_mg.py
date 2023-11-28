import os
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
import datetime
import torchvision.transforms.functional as ttF
from PIL import Image
from torch.utils.data import Dataset
# from albumentations import ShiftScaleRotate, Normalize, Resize, Compose
# from albumentations.pytorch import ToTensor
# from albumentations.pytorch.transforms import ToTensor
from torchvision import transforms
from datetime import datetime, timedelta
import random
class ImageBaseDataset(Dataset):
    def __init__(
        self,
        cfg,
        split="train",
        transform=None,
    ):

        self.cfg = cfg
        self.mip = self.cfg.mip
        self.transform = transform
        self.masktransform = transforms.Compose(
            [transforms.ToTensor()])
        self.split = split

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def read_from_dicom(self, img_path):
        raise NotImplementedError

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img

    def train_transform(self, image, mask):
        if random.random() > 0.5:
            image = ttF.hflip(image)
            mask = ttF.hflip(mask)

        if random.random() > 0.5:
            image = ttF.vflip(image)
            mask = ttF.vflip(mask)

        return image, mask

class INBImageDataset(ImageBaseDataset):
    def __init__(self, cfg, args, split="train", transform=None, img_type="Frontal"):
        self.cfg = cfg
        self.df = pd.read_csv(cfg.csv)
        try:
            if split == "valid":
                split = "valid"
        except:
            split = 'test'

        self.df = self.df.loc[(self.df['split'] == split)]

        if cfg.data.frac != 1 and split == "train":
            self.df = self.df.sample(frac=cfg.data.frac)
        self.df = self.df.fillna(0)
        self.imgpath = '/'
        super(INBImageDataset, self).__init__(cfg, split, transform)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = str(row["aid"])
        date1 = str(int(row['time1']))
        date2 = str(int(row['time2']))

        y = int(row["pcr"])
        lcc = img_path + '_' + date1 + "_L_CC.png"
        lmlo = img_path + '_' + date1 + "_L_MLO.png"
        rcc = img_path + '_' + date1 + "_R_CC.png"
        rmlo = img_path + '_' + date1 + "_R_MLO.png"

        image_path = []
        image_path.append(lcc)
        image_path.append(lmlo)
        image_path.append(rcc)
        image_path.append(rmlo)

        mask_path = []
        lcc2 = img_path + '_' + date2 + '_L_CC.png'
        lmlo2 = img_path + '_' + date2 + '_L_MLO.png'
        rcc2 = img_path + '_' + date2 + '_R_CC.png'
        rmlo2 = img_path + '_' + date2 + '_R_MLO.png'

        mask_path.append(lcc2)
        mask_path.append(lmlo2)
        mask_path.append(rcc2)
        mask_path.append(rmlo2)
        mask = []
        for i in range(len(mask_path)):
            if (os.path.exists(mask_path[i])):
                try:
                    mask_tmp = cv2.imread(mask_path[i], 0)
                    mask_tmp = self._resize_img(mask_tmp, self.cfg.data.image.imsize)
                    mask_tmp = Image.fromarray(mask_tmp).convert("RGB")
                except:
                    mask_tmp = Image.new('L', (self.cfg.data.image.imsize, self.cfg.data.image.imsize), (0)).convert("RGB")
                    print(img_path + '_' + date2 + 't2 error download!')
            else:
                mask_tmp = Image.new('L', (self.cfg.data.image.imsize, self.cfg.data.image.imsize), (0)).convert("RGB")

            mask_tmp = self.transform(mask_tmp)
            mask.append(mask_tmp)
        x2 = torch.stack(mask)

        imag1 = []
        for i in range(len(image_path)):
            if (os.path.exists(image_path[i])):
                try:
                    mask_tmp = cv2.imread(image_path[i], 0)
                    mask_tmp = self._resize_img(mask_tmp, self.cfg.data.image.imsize)
                    mask_tmp = Image.fromarray(mask_tmp).convert("RGB")
                except:
                    mask_tmp = Image.new('L', (self.cfg.data.image.imsize, self.cfg.data.image.imsize), (0)).convert("RGB")
                    print(img_path + '_' + date1 + 't1 error download!')
            else:
                mask_tmp = Image.new('L', (self.cfg.data.image.imsize, self.cfg.data.image.imsize), (0)).convert("RGB")
            mask_tmp = self.transform(mask_tmp)
            imag1.append(mask_tmp)
        x1 = torch.stack(imag1)
        if self.split == 'train':
            x1,x2 = self.train_transform(x1,x2)
        # input
        c_stage = int(row["clinical_stage"])
        morph = int(row["Morphology_code"])
        tumor_behavior = int(row["Tumor_behaviour_code"])
        tumor_type = int(row["PA_diagnosi_code"])
        N = int(row["TRTU_Klinische_N_code"])
        M = int(row["TRTU_Klinische_M_code"])
        molecular = int(row["molecular_type_code"])
        radiotherapie = int(row["neo_radiotherapie"])
        chemotherapie = int(row["neo_chemotherapie"])
        immunotherapie = int(row["neo_immunotherapie"])
        hormon = int(row["neo_hormone"])
        image_state1 = int(row["image_state1"])
        image_state2 = int(row["image_state2"])
        # predict
        birads1 = int(row["birads1"])
        birads2 = int(row["birads2"])
        density1 = int(row["density1"])
        density2 = int(row["density2"])
        location = int(row["Location_code"])
        lateral = int(row["Lateral_code"])
        clinical_size = int(row["clinical_size"])
        multifocal = int(row["multifocal_code"])
        insitu = int(row["insitu_code"])
        differentiation = int(row["Differentiation_code"])
        T = int(row["TRTU_Klinische_T_code"])

        post_T = int(row["TRTU_Post_chir_T_code"])
        post_N = int(row["TRTU_Post_chir_N_code"])

        # personal info
        age = int(row["age"])
        sex = int(row["sex"])
        weight = int(row["weight"])
        brac1 = int(row["brca1"])
        brac2 = int(row["brca2"])
        chek2 = int(row["chek2"])
        tq53 = int(row["tp53"])
        menarche = int(row["menarche"])
        menopause = int(row["menopausal"])

        ignore_clinicalsize = 1
        ignore_T = 1
        x1 = torch.split(x1,1,dim=1)[0].view(1,4,x1.size(2),x1.size(3))
        x2 = torch.split(x2, 1, dim=1)[0].view(1, 4, x2.size(2), x2.size(3))
        if self.cfg.mip:
            x_ = torch.cat((x1[0].half().float(), x2[0].half().float()), dim=0)
            return x_, y, img_path + '_' + date1, img_path + '_' + date2, image_state1, image_state2, c_stage, morph, tumor_behavior, tumor_type, N, M, molecular, radiotherapie, chemotherapie, immunotherapie, hormon, location, lateral, clinical_size, multifocal, insitu, differentiation, T, ignore_clinicalsize, ignore_T, post_T, post_N, birads1, birads2, density1, density2, age, weight,sex, brac1,brac2,chek2,tq53,menarche,menopause
        else:
            return x1[0].half().float(), y, img_path + '_' + date1, img_path + '_' + date2, image_state1, image_state2, c_stage, morph, tumor_behavior, tumor_type, N, M, molecular, radiotherapie, chemotherapie, immunotherapie, hormon, location, lateral, clinical_size, multifocal, insitu, differentiation, T, ignore_clinicalsize, ignore_T, post_T, post_N, birads1, birads2, density1, density2, age, weight,sex, brac1,brac2,chek2,tq53,menarche,menopause


    def __len__(self):
        return len(self.df)

    def ERPRHER2_subgroup_anay(self, all):

        if all[(all['TRTU_Risicofactor_ER'] == 1) & (all['TRTU_Risicofactor_PR'] == 1) & (all['TRTU_Risicofactor_HER2'] == 0)]:
            all_sub = '0'#LuminaA
        if all[(all['TRTU_Risicofactor_ER'] == 1) & (all['TRTU_Risicofactor_PR'] == 1) & (all['TRTU_Risicofactor_HER2'] == 1)]:
            all_sub = '1'#LuminaB
        if  all[
                (all['TRTU_Risicofactor_ER'] == 0) & (all['TRTU_Risicofactor_PR'] == 0) & (all['TRTU_Risicofactor_HER2'] == 1)]:
            all_sub = '2'#HER2-enriched
        if all[(all['TRTU_Risicofactor_ER'] == 0) & (all['TRTU_Risicofactor_PR'] == 0) & (all['TRTU_Risicofactor_HER2'] == 0)]:
            all_sub = '3'#TN

        return all_sub

