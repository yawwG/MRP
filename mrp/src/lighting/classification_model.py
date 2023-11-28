import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import copy
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score, roc_auc_score
from .. import builder
from .. import utils
from .. import src
from torchvision import transforms
import sys
import csv
from . import load_statedict
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
from pytorch_lightning.core import LightningModule
import math
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skl_metrics
from sklearn.utils import resample
from sklearn import metrics
from matplotlib.ticker import FixedFormatter
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import levene
from statistics import mean
# from tools import csvTools
from pathlib import Path
from scipy import stats
import pickle
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def matrix(predict, ground_truth):
    cm = metrics.confusion_matrix(predict, ground_truth)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    NPV = tn / (tn + fn + 1e-3)
    PPV = tp / (tp + fp + 1e-3)
    sensitivity = tp / (tp + fn + 1e-3)
    specifity = tn / (tn + fp + 1e-3)
    return NPV, PPV, sensitivity, specifity

class ClassificationModel(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg
        self.model = builder.build_img_model(cfg, True)  # non_pretrain_weight, imagenet,
        self.sigmoid = nn.Sigmoid()
        self.loss = builder.build_loss(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.mip = cfg.mip #multi-time-point input
        self.mseloss = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.best_auc = 0
        self.best_epoch = 0
        self.features_imp = {'MRI':[],
                             'therapy_stage': [], 'tumor_morphology': [],
                             'tumor_histology': [], 'cN': [], 'cM': [], 'molecular_subtype': [], 'therapy': [], 'location': [],
                             'lateral': [],  'multi-focal': [], 'in-situ': [],
                             'tumor_differentiation': [], 'cT': [],  'age': [], 'weight': [], 'sex': [], 'BRAC1': [],
                             'BRAC2': [], 'CHEK2': [], 'TP53': [], 'menarche': [], 'menopause': []
                            }
        self.std_features_imp = {'MRI':[],
                             'therapy_stage': [], 'tumor_morphology': [],
                             'tumor_histology': [], 'cN': [], 'cM': [], 'molecular_subtype': [], 'therapy': [], 'location': [],
                             'lateral': [],  'multi-focal': [], 'in-situ': [],
                             'tumor_differentiation': [], 'cT': [],  'age': [], 'weight': [], 'sex': [], 'BRAC1': [],
                             'BRAC2': [], 'CHEK2': [], 'TP53': [], 'menarche': [], 'menopause': []}

        self.columns_names = ['MRI', 'therapy_stage', 'tumor_morphology', 'tumor_histology', 'cN',
                              'cM', 'molecular_subtype', 'therapy', 'location', 'lateral',  'multi-focal',
                              'in-situ','tumor_differentiation', 'cT', 'age', 'weight', 'sex', 'BRAC1', 'BRAC2', 'CHEK2', 'TP53',
                              'menarche', 'menopause']

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def f_contribution(self,input, logits, y, y_, t1_path, t2_path, location, lateral, clinical_size, multifocal, insitu, differentiation, T, ignore_clinicalsize, ignore_T, post_T, post_N, age, weight,sex, brac1,brac2,chek2,tp53,menarche,menopause):
        error = 10 * self.loss(logits, y.view(y.size(0), 1).float())
        c1 = input[:, 1024 + 16 * 0:1024 + 16 * 1]  # image_state1
        c2 = input[:, 1024 + 16 * 1:1024 + 16 * 2]  # image_state2
        c3 = input[:, 1024 + 16 * 2:1024 + 16 * 3]  # c_stage
        c4 = input[:, 1024 + 16 * 3:1024 + 16 * 4]  # N
        c5 = input[:, 1024 + 16 * 4:1024 + 16 * 5]  # M
        c6 = input[:, 1024 + 16 * 5:1024 + 16 * 6]  # tumor_behavior
        c7 = input[:, 1024 + 16 * 6:1024 + 16 * 7]  # radiotherapie
        c8 = input[:, 1024 + 16 * 7:1024 + 16 * 8]  # chemotherapie
        c9 = input[:, 1024 + 16 * 8:1024 + 16 * 9]  # immunotherapie
        c10 = input[:, 1024 + 16 * 9:1024 + 16 * 10]  # hormon
        c11 = input[:, 1024 + 16 * 10:1024 + 16 * 11]  # tumor_type
        c12 = input[:, 1024 + 16 * 11:1024 + 16 * 12]  # morph
        c13 = input[:, 1024 + 16 * 12:1024 + 16 * 13]  # molecular

        c14 = input[:, 1024 + 16 * 13:1024 + 16 * 14]#location
        c15 = input[:, 1024 + 16 * 14:1024 + 16 * 15]#lateral
        c16 = input[:, 1024 + 16 * 15:1024 + 16 * 16]#clinical_size
        c17 = input[:, 1024 + 16 * 16:1024 + 16 * 17]#multifocal
        c18 = input[:, 1024 + 16 * 17:1024 + 16 * 18]#insitu
        c19 = input[:, 1024 + 16 * 18:1024 + 16 * 19]#differentiation
        c20 = input[:, 1024 + 16 * 19:1024 + 16 * 20]#T
        c21 = input[:, 1024 + 16 * 20:1024 + 16 * 21]#age
        c22 = input[:, 1024 + 16 * 21:1024 + 16 * 22]#weight
        c23 = input[:, 1024 + 16 * 22:1024 + 16 * 23]#sex
        c24 = input[:, 1024 + 16 * 23:1024 + 16 * 24]#brac1
        c25 = input[:, 1024 + 16 * 24:1024 + 16 * 25]#brac2
        c26 = input[:, 1024 + 16 * 25:1024 + 16 * 26]#chek2
        c27 = input[:, 1024 + 16 * 26:1024 + 16 * 27]#tp53
        c28 = input[:, 1024 + 16 * 27:1024 + 16 * 28]#menarche
        c29 = input[:, 1024 + 16 * 28:1024 + 16 * 29]#menopause
        for fea_num in range(len(self.columns_names)):
            _importance = []
            for _ in range(10):  # 100
                columns_name = self.columns_names[fea_num]
                if fea_num == 0:  # MRI c0
                    u1 = input[:, 0:1024]
                    _X = u1.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    logits_, logits2_, pred_, input_ = self.model(torch.from_numpy(_X).cuda(), y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6, c11, c4, c5, c13, c7,
                                                                  c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 1:  # timepoint c1 c2
                    _X = input[:, 1024:16 * 2 + 1024].detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, torch.split(_X, 16, dim=1)[0],
                                                                  torch.split(_X, 16, dim=1)[1], c3, c12, c6, c11, c4,
                                                                  c5, c13, c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                # if fea_num == 2:  # c_stage c3 nonnead
                #     _X = c3.detach().cpu().numpy()
                #     _X = np.random.permutation(_X)
                #     _X = torch.from_numpy(_X).cuda()
                #     logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                #                                                   t2_path, c1, c2, _X, c12, c6, c11, c4, c5, c13, c7,
                #                                                   c8, c9, c10,
                #                                                   c14, c15, c16, c17, c18, c19, c20,
                #                                                   ignore_clinicalsize, ignore_T, post_T, post_N,
                #                                                   c21, c22, c23, c24, c25, c26, c27, c28, c29,plot=True)
                if fea_num == 2:  # morph c12
                    _X = c12.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, _X, c6, c11, c4, c5, c13, c7, c8,
                                                                  c9, c10,
                                                                  c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,plot=True)
                # if fea_num == 4:  # tumor_behavior c6 nonnead
                #     _X = c6.detach().cpu().numpy()
                #     _X = np.random.permutation(_X)
                #     _X = torch.from_numpy(_X).cuda()
                #     logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                #                                                   t2_path, c1, c2, c3, c12, _X,
                #                                                   c11, c4, c5, c13, c7, c8, c9, c10,
                #                                                   c14, c15, c16, c17, c18, c19, c20,
                #                                                   ignore_clinicalsize, ignore_T, post_T, post_N,
                #                                                   c21, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 3:  # tumor_type c11

                    _X = c11.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  _X, c4, c5, c13, c7, c8, c9, c10,
                                                                  c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 4:  # cN c4
                    _X = c4.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, _X, c5, c13, c7, c8, c9, c10,
                                                                  c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,plot=True)
                if fea_num == 5:  # cM c5
                    _X = c5.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, _X, c13, c7, c8, c9, c10,
                                                                  c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 6:  # molecular c13
                    _X = c13.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, _X, c7, c8, c9, c10,
                                                                  c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 7:  # therapy c7,8,9,10
                    _X = input[:, 16 * 6 + 1024:16 * 10 + 1024].detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  torch.split(_X, 16, dim=1)[0],
                                                                  torch.split(_X, 16, dim=1)[1],
                                                                  torch.split(_X, 16, dim=1)[2],
                                                                  torch.split(_X, 16, dim=1)[3],
                                                                  c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,plot=True)
                if fea_num == 8:  # location c14
                    _X = c14.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7,c8,c9,c10, _X, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 9:  # lateral c15
                    _X = c15.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7,c8,c9,c10, c14, _X, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                # if fea_num == 12:  # clinical_size c16 nonnead
                #     _X = c16.detach().cpu().numpy()
                #     _X = np.random.permutation(_X)
                #     _X = torch.from_numpy(_X).cuda()
                #     logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                #                                                   t2_path, c1, c2, c3, c12, c6,
                #                                                   c11, c4, c5, c13,
                #                                                   c7, c8, c9, c10, c14, c15, _X, c17, c18, c19, c20,
                #                                                   ignore_clinicalsize, ignore_T, post_T, post_N,
                #                                                   c21, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 10:  # multifocal c17
                    _X = c17.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, _X, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 11:  # insitu c18
                    _X = c18.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, _X, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 12:  # differentiation c19
                    _X = c19.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, _X, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 13:  # T c20
                    _X = c20.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, _X,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 14:  # age c21
                    _X = c21.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  _X, c22, c23, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 15:  # weight c22
                    _X = c22.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, _X, c23, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 16:  # sex c23
                    _X = c23.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, _X, c24, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 17:  # brac1 c24
                    _X = c24.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, _X, c25, c26, c27, c28, c29, plot=True)
                if fea_num == 18:  # brac2 c25
                    _X = c25.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, _X, c26, c27, c28, c29, plot=True)
                if fea_num == 19:  # chek2 c26
                    _X = c26.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, _X, c27, c28, c29, plot=True)
                if fea_num == 20:  # tp53 c27
                    _X = c27.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, _X, c28, c29, plot=True)
                if fea_num == 21:  # mernache c28
                    _X = c28.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, _X, c29, plot=True)
                if fea_num == 22:  # menapause c29
                    _X = c29.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:1024], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, _X, plot=True)
                loss_feature = 10 * self.loss(logits_, y.view(y.size(0), 1).float())
                _importance.append(loss_feature - error)

            self.features_imp[columns_name].append(torch.mean(torch.stack(_importance)).detach().cpu().numpy())
            self.std_features_imp[columns_name].append(torch.std(torch.stack(_importance)).detach().cpu().numpy())

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val", batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test", batch_idx)

    def training_epoch_end(self, training_step_outputs):
        return self.shared_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        return self.shared_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        return self.shared_epoch_end(test_step_outputs, "test")

    def shared_step(self, batch, split, batch_idx):
        """Similar to traning step"""

        x, y_, t1_path, t2_path, image_state1, image_state2, c_stage, morph, tumor_behavior, tumor_type, N, M, molecular,ER, PR, HER2, ki67, radiotherapie, chemotherapie, immunotherapie, hormon, location, lateral, clinical_size, multifocal, insitu, differentiation, T, ignore_clinicalsize, ignore_T, post_T, post_N, age, weight, sex, brac1, brac2, chek2, tp53, menarche, menopause, ignore_N, ignore_M, ignore_postT, ignore_postN, ignore_tumortype, ignore_age, ignore_weight, ignore_menarche  = batch
        image_state1 = image_state1.view(image_state1.size(0), 1)
        y = y_
        logits, logits2, pred, input = self.model(x, y_, t1_path, t2_path, image_state1, image_state2, c_stage, morph, tumor_behavior, tumor_type, N, M, molecular, ER, PR, HER2, ki67, radiotherapie, chemotherapie, immunotherapie, hormon, location, lateral, clinical_size, multifocal, insitu, differentiation, T, ignore_clinicalsize, ignore_T, post_T, post_N, age, weight,sex, brac1,brac2,chek2,tp53,menarche,menopause)

        if self.cfg.clinical == 'purepathological':
            loss = 10 * self.loss(logits, y.view(y.size(0), 1).float())
        elif self.cfg.clinical=='pathological':
            loss = 10 * self.loss(logits, y.view(y.size(0), 1).float())+ 10 * self.loss(logits2, y.view(y.size(0), 1).float()) + 0.0001 * self.mseloss(pred['post_T'], post_T.float()) + 0.0001 * self.mseloss(pred['post_N'], post_N.float())
        elif self.cfg.clinical=='radiological':
            loss = 10 * self.loss(logits, y.view(y.size(0), 1).float()) + 0.0001 * self.mseloss(
                pred['clinical_size'] , clinical_size.float() ) +  self.ce(
                pred['lateral'], lateral) +  self.ce(pred['location'], location) +  0.1 * self.ce(
                pred['multifocal'], multifocal) + 0.1 * self.ce(pred['insitu'], insitu)  +  0.1 * self.ce(
                pred['differentiation'], differentiation)  + 0.0001 * self.mseloss(pred['T'] ,T.float()) + 0.0001 * self.mseloss(pred['post_T'], post_T.float()) + 0.0001 * self.mseloss(pred['post_N'], post_N.float())
        elif self.cfg.clinical == 'both':
            if self.cfg.factor_predictor:
                if self.cfg.finetune:
                    loss = 10 * self.loss(logits, y.view(y.size(0), 1).float())
                else:
                    loss = 10 * self.loss(logits, y.view(y.size(0), 1).float()) \
                           + 10 * self.loss(logits2, y.view(y.size(0), 1).float()) \
                           + 0.0001 * self.mseloss(pred['clinical_size'] * ignore_clinicalsize, clinical_size.float() * ignore_clinicalsize) \
                           + self.ce(pred['lateral'], lateral.long()) + self.ce(pred['location'], location.long()) \
                           + 0.1 * self.ce(pred['multifocal'], multifocal.long()) \
                           + 0.1 * self.ce(pred['insitu'], insitu.long()) \
                           + 0.1 * self.ce(pred['differentiation'], differentiation.long()) \
                           + 0.0001 * self.mseloss(pred['T'] * ignore_postN, T.float() * ignore_postN) \
                           + 0.0001 * self.mseloss(pred['post_T'] * ignore_postN, post_T.float() * ignore_postN) \
                           + 0.0001 * self.mseloss(pred['post_N'] * ignore_postN, post_N.float() * ignore_postN) \
                           + 0.0001 * self.mseloss(pred['N'] * ignore_clinicalsize, N.float() * ignore_clinicalsize) \
                           + 0.0001 * self.mseloss(pred['M'] * ignore_M, M.float() * ignore_M) \
                           + 0.025 * self.ce(pred['radiotherapie'], radiotherapie.long()) \
                           + 0.025 * self.ce(pred['chemotherapie'], chemotherapie.long()) \
                           + 0.025 * self.ce(pred['immunotherapie'], immunotherapie.long()) \
                           + 0.025 * self.ce(pred['hormon'], hormon.long()) \
                           + 0.001 * self.mseloss(pred['tumor_type'] * ignore_M, tumor_type.float() * ignore_M) \
                           + 0.01 * self.ce(pred['morph'], morph) \
                           + 0.0001 * self.mseloss(pred['age'] * ignore_M, age.float() * ignore_M) \
                           + 0.0001 * self.mseloss(pred['weight'] * ignore_weight, weight.float() * ignore_weight) \
                           + 0.0001 * self.ce(pred['sex'], sex.long()) \
                           + 0.01 * self.ce(pred['brac1'], brac1.long()) \
                           + 0.01 * self.ce(pred['brac2'], brac2.long()) \
                           + 0.01 * self.ce(pred['chek2'], chek2.long()) \
                           + 0.01 * self.ce(pred['tp53'], tp53.long()) \
                           + 0.0001 * self.mseloss(pred['menarche'] * ignore_weight, menarche.float() * ignore_weight) \
                           + 0.01 * self.ce(pred['menopause'], menopause.long())  \
                           + self.ce(pred['molecular'], molecular.long())
                           # + self.ce(pred['ER'], ER.long()) \
                           # + self.ce(pred['PR'], PR.long()) \
                           # + self.ce(pred['HER2'], HER2.long()) \
                           # + self.ce(pred['ki67'], ki67.long()) \
            else:
                loss = 10 * self.loss(logits, y.view(y.size(0), 1).float()) \
                       + 10 * self.loss(logits2, y.view(y.size(0), 1).float())

            if split!='train' and self.cfg.test.batch_size != 1:

                self.f_contribution(input, logits, y, y_, t1_path, t2_path, location, lateral, clinical_size, multifocal, insitu, differentiation, T, ignore_clinicalsize, ignore_T, post_T, post_N, age, weight,sex, brac1,brac2,chek2,tp53,menarche,menopause )

        else:#noncli

            loss = 10 * self.loss(logits, y.view(y.size(0), 1).float())

        log_iter_loss = True

        self.log(
            f"{split}_loss",
            loss.item(),  # loss
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )
        return_dict = {"loss": loss.view(1, 1), "logit": logits, "y": y.float(), 'logit2': logits2}
        return return_dict

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def shared_epoch_end(self, step_outputs, split):
        if split != 'train' and self.cfg.test.batch_size != 1:
            feat_imp_mean = pd.Series(self.features_imp)
            feat_imp_std = pd.Series(self.std_features_imp)
            for show_num in range(len(self.columns_names)):
                feat_imp_mean[self.columns_names[show_num]] = np.array(
                    np.mean(feat_imp_mean[self.columns_names[show_num]]))
                feat_imp_std[self.columns_names[show_num]] = np.array(
                    np.mean(feat_imp_std[self.columns_names[show_num]]))
            order = feat_imp_mean.sort_values(ascending=False).index
            feat_imp_mean = feat_imp_mean[order]
            feat_imp_std = feat_imp_std[order]
            feat_imp_mean = feat_imp_mean.astype(float)
            feat_imp_std = feat_imp_std.astype(float)
            concat_mean_std = pd.concat((feat_imp_mean, feat_imp_std), axis=1)
            concat_mean_std.colums = ['mean', 'std']
            concat_mean_std.to_csv(self.cfg.output_dir + '/feat_importance_mean_std.csv', header=0)
            fig, axes = plt.subplots(figsize=(5, 8), constrained_layout=True)
            plt.xlim(-0.5, 2)
            fig = utils.plot_coefficients(self.cfg, fig, axes, concat_mean_std, title='Feature Importance (iMRrhpc)')
            fig.savefig(os.path.join(self.cfg.output_dir, 'Feature-Importance-(iMRrhpc).pdf'), dpi=600)
            fig.savefig(os.path.join(self.cfg.output_dir, 'Feature-Importance-(iMRrhpc).png'), dpi=600)

        logit = torch.cat([x["logit"] for x in step_outputs])
        print('\n')
        for key in step_outputs[0]:
            if 'loss' in key:
                if key=='loss':
                    step_outputs_ = torch.cat([x[key] for x in step_outputs]).detach().cpu().numpy()
                    print(key, ": ", np.mean(step_outputs_), '; percentage:',1)
                else:
                    step_outputs_ = [x[key] for x in step_outputs]
                    print(key, ": ", np.mean(step_outputs_), '; percentage:', np.mean(step_outputs_) / np.mean(torch.cat([x['loss'] for x in step_outputs]).detach().cpu().numpy()))

        y = torch.cat([x["y"] for x in step_outputs])
        y = y.view(y.size(0), 1)
        prob  =logit

        y = y.detach().cpu().numpy()
        prob = prob.detach().cpu().numpy()
        auroc_list, auprc_list = [], []
        for i in range(y.shape[1]):
            y_cls = y[:]
            prob_cls = prob[:]

            if np.isnan(prob_cls).any():
                auprc_list.append(0)
                auroc_list.append(0)
            else:
                try:
                    auroc_list.append(roc_auc_score(y_cls, prob_cls))
                except:

                    print('problem!,prob_cls:', prob_cls)
                    auroc_list.append(0)

        auroc = np.mean(auroc_list)
        prob_ = np.where(prob > 0.5, 1, 0)
        self.log(f"{split}_auroc", auroc, on_epoch=True, logger=True, prog_bar=True)

        Sensi_mean=0
        Sensi_lower_ci=0
        Sensi_upper_ci=0
        Speci_mean=0
        Speci_lower_ci=0
        Speci_upper_ci=0
        ppv_mean=0
        ppv_lower_ci=0
        ppv_upper_ci=0
        npv_mean=0
        npv_lower_ci=0
        npv_upper_ci=0

        if split != "train":
            if (self.best_auc < auroc):
                self.best_auc = auroc
                self.best_epoch = self.current_epoch
                report = classification_report(y[:, 0], prob_[:, 0], digits=3)
                report_path = self.cfg.output_dir + split + "_report.txt"
                text_file = open(report_path, "w")
                n = text_file.write(report)
                text_file.close()
                with open(self.cfg.output_dir + split + 'pro.csv', 'w', newline='') as file:
                    mywriter = csv.writer(file, delimiter=',')
                    mywriter.writerows(self.sigmoid(prob))
                with open(self.cfg.output_dir + split + 'label.csv', 'w', newline='') as file:
                    mywriter = csv.writer(file, delimiter=',')
                    mywriter.writerows(y)

                fpr, tpr, thresholds = metrics.roc_curve(y_cls, prob_cls)
                optimal_thm, pointm = Find_Optimal_Cutoff(tpr, fpr, threshold=thresholds)
                predict = [
                    (prob_cls > optimal_thm).astype(np.float32)
                ]
                cm = metrics.confusion_matrix(y_cls, predict[0])
                tn = cm[0][0]
                fn = cm[1][0]
                tp = cm[1][1]
                fp = cm[0][1]
                NPV = tn / (tn + fn + 1e-3)
                PPV = tp / (tp + fp + 1e-3)
                recall = tp / (tp + fn + 1e-3)
                precision = tp / (tp + fp + 1e-3)
                f1 = 2 * recall * precision / (recall + precision + 1e-3)
                npv = []
                ppv = []
                Sensi = []
                Speci = []
                alpha = 100 - 95
                for i in range(1000):
                    predictions_i1_r1_C_bs, ground_truth_bs = resample(prob_cls, y_cls, replace=True)

                    predict = [
                        (predictions_i1_r1_C_bs > 0.5).astype(np.float32)
                    ]
                    NPV_, PPV_, sensitivity, specifity = matrix(predict[0], ground_truth_bs)
                    npv.append(NPV_)
                    ppv.append(PPV_)
                    Sensi.append(sensitivity)
                    Speci.append(specifity)
                npv_mean = np.mean(npv)
                ppv_mean = np.mean(ppv)
                Sensi_mean = np.mean(Sensi)
                Speci_mean = np.mean(Speci)
                npv_lower_ci = np.percentile(npv, alpha / 2)
                npv_upper_ci = np.percentile(npv, 100 - alpha / 2)
                ppv_lower_ci = np.percentile(ppv, alpha / 2)
                ppv_upper_ci = np.percentile(ppv, 100 - alpha / 2)
                Sensi_lower_ci = np.percentile(Sensi, alpha / 2)
                Sensi_upper_ci = np.percentile(Sensi, 100 - alpha / 2)
                Speci_lower_ci = np.percentile(Speci, alpha / 2)
                Speci_upper_ci = np.percentile(Speci, 100 - alpha / 2)

                print('point: 1-specifity, sensitivity', pointm,
                      ', best Sensity', pointm[1],
                      ', best Specifity', 1 - pointm[0],
                      ', best PPV', PPV,
                      ', best NPV', NPV,
                      ', auc1_i1_r1_C:', auroc,
                      ', mean sensitivity(TPR):', Sensi_mean, [Sensi_lower_ci, Sensi_upper_ci],
                      ', mean specificity(TNR):', Speci_mean, [Speci_lower_ci, Speci_upper_ci],
                      ', mean PPV:', ppv_mean, [ppv_lower_ci, ppv_upper_ci],
                      ', mean NPV:', npv_mean, [npv_lower_ci, npv_upper_ci])

            results = {"epoch": self.current_epoch, "auroc": auroc, "bestauc": self.best_auc,'TPR': Sensi_mean, 'Sensi_lower_ci': Sensi_lower_ci, "Sensi_upper_ci": Sensi_upper_ci,
                      'TNR': Speci_mean, "Speci_lower_ci": Speci_lower_ci, "Speci_upper_ci": Speci_upper_ci,
                      'PPV': ppv_mean, "ppv_lower_ci": ppv_lower_ci, "ppv_upper_ci": ppv_upper_ci,
                      'NPV': npv_mean, "npv_lower_ci": npv_lower_ci, "npv_upper_ci":npv_upper_ci,
                       "bestEpoch": self.best_epoch}
            results_csv = os.path.join(self.cfg.output_dir, split + "_auc.csv")
            with open(results_csv, "a") as fp:
                json.dump(results, fp)
                json.dump("/n", fp)
