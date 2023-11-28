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
from .. import models
import pandas as pd
import matplotlib.pyplot as plt
import sys
import csv
import SimpleITK as sitk
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from transformers import DistilBertModel, DistilBertConfig
from pytorch_lightning.core import LightningModule
class ClassificationModel_mg(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg
        self.model = builder.build_img_model(cfg, True)
        self.best_epoch = 0
        self.sigmoid = nn.Sigmoid()
        self.loss = builder.build_loss(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.mip = cfg.mip #multi-time-point input
        self.mseloss = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.best_auc = 0
        self.features_imp = {'MG': [], 'therapy_stage': [], 'tumor_morphology': [],
                             'tumor_histology': [], 'cN': [], 'cM': [], 'molecular_subtype': [], 'therapy': [], 'location': [],
                             'lateral': [],  'multi-focal': [], 'in-situ': [],
                             'tumor_differentiation': [], 'cT': [], 'density':[], 'age': [], 'weight': [], 'sex': [], 'BRAC1': [],
                             'BRAC2': [], 'CHEK2': [], 'TP53': [], 'menarche': [], 'menopause': []}
        self.std_features_imp = {'MG': [], 'therapy_stage': [], 'tumor_morphology': [],
                                 'tumor_histology': [], 'cN': [], 'cM': [], 'molecular_subtype': [],
                                 'therapy': [], 'location': [], 'lateral': [], 'multi-focal': [],
                                 'in-situ': [], 'tumor_differentiation': [], 'cT': [],  'density':[], 'age': [], 'weight': [], 'sex': [], 'BRAC1': [],
                                 'BRAC2': [], 'CHEK2': [], 'TP53': [], 'menarche': [], 'menopause': []}
        self.columns_names = ['MG', 'therapy_stage', 'tumor_morphology', 'tumor_histology', 'cN',
                              'cM', 'molecular_subtype', 'therapy', 'location', 'lateral',  'multi-focal',
                              'in-situ','tumor_differentiation', 'cT', 'density', 'age', 'weight', 'sex', 'BRAC1', 'BRAC2', 'CHEK2', 'TP53',
                              'menarche', 'menopause']


    def f_contribution(self, input, logits, y, t1_path, t2_path, location, lateral, clinical_size, multifocal,
                       insitu, differentiation, T, ignore_clinicalsize, ignore_T, post_T, post_N, age, weight, sex,
                       brac1, brac2, chek2, tp53, menarche, menopause):
        error = 10 * self.loss(logits, y.view(y.size(0), 1).float())
        y_ = y
        c1 = input[:, 4096 + 16 * 0:4096 + 16 * 1]  # image_state1
        c2 = input[:, 4096 + 16 * 1:4096 + 16 * 2]  # image_state2
        c3 = input[:, 4096 + 16 * 2:4096 + 16 * 3]  # c_stage
        c4 = input[:, 4096 + 16 * 3:4096 + 16 * 4]  # N
        c5 = input[:, 4096 + 16 * 4:4096 + 16 * 5]  # M
        c6 = input[:, 4096 + 16 * 5:4096 + 16 * 6]  # tumor_behavior
        c7 = input[:, 4096 + 16 * 6:4096 + 16 * 7]  # radiotherapie
        c8 = input[:, 4096 + 16 * 7:4096 + 16 * 8]  # chemotherapie
        c9 = input[:, 4096 + 16 * 8:4096 + 16 * 9]  # immunotherapie
        c10 = input[:, 4096 + 16 * 9:4096 + 16 * 10]  # hormon
        c11 = input[:, 4096 + 16 * 10:4096 + 16 * 11]  # tumor_type
        c12 = input[:, 4096 + 16 * 11:4096 + 16 * 12]  # morph
        c13 = input[:, 4096 + 16 * 12:4096 + 16 * 13]  # molecular

        c14 = input[:, 4096 + 16 * 13:4096 + 16 * 14]  # location
        c15 = input[:, 4096 + 16 * 14:4096 + 16 * 15]  # lateral
        c16 = input[:, 4096 + 16 * 15:4096 + 16 * 16]  # clinical_size
        c17 = input[:, 4096 + 16 * 16:4096 + 16 * 17]  # multifocal
        c18 = input[:, 4096 + 16 * 17:4096 + 16 * 18]  # insitu
        c19 = input[:, 4096 + 16 * 18:4096 + 16 * 19]  # differentiation
        c20 = input[:, 4096 + 16 * 19:4096 + 16 * 20]  # T

        c21 = input[:, 4096 + 16 * 20:4096 + 16 * 21]  # birads1
        c22 = input[:, 4096 + 16 * 21:4096 + 16 * 22]  # birads2
        c23 = input[:, 4096 + 16 * 22:4096 + 16 * 23]  # density1
        c24 = input[:, 4096 + 16 * 23:4096 + 16 * 24]  # density2

        c25 = input[:, 4096 + 16 * 24:4096 + 16 * 25]  # age
        c26 = input[:, 4096 + 16 * 25:4096 + 16 * 26]  # weight
        c27 = input[:, 4096 + 16 * 26:4096 + 16 * 27]  # sex
        c28 = input[:, 4096 + 16 * 27:4096 + 16 * 28]  # brac1
        c29 = input[:, 4096 + 16 * 28:4096 + 16 * 29]  # brac2
        c30 = input[:, 4096 + 16 * 29:4096 + 16 * 30]  # chek2
        c31 = input[:, 4096 + 16 * 30:4096 + 16 * 31]  # tp53
        c32 = input[:, 4096 + 16 * 31:4096 + 16 * 32]  # menarche
        c33 = input[:, 4096 + 16 * 32:4096 + 16 * 33]  # menopause
        for fea_num in range(len(self.columns_names)):
            _importance = []
            for _ in range(10):  # 100
                columns_name = self.columns_names[fea_num]
                if fea_num == 0:  # MG c0
                    u1 = input[:, 0:4096]
                    _X = u1.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    logits_, logits2_, pred_, input_ = self.model(torch.from_numpy(_X).cuda(), y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6, c11, c4, c5, c13, c7,
                                                                  c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)
                if fea_num == 1:  # timepoint c1 c2
                    _X = input[:, 4096:16 * 2 + 4096].detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, torch.split(_X, 16, dim=1)[0],
                                                                  torch.split(_X, 16, dim=1)[1], c3, c12, c6, c11, c4,
                                                                  c5, c13, c7, c8, c9, c10, c14, c15, c16, c17, c18,
                                                                  c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)
                # if fea_num == 2:  # c_stage c3
                #     _X = c3.detach().cpu().numpy()
                #     _X = np.random.permutation(_X)
                #     _X = torch.from_numpy(_X).cuda()
                #     logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                #                                                   t2_path, c1, c2, _X, c12, c6, c11, c4, c5, c13, c7,
                #                                                   c8, c9, c10,
                #                                                   c14, c15, c16, c17, c18, c19, c20,
                #                                                   ignore_clinicalsize, ignore_T, post_T, post_N,
                #                                                   c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                #                                                   plot=True)
                if fea_num == 2:  # morph c12
                    _X = c12.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, _X, c6, c11, c4, c5, c13, c7, c8,
                                                                  c9, c10,
                                                                  c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)
                # if fea_num == 4:  # tumor_behavior c6
                #     _X = c6.detach().cpu().numpy()
                #     _X = np.random.permutation(_X)
                #     _X = torch.from_numpy(_X).cuda()
                #     logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                #                                                   t2_path, c1, c2, c3, c12, _X,
                #                                                   c11, c4, c5, c13, c7, c8, c9, c10,
                #                                                   c14, c15, c16, c17, c18, c19, c20,
                #                                                   ignore_clinicalsize, ignore_T, post_T, post_N,
                #                                                   c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                #                                                   plot=True)
                if fea_num == 3:  # tumor_type c11

                    _X = c11.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  _X, c4, c5, c13, c7, c8, c9, c10,
                                                                  c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)
                if fea_num == 4:  # cN c4
                    _X = c4.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, _X, c5, c13, c7, c8, c9, c10,
                                                                  c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)
                if fea_num == 5:  # cM c5
                    _X = c5.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, _X, c13, c7, c8, c9, c10,
                                                                  c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)
                if fea_num == 6:  # molecular c13
                    _X = c13.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, _X, c7, c8, c9, c10,
                                                                  c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)
                if fea_num == 7:  # therapy c7,8,9,10
                    _X = input[:, 16 * 6 + 4096:16 * 10 + 4096].detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  torch.split(_X, 16, dim=1)[0],
                                                                  torch.split(_X, 16, dim=1)[1],
                                                                  torch.split(_X, 16, dim=1)[2],
                                                                  torch.split(_X, 16, dim=1)[3],
                                                                  c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)

                if fea_num == 8:  # location c14
                    _X = c14.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, _X, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)
                if fea_num == 9:  # lateral c15
                    _X = c15.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, _X, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)
                # if fea_num == 12:  # clinical_size c16
                #     _X = c16.detach().cpu().numpy()
                #     _X = np.random.permutation(_X)
                #     _X = torch.from_numpy(_X).cuda()
                #     logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                #                                                   t2_path, c1, c2, c3, c12, c6,
                #                                                   c11, c4, c5, c13,
                #                                                   c7, c8, c9, c10, c14, c15, _X, c17, c18, c19, c20,
                #                                                   ignore_clinicalsize, ignore_T, post_T, post_N,
                #                                                   c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                #                                                   plot=True)
                if fea_num == 10:  # multifocal c17
                    _X = c17.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, _X, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)
                if fea_num == 11:  # insitu c18
                    _X = c18.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, _X, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)
                if fea_num == 12:  # differentiation c19
                    _X = c19.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, _X, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)
                if fea_num == 13:  # T c20
                    _X = c20.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, _X,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33,
                                                                  plot=True)
                # if fea_num == 14:  # birads1&2 c21 22
                #     _X = torch.cat((c21,c22), dim=1).detach().cpu().numpy()
                #     _X = np.random.permutation(_X)
                #     _X = torch.from_numpy(_X).cuda()
                #     logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                #                                                   t2_path, c1, c2, c3, c12, c6,
                #                                                   c11, c4, c5, c13,
                #                                                   c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                #                                                   ignore_clinicalsize, ignore_T, post_T, post_N,
                #                                                   torch.split(_X, 16, dim=1)[0], torch.split(_X, 16, dim=1)[1], c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,c33, plot=True)
                if fea_num == 14:  # density1&2 c23 24
                    _X = torch.cat((c23,c24), dim=1).detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, torch.split(_X, 16, dim=1)[0], torch.split(_X, 16, dim=1)[1], c25, c26, c27, c28, c29,c30,c31,c32,c33, plot=True)

                if fea_num == 15:  # age c25
                    _X = c25.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, _X, c26, c27, c28, c29,c30,c31,c32,c33, plot=True)

                if fea_num == 16:  # weight c26
                    _X = c26.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, _X, c27, c28, c29,c30,c31,c32,c33, plot=True)
                if fea_num == 17:  # sex c27
                    _X = c27.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, _X, c28, c29,c30,c31,c32,c33, plot=True)
                if fea_num == 18:  # brac1 c28
                    _X = c28.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, _X, c29,c30,c31,c32,c33, plot=True)
                if fea_num == 19:  # brac2 c29
                    _X = c29.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, _X, c30, c31,
                                                                  c32, c33, plot=True)
                if fea_num == 20:  # chek2 c30
                    _X = c30.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,_X,c31,c32,c33, plot=True)
                if fea_num == 21:  # tp53 c31
                    _X = c31.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,_X,c32,c33, plot=True)
                if fea_num == 22:  # mernache c32
                    _X = c32.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,_X,c33, plot=True)
                if fea_num == 23:  # menapause c33
                    _X = c33.detach().cpu().numpy()
                    _X = np.random.permutation(_X)
                    _X = torch.from_numpy(_X).cuda()
                    logits_, logits2_, pred_, input_ = self.model(input[:, 0:4096], y_, t1_path,
                                                                  t2_path, c1, c2, c3, c12, c6,
                                                                  c11, c4, c5, c13,
                                                                  c7, c8, c9, c10, c14, c15, c16, c17, c18, c19, c20,
                                                                  ignore_clinicalsize, ignore_T, post_T, post_N,
                                                                  c21, c22, c23, c24, c25, c26, c27, c28, c29,c30,c31,c32,_X, plot=True)
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

        x, y, t1_path, t2_path, image_state1, image_state2, c_stage, morph, tumor_behavior, tumor_type, N, M, molecular, radiotherapie, chemotherapie, immunotherapie, hormon, location, lateral, clinical_size, multifocal, insitu, differentiation, T, ignore_clinicalsize, ignore_T, post_T, post_N, birads1, birads2, density1, density2, age, weight,sex, brac1,brac2,chek2,tp53,menarche,menopause = batch
        image_state1 = image_state1.view(image_state1.size(0), 1)
        logits,logits2, pred, input = self.model(x, y, t1_path, t2_path, image_state1, image_state2, c_stage, morph, tumor_behavior,
                                  tumor_type, N, M, molecular, radiotherapie, chemotherapie, immunotherapie, hormon,
                                  location, lateral, clinical_size, multifocal, insitu, differentiation, T,
                                  ignore_clinicalsize, ignore_T, post_T, post_N, birads1, birads2, density1, density2, age, weight,sex, brac1,brac2,chek2,tp53,menarche,menopause)

        if self.cfg.clinical=='pathological':
            loss = 10 * self.loss(logits, y.view(y.size(0), 1).float())+ 10 * self.loss(logits2, y.view(y.size(0), 1).float()) + 0.0001 * self.mseloss(pred['post_T'], post_T.float()) + 0.0001 * self.mseloss(pred['post_N'], post_N.float())
        elif self.cfg.clinical=='radiological':
            loss = 10 * self.loss(logits, y.view(y.size(0), 1).float()) + 0.0001 * self.mseloss(
                pred['clinical_size'] , clinical_size.float() ) +  self.ce(
                pred['lateral'], lateral) +  self.ce(pred['location'], location) +  0.1 * self.ce(
                pred['multifocal'], multifocal) + 0.1 * self.ce(pred['insitu'], insitu)  +  0.1 * self.ce(
                pred['differentiation'], differentiation)  +\
                   0.0001 * self.mseloss(pred['T'] ,T.float()) + \
                   0.0001 * self.mseloss(pred['post_T'], post_T.float()) + \
                   0.0001 * self.mseloss(pred['post_N'], post_N.float()) + \
                   0.01 * self.ce(pred['birads1'], birads1)+ \
                   0.01 * self.ce(pred['birads2'], birads2)+ \
                   0.0001 * self.mseloss(pred['density1'], density1.float())+ \
                   0.0001 * self.mseloss(pred['density2'], density2.float())
        elif self.cfg.clinical == 'both':
            if self.cfg.factor_predictor:
                loss = 10 * self.loss(logits, y.view(y.size(0), 1).float()) \
                       + 10 * self.loss(logits2, y.view(y.size(0), 1).float()) \
                       + 0.0001 * self.mseloss(pred['clinical_size'], clinical_size.float()) \
                       + self.ce(pred['lateral'], lateral.long()) + self.ce(pred['location'], location.long()) \
                       + 0.1 * self.ce(pred['multifocal'], multifocal.long()) \
                       + 0.1 * self.ce(pred['insitu'], insitu.long()) \
                       + 0.1 * self.ce(pred['differentiation'], differentiation.long()) \
                       + 0.0001 * self.mseloss(pred['T'], T.float()) \
                       + 0.0001 * self.mseloss(pred['post_T'], post_T.float()) \
                       + 0.0001 * self.mseloss(pred['post_N'], post_N.float()) \
                       + 0.01 * self.ce(pred['birads1'], birads1)\
                       + 0.01 * self.ce(pred['birads2'], birads2)\
                       + 0.0001 * self.mseloss(pred['density1'], density1.float())\
                       + 0.0001 * self.mseloss(pred['density2'], density2.float())\
                       + 0.0001 * self.mseloss(pred['N'], N.float()) \
                       + 0.0001 * self.mseloss(pred['M'], M.float()) \
                       + 0.025 * self.ce(pred['radiotherapie'], radiotherapie.long()) \
                       + 0.025 * self.ce(pred['chemotherapie'], chemotherapie.long()) \
                       + 0.025 * self.ce(pred['immunotherapie'], immunotherapie.long()) \
                       + 0.025 * self.ce(pred['hormon'], hormon.long()) \
                       + 0.001 * self.mseloss(pred['tumor_type'], tumor_type.float()) \
                       + 0.01 * self.ce(pred['morph'], morph) \
                       + self.ce(pred['molecular'], molecular.long()) \
                       + 0.0001 * self.mseloss(pred['age'], age.float()) \
                       + 0.0001 * self.mseloss(pred['weight'], weight.float()) \
                       + 0.0001 * self.ce(pred['sex'], sex.long()) \
                       + 0.01 * self.ce(pred['brac1'], brac1.long()) \
                       + 0.01 * self.ce(pred['brac2'], brac2.long()) \
                       + 0.01 * self.ce(pred['chek2'], chek2.long()) \
                       + 0.01 * self.ce(pred['tp53'], tp53.long()) \
                       + 0.0001 * self.mseloss(pred['menarche'], menarche.float()) \
                       + 0.01 * self.ce(pred['menopause'], menopause.long())
            else:
                loss = 10 * self.loss(logits, y.view(y.size(0), 1).float()) \
                        + 10 * self.loss(logits2, y.view(y.size(0),1).float())

            if split != 'train' and self.cfg.test.batch_size != 1:
                self.f_contribution(input, logits, y, t1_path, t2_path, location, lateral, clinical_size,
                                    multifocal, insitu, differentiation, T, ignore_clinicalsize, ignore_T, post_T,
                                    post_N, age, weight, sex, brac1, brac2, chek2, tp53, menarche, menopause)
        else:#noncli
            loss = 10 * self.loss(logits, y.view(y.size(0), 1).float())
        log_iter_loss = True

        self.log(
            f"{split}_loss",
            loss.item(),
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )
        return_dict = {"loss": loss, "logit": logits, "y": y.float()}


        return return_dict

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
            concat_mean_std.to_csv(self.cfg.output_dir + '/feat_importance_mean_std_iMGrhpc.csv', header=0)
            fig, axes = plt.subplots(figsize=(5, 8), constrained_layout=True)
            plt.xlim(-1.5, 9)
            fig = utils.plot_coefficients(self.cfg, fig, axes, concat_mean_std, title='Feature Importance (iMGrhpc)')
            fig.savefig(os.path.join(self.cfg.output_dir, 'Feature-Importance-(iMGrhpc).pdf'), dpi=600)
            fig.savefig(os.path.join(self.cfg.output_dir, 'Feature-Importance-(iMGrhpc).png'), dpi=600)

        logit = torch.cat([x["logit"] for x in step_outputs])
        # print(logit)
        y = torch.cat([x["y"] for x in step_outputs])
        y = y.view(y.size(0), 1)
        # prob = torch.sigmoid(logit)
        prob = logit

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
                    print('problem!,ycls:', y_cls)
                    print('problem!,prob_cls:', prob_cls)
                    auroc_list.append(0)

        auroc = np.mean(auroc_list)
        prob_ = np.where(prob>0.5,1,0)
        probsize = prob.shape[0]
        if split != "train":
            if (self.best_auc < auroc) and probsize > 16:
                self.best_auc = auroc
                self.best_epoch = self.current_epoch
                report = classification_report(y[:,0], prob_[:,0], digits=3)
                report_path = self.cfg.output_dir + split + "_report.txt"
                text_file = open(report_path, "w")
                n = text_file.write(report)
                text_file.close()
                with open(self.cfg.output_dir + split + '_pro.csv', 'w', newline='') as file:
                    mywriter = csv.writer(file, delimiter=',')
                    mywriter.writerows(prob)
                with open(self.cfg.output_dir + split + '_label.csv', 'w', newline='') as file:
                    mywriter = csv.writer(file, delimiter=',')
                    mywriter.writerows(y)

            results = {"epoch": self.current_epoch, "auroc": auroc, "bestauc": self.best_auc,
                       "bestEpoch": self.best_epoch}
            results_csv = os.path.join(self.cfg.output_dir, split +"_auc.csv")
            with open(results_csv, "a") as fp:
                json.dump(results, fp)
                json.dump("/n", fp)

        self.log(f"{split}_auroc", auroc, on_epoch=True, logger=True, prog_bar=True)


