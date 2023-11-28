import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as resmodels

from . import models
from . import lightning
from . import datasets
from . import loss
from . import resnet_mixed
from . import resnet3D,resnet2D,resnet3D_2

from voxelmorph3d import UNet3d

def build_data_module(cfg, args):
    data_module = datasets.DATA_MODULES["INB"]
    return data_module(cfg, args)

def build_lightning_model(cfg, dm):
    module = lightning.LIGHTNING_MODULES[cfg.phase.lower()]
    module = module(cfg)
    module.dm = dm
    return module

def build_img_model(cfg, MG):
    if MG==True:
        if cfg.phase.lower() =='classification_mg':
            if cfg.mip:
                image_model = resnet2D.resnet18model(cfg)
                try:
                    pretrain = torch.load(cfg.model.checkpoint)
                except:
                    return image_model
            else:
                resnet50 = resmodels.resnet50(pretrained=True)
                resnet_layer = nn.Sequential(*list(resnet50.children())[:-1])
                resnet50 = resnet_layer
                # resnet50.fc = nn.Linear(resnet50.fc.in_features, 2)
                # image_model = resnet_mixed.resnet50_2D()
                return resnet50

        else:
            image_model = models.IMAGE_MODELS[cfg.phase.lower()]
            return image_model(cfg)
    else:#mri
        if cfg.phase.lower() == 'classification':
            model = resnet3D.resnet18(
                cfg =cfg,
                sample_size=128,
                sample_duration=16)
        elif cfg.phase.lower() == 'classification_mri2':
            model = resnet3D_2.resnet18(
                cfg=cfg,
                sample_size=128,
                sample_duration=16)
        net_dict = model.state_dict()
        if cfg.finetune:
            pretrain = torch.load(cfg.model.checkpoint)
            fixed_ckpt_dict = {}
            i = 0
            for k, v in pretrain['state_dict'].items():
                new_key = list(net_dict.keys())[i]
                fixed_ckpt_dict[new_key] = v
                i += 1
            ckpt_dict = fixed_ckpt_dict
        pretrain_dict = {k: v for k, v in ckpt_dict.items() if k in net_dict.keys()}
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        return model

def build_optimizer(cfg, lr, model):

    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)

    # define optimizers
    if cfg.train.optimizer.name == "SGD":
        return torch.optim.SGD(
            params, lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    elif cfg.train.optimizer.name == "Adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=cfg.train.optimizer.weight_decay,
            betas=(0.5, 0.999),
        )
    elif cfg.train.optimizer.name == "AdamW":
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=cfg.train.optimizer.weight_decay
        )


def build_scheduler(cfg, optimizer, dm=None):

    if cfg.train.scheduler.name == "warmup":

        def lambda_lr(epoch):
            if epoch <= 3:
                return 0.001 + epoch * 0.003
            if epoch >= 22:
                return 0.01 * (1 - epoch / 200.0) ** 0.9
            return 0.01

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    elif cfg.train.scheduler.name == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif cfg.train.scheduler.name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
    elif cfg.train.scheduler.name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    else:
        scheduler = None

    if cfg.lightning.trainer.val_check_interval is not None:
        cfg.train.scheduler.interval = "step"
        num_iter = len(dm.train_dataloader().dataset)
        if type(cfg.lightning.trainer.val_check_interval) == float:
            frequency = int(num_iter * cfg.lightning.trainer.val_check_interval)
            cfg.train.scheduler.frequency = frequency
        else:
            cfg.train.scheduler.frequency = cfg.lightning.trainer.val_check_interval

    scheduler = {
        "scheduler": scheduler,
        "monitor": cfg.train.scheduler.monitor,
        "interval": cfg.train.scheduler.interval,
        "frequency": cfg.train.scheduler.frequency,
    }

    return scheduler


def build_loss(cfg):

    if cfg.train.loss_fn.type == "DiceLoss":
        return loss.segmentation_loss.DiceLoss()
    elif cfg.train.loss_fn.type == "FocalLoss":
        return loss.segmentation_loss.FocalLoss()
    elif cfg.train.loss_fn.type == "MixedLoss":
        return loss.segmentation_loss.MixedLoss(alpha=cfg.train.loss_fn.alpha)
    elif cfg.train.loss_fn.type == "BCE":
        # if cfg.train.loss_fn.class_weights is not None:
        #     weight = torch.Tensor(cfg.train.loss_fn.class_weights)
        #     loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        # else:
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn
    elif cfg.train.loss_fn.type == "CE":
        # if cfg.train.loss_fn.class_weights is not None:
        #     weight = torch.Tensor(cfg.train.loss_fn.class_weights)
        #     loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        # else:
        loss_fn = nn.CEWithLogitsLoss()
        return loss_fn
    else:
        raise NotImplementedError(f"{cfg.train.loss_fn} not implemented yet")


def build_transformation(cfg, split):

    t = []
    if split == "train":
        if cfg.transforms.random_crop is not None:
            t.append(transforms.RandomCrop(cfg.transforms.random_crop.crop_size))

        if cfg.transforms.random_horizontal_flip is not None:
            t.append(
                transforms.RandomHorizontalFlip(p=cfg.transforms.random_horizontal_flip)
            )

        if cfg.transforms.random_affine is not None:
            t.append(
                transforms.RandomAffine(
                    cfg.transforms.random_affine.degrees,
                    translate=[*cfg.transforms.random_affine.translate],
                    scale=[*cfg.transforms.random_affine.scale],
                )
            )

        if cfg.transforms.color_jitter is not None:
            t.append(
                transforms.ColorJitter(
                    brightness=[*cfg.transforms.color_jitter.bightness],
                    contrast=[*cfg.transforms.color_jitter.contrast],
                )
            )
    else:
        if cfg.transforms.random_crop is not None:
            t.append(transforms.CenterCrop(cfg.transforms.random_crop.crop_size))

    t.append(transforms.ToTensor())
    if cfg.transforms.norm is not None:
        if cfg.transforms.norm == "imagenet":
            t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        elif cfg.transforms.norm == "half":
            t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            raise NotImplementedError("Normaliation method not implemented")

    return transforms.Compose(t)
