import os
import cv2
import torch
import pydicom
import snapshot
import numpy as np
import pandas as pd
import datetime
from PIL import Image
import PIL
import torch.nn as nn
from torch.utils.data import Dataset
# from albumentations import ShiftScaleRotate, Normalize, Resize, Compose
# from albumentations.pytorch import ToTensor
# from albumentations.pytorch.transforms import ToTensor

from torchvision import transforms
from scipy.ndimage import zoom
import SimpleITK as sitk
from datetime import datetime, timedelta
from networks import Unet


def resampleDCM(itk_imgs, new_spacing= [1, 1, 1], is_label= False, new_size = None,
                new_origin= None, new_direction = None):
    """ resample SimpleITK Image variable
    itk_imgs:      SimpleITK Image variable for input image to be resampled
    new_spacing:   output spacing
    is_label:      whether to resample a image or a label
    new_size:      output size
    new_origin:    output origin
    new_direction: output direction [discarded]
    """
    resample = sitk.ResampleImageFilter()

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    if new_origin is None:
        new_origin = itk_imgs.GetOrigin()
    if new_direction is None:
        new_direction = itk_imgs.GetDirection()
    resample.SetOutputOrigin(new_origin)
    resample.SetOutputSpacing(new_spacing)
    # resample.SetOutputDirection(new_direction)
    resample.SetDefaultPixelValue(itk_imgs.GetPixel(0, 0, 0))

    ori_size = np.array(itk_imgs.GetSize(), dtype=np.int32)
    ori_spacing = np.asarray(itk_imgs.GetSpacing())
    if new_size is None:
        new_size = ori_size * (ori_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(np.int32)
        new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)

    out = resample.Execute(itk_imgs)
    return out, {'size': new_size, 'origin': new_origin, 'direction': new_direction}


def normSize(itk_imgs, new_size=[125, 260, 260]):
    """
    itk_imgs: SimpleITK Image variable for input image to be normed size
    new_size: output size [h, w, d]
    """
    size = itk_imgs.GetSize()

    # padding
    pad = np.maximum(np.array(new_size) - np.array(size), 0)
    pad_low = pad // 2
    pad_high = pad - pad_low
    itk_imgs = sitk.ConstantPad(itk_imgs, pad_low.tolist(), pad_high.tolist(), itk_imgs.GetPixel(0, 0, 0))

    size = itk_imgs.GetSize()

    # crop
    crop = np.maximum(np.array(size) - np.array(new_size), 0)
    crop_low = crop // 2
    crop_high = crop_low + new_size
    itk_imgs = itk_imgs[crop_low[0]:crop_high[0], crop_low[1]:crop_high[1], crop_low[2]:crop_high[2]]

    return itk_imgs


def histMatch(src, tgt, hist_level: int = 1024, match_points: int = 7):
    """ histogram matching from source image to target image
    src:          SimpleITK Image variable for source image
    tgt:          SimpleITK Image variable for target image
    hist_level:   number of histogram levels
    match_points: number of match points
    """
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(hist_level)
    matcher.SetNumberOfMatchPoints(match_points)
    matcher.ThresholdAtMeanIntensityOn()
    dist = matcher.Execute(src, tgt)
    size = dist.GetSize()
    return dist

def resampleVolume(outspacing, vol):
    outsize = [0, 0, 0]
    transform = sitk.Transform()
    transform.SetIdentity()
    outsize[0] = round(256)
    outsize[1] = round(256)
    outsize[2] = round(256)
    # outsize = [256,256,256]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol


def preprocess_1(x):
    size = (125, 360, 360)
    d, w, h = x.shape
    # if d < size[0] or w < size[1] or h < size[2]:
    pd = np.max([0, size[0] - d])
    pw = np.max([0, size[1] - w])
    ph = np.max([0, size[2] - h])

    if pd == 0 and pw == 0 and ph == 0:
        ph = np.max([0, size[2] + 40 - h])
        # pw = np.max([0, size[1] +20 - w])
        x = np.pad(x, ((pd, pd), (pw, pw), (ph, ph)))
        x = x[pd // 2:pd // 2 + size[0], (pw // 2): (pw // 2) + size[1], (ph // 2) + 40: (ph // 2 + size[2] + 40)]
    else:
        x = np.pad(x, ((pd, pd), (pw, pw), (ph, ph)))
        x = x[pd // 2:pd // 2 + size[0], (pw // 2): (pw // 2) + size[1], ph // 2:ph // 2 + size[2]]

    x = np.expand_dims(x, 0)
    return x


def load_nii(nii_file):
    set_spacing = [1, 1, 1]
    # set_size = [h, w, d]  # d,w,h = image.array.shape
    set_size = [360, 360, 125]
    new_size = None
    new_origin = None
    itk_img = sitk.ReadImage(nii_file)
    itk_img = sitk.DICOMOrient(itk_img, 'LPS') # present 'RAI' in itk-snap
    img, _ = resampleDCM(itk_img, new_spacing=set_spacing, is_label=False, new_size=new_size,
                                                 new_origin=new_origin)
    img = normSize(img, set_size)


    return img


def write_nii(tgt_path, mri_img_path, date1):
    dcms, tags = utils.readDicom(
        os.path.join(mri_img_path,
                     date1 + '_sinwas.nii.gz'))
    tags = tags[list(dcms.keys())[0]]
    dcm = dcms[list(dcms.keys())[0]][tags['SeriesDescription']]

    sitk.WriteImage(dcm, os.path.join(tgt_path, '{}_{}_sinwas.nii.gz'.format(mri_img_path, date1)))


def plot_post(view, view0slicey1, view0slicey2, path1, path2, label):
    p = np_to_pil(view0slicey1)
    p.save(path1)

    p = np_to_pil(view0slicey2)
    p.save(path2.split('.png')[0] + 'pcr_' + str(label) + '.png')


def createMIP(img, slices_num=1):
    ''' create the mip image from original image, slice_num is the number of
    slices for maximum intensity projection'''
    # np_img = np.asarray(img)
    np_img = sitk.GetArrayFromImage(img)
    # np_img = img.cpu().numpy()
    # np_img = np_img[0][0]
    a = np_img[0:125, :, :]
    img_np = np.amax(a, 0)

    img = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    ar = np.uint8(img * 255)
    # ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    # if img_np.shape[0] == 1:
    #     ar = ar[0]
    # else:
    #     assert img_np.shape[0] == 3, img_np.shape
    #     ar = ar.transpose(1, 2, 0)
    #

        # np_mip[i, :, :] = np.amax(a, 0)
    return ar

def seg_breast(img):
    inshape = (125,360,360)#(192,192,192)

    enc_nf = [32, 64, 128, 256]
    dec_nf = [256, 128, 64, 32, 16]
    net = Unet(
        inshape=inshape,
        nb_features=[enc_nf, dec_nf],
        n_class=2,
    )
    net = net.float()
    net.load_state_dict(torch.load('/med_labelme/MR_breast_segmentation/ckpt/ckpt_37000_0.94541.pth', map_location='cpu'))
    img_np = sitk.GetArrayFromImage(img)
    tensor = torch.from_numpy(img_np.astype(np.float))
    img = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img = torch.from_numpy(img)
    pred = nn.Softmax(1)(net(img.float().unsqueeze(0).unsqueeze(0)))
    pred = torch.where(pred > 0.5,1,0)
    pred = torch.split(pred, 1, dim =1)[1]

    return pred*tensor

if __name__ == "__main__":

    df = pd.read_csv('duke.csv')

    mip_path = '/preprocessMRI/mip/'

    for path, dir_list, file_list in os.walk("/Duke/"):
        for x_idx1 in file_list:
            try:
                img1 = load_nii(path+'/dce0.nii.gz')  # [125,336.336]original size
                img2 = load_nii(path+'/dce1.nii.gz')
            except:

                continue
            else:
                dist = histMatch(sitk.Cast(img1, sitk.sitkInt16),sitk.Cast(img2, sitk.sitkInt16))
                sitk.WriteImage(sitk.Cast((img2 - img1), sitk.sitkInt16),'/Duke/subtraction/nii/' + path.split('/')[-1] + '.nii.gz')
                mip_i1 = createMIP(sitk.Cast((img2 - img1), sitk.sitkInt16))
                mip_i1 = Image.fromarray(mip_i1)
                mip_i1.save('/Duke/subtraction/mip/'+ path.split('/')[-1]+'.png')
                break
print('finish!')



