import numpy as np
import scipy.io as sio
import glob
import zipfile
import numpy as np
import torch
from vgg import *
from resnet import *
from alexnet import *
from torch.autograd import Variable as V
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

def run_alexnet(image_dir,net_save_dir):
    """
    This generates activations and saves in net_save_dir
    Input:
    image_dir: Image directory containing .jpg files
    net_save_dir: directory for saving activations
    Activations are saved in format XY.mat where XY is the image file
    XY.mat contains activations for specific layers in with corresponding layer's name
    """
    model = AlexNet()
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    centre_crop = trn.Compose([
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_list = glob.glob(image_dir +"\\*.jpg")
    image_list.sort()
    image_list=image_list
    print(image_list)
    for image in image_list:
        img = Image.open(image)
        filename=image.split("\\")[-1].split(".")[0]
        input_img = V(centre_crop(img).unsqueeze(0))
        print(input_img.size(),filename)
        print(input_img.size())
        if torch.cuda.is_available():
            input_img=input_img.cuda()
        x = model.forward(input_img)
        save_path = os.path.join(net_save_dir,filename+".mat")
        feats={}
        for i,feat in tqdm(enumerate(x)):
            print(save_path)
            print(feat.data.cpu().numpy().shape)
            feats[model.feat_list[i]] = feat.data.cpu().numpy()
        sio.savemat(save_path,feats)
def run_vgg(image_dir,net_save_dir):
    """
    This generates activations and saves in net_save_dir
    Input:
    image_dir: Image directory containing .jpg files
    net_save_dir: directory for saving activations
    Activations are saved in format XY.mat where XY is the image file
    XY.mat contains activations for specific layers in with corresponding layer's name
    """
    model = VGGNet()
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    centre_crop = trn.Compose([
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_list = glob.glob(image_dir +"\\*.jpg")
    image_list.sort()
    image_list=image_list
    print(image_list)
    for image in image_list:
        img = Image.open(image)
        filename=image.split("\\")[-1].split(".")[0]
        input_img = V(centre_crop(img).unsqueeze(0))
        print(input_img.size(),filename)
        print(input_img.size())
        if torch.cuda.is_available():
            input_img=input_img.cuda()
        x = model.forward(input_img)
        save_path = os.path.join(net_save_dir,filename+".mat")
        feats={}
        for i,feat in tqdm(enumerate(x)):
            print(save_path)
            print(feat.data.cpu().numpy().shape)
            feats[model.feat_list[i]] = feat.data.cpu().numpy()
        sio.savemat(save_path,feats)
def run_resnet(image_dir,net_save_dir):
    """
    This generates activations and saves in net_save_dir
    Input:
    image_dir: Image directory containing .jpg files
    net_save_dir: directory for saving activations
    Activations are saved in format XY.mat where XY is the image file
    XY.mat contains activations for specific layers in with corresponding layer's name
    """
    model = resnet50(pretrained=True)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    centre_crop = trn.Compose([
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_list = glob.glob(image_dir +"\\*.jpg")
    image_list.sort()
    image_list=image_list
    print(image_list)
    for image in image_list:
        img = Image.open(image)
        filename=image.split("\\")[-1].split(".")[0]
        input_img = V(centre_crop(img).unsqueeze(0))
        print(input_img.size(),filename)
        print(input_img.size())
        if torch.cuda.is_available():
            input_img=input_img.cuda()
        x = model.forward(input_img)
        save_path = os.path.join(net_save_dir,filename+".mat")
        feats={}
        for i,feat in tqdm(enumerate(x)):
            print(save_path)
            print(feat.data.cpu().numpy().shape)
            feats[model.feat_list[i]] = feat.data.cpu().numpy()
        sio.savemat(save_path,feats)



def zip(src, dst):
    zf = zipfile.ZipFile("%s.zip" % (dst), "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dirname, subdirs, files in os.walk(src):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            arcname = absname[len(abs_src) + 1:]
            print("zipping {} as {}".format(os.path.join(dirname, filename),arcname))
            zf.write(absname, arcname)
    zf.close()
