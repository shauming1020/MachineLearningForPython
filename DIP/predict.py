# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:30:51 2019

@author: DCMC
"""
import logging
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval().cuda()

    ds = BasicDataset('./data/imgs', './data/masks', scale=scale_factor)
    img = torch.from_numpy(ds.preprocess(full_img))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    in_files = "./data/imgs_test/0058.png"

    net = UNet(n_channels=1, n_classes=1)
    Loaded_model = "./model/BEST.pth"
    
    logging.info("Loading model {}".format(Loaded_model))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(Loaded_model, map_location=device))
    
    logging.info("Model loaded !")
    
    logging.info("\nPredicting image {} ...".format(in_files))
    
    img = Image.open(in_files)
    
    mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=0.2,
                        out_threshold=0.9,
                        device=device)
    logging.info("Visualizing results for image {}, close to continue ...".format(in_files))
    plot_img_and_mask(img, mask)
    
    # write
    mask_bgr = mask_to_image(mask)    
    mask_bgr = mask_bgr.resize((500, 1200))
    mask_bgr.save('./predict/predict.png')
    
