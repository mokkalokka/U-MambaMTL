#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
from pathlib import Path
import numpy as np
import torch
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from report_guided_annotation import extract_lesion_candidates
from UMambaBot_3d import UMambaBot
import os
from monai import transforms


def predict_prostate():
    """
    Wrapper to deploy trained U-Mamba Model as a
    grand-challenge.org algorithm.
    """

    # directory to model weights
    algorithm_weights_dir = Path("weights/")

    # define compute used for training/inference ('cpu' or 'cuda')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # path to image files
    image_input_dirs = [
        "/input/images/transverse-t2-prostate-mri/",
        "/input/images/transverse-adc-prostate-mri/",
        "/input//images/transverse-hbv-prostate-mri/",
    ]
    datalist = {"t2w": image_input_dirs[0] + os.listdir(image_input_dirs[0])[0],
                "hbv": image_input_dirs[1] + os.listdir(image_input_dirs[1])[0],
                "adc": image_input_dirs[2] + os.listdir(image_input_dirs[2])[0]}

    ##### Setting up Transforms #####
    
    img_keys = ["t2w", "hbv", "adc"]
    monai_transforms = transforms.Compose(
        [
        transforms.LoadImageD(keys="t2w", ensure_channel_first=True, reader="ITKReader"),
        transforms.LoadImageD(keys="hbv", ensure_channel_first=True, reader="ITKReader"),
        transforms.LoadImageD(keys="adc", ensure_channel_first=True, reader="ITKReader"),
        transforms.ResampleToMatchD(keys=["hbv", "adc"], key_dst="t2w"),
        
        transforms.Orientationd(keys=img_keys, axcodes="RAS"),
        transforms.SpacingD(keys=img_keys, pixdim=[.5, .5, 3]),
        transforms.SpatialPadd(keys=img_keys, spatial_size=[128,128,20], mode="reflect"),
        transforms.CenterSpatialCropd(keys=img_keys, roi_size=[256,256,20]),
        
        transforms.ConcatItemsd(img_keys, name="image"),
        transforms.ClipIntensityPercentilesd(keys=["image"], 
                                            lower=1, 
                                            upper=99, 
                                            channel_wise=True),
        transforms.NormalizeIntensityd(keys=["image"], channel_wise=True),
        transforms.ToTensorD(["image"], device=device)
        
        ]
    )
    
    save_im_fn = transforms.SaveImaged(
                    keys="prostate",
                    meta_keys="pred_meta_dict",
                    output_dir="/output/images/prostate/" ,
                    output_postfix="prostate",
                    resample=True,
                    mode="nearest",
                    padding_mode="zeros",
                    output_ext=".mha",
                    writer="ITKWriter",
                    separate_folder=False
                    ) 
    save_im_fn.set_options(write_kwargs={"compression": True})
        
        
    ##### Setting up Models #####
    
    model = UMambaBot(
        input_channels=3,
        n_stages=7,
        features_per_stage=[32, 64, 128, 256, 256, 384, 512],
        conv_op=torch.nn.modules.conv.Conv3d,
        kernel_sizes=[[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        num_classes=2,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2, 2],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-05, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.modules.activation.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        strides=[[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]],
        )


    state_dict = torch.load(f"{algorithm_weights_dir}/prostate_weights.ckpt", map_location=device)['state_dict']
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Preprocessing Images ...")
    x = monai_transforms(datalist)
    print("Complete.")

    print("Generating Predictions ...")
    # switch model to evaluation mode
    model.eval()

    # scope to disable gradient updates
    with torch.no_grad():
        # aggregate predictions for all tta samples

        pred = torch.sigmoid(model(x["image"][None]))[:, 1, ...].detach().cpu().numpy()[0]
    
    pred = x["t2w"].clone().detach().set_array(torch.tensor(pred[None]))
    data = {"prostate": pred}
    res = save_im_fn(data)

