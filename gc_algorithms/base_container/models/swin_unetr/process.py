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
from monai.networks.nets import SwinUNETR
import os
from monai import transforms
from process_prostate import predict_prostate
from monai.transforms.utils import allow_missing_keys_mode


class csPCaAlgorithm(SegmentationAlgorithm):
    """
    Wrapper to deploy trained U-Mamba Model as a
    grand-challenge.org algorithm.
    """

    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        
        print("Predicting prostate...")
        predict_prostate()
        print("Done!")

        # set expected i/o paths in gc env (image i/p, algorithms, prediction o/p)
        # see grand-challenge.org/algorithms/interfaces/ for expected path per i/o interface
        # note: these are fixed paths that should not be modified

        # directory to model weights
        self.algorithm_weights_dir = Path("weights/")

        # define compute used for training/inference ('cpu' or 'cuda')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # path to image files
        self.image_input_dirs = [
            "/input/images/transverse-t2-prostate-mri/",
            "/input/images/transverse-adc-prostate-mri/",
            "/input/images/transverse-hbv-prostate-mri/",
        ]
        self.datalist = {"t2w": self.image_input_dirs[0] + os.listdir(self.image_input_dirs[0])[0],
                    "hbv": self.image_input_dirs[1] + os.listdir(self.image_input_dirs[1])[0],
                    "adc": self.image_input_dirs[2] + os.listdir(self.image_input_dirs[2])[0],
                    "prostate": "/output/images/prostate/" + os.listdir("/output/images/prostate/")[0]}

        ##### Setting up Transforms #####
        
        self.img_keys = ["t2w", "hbv", "adc"]
        self.all_keys = self.img_keys + ["prostate"]
        self.monai_transforms = transforms.Compose(
            [
            transforms.LoadImageD(keys="t2w", ensure_channel_first=True, reader="ITKReader"),
            transforms.LoadImageD(keys="hbv", ensure_channel_first=True, reader="ITKReader"),
            transforms.LoadImageD(keys="adc", ensure_channel_first=True, reader="ITKReader"),
            transforms.LoadImageD(keys="prostate", ensure_channel_first=True, reader="ITKReader"),
            transforms.AsDiscreted(keys="prostate", threshold=0.5, allow_missing_keys=True),
            transforms.ResampleToMatchD(keys=["hbv", "adc"], key_dst="t2w"),
            
            transforms.Orientationd(keys=self.all_keys, axcodes="RAS"),
            transforms.SpacingD(keys=self.all_keys, pixdim=[.5, .5, 3]),
            transforms.SpatialPadd(keys=self.all_keys, spatial_size=[128,128,15], mode="reflect"),
            
            transforms.CropForegroundd(keys=self.all_keys, 
                                            source_key="prostate", 
                                            margin=[128,128,15], 
                                            allow_smaller=False, 
                                            mode="reflect"),
            transforms.CenterSpatialCropd(keys=self.all_keys, roi_size=[256,256,32]),
            
            transforms.ConcatItemsd(self.img_keys, name="image"),
            transforms.ClipIntensityPercentilesd(keys=["image"], 
                                                lower=1, 
                                                upper=99, 
                                                channel_wise=True),
            transforms.NormalizeIntensityd(keys=["image"], channel_wise=True),
            transforms.ToTensorD(["image"], device=self.device)
            
            ]
        )
        
        self.save_im_fn = transforms.SaveImaged(
                        keys="det_map",
                        meta_keys="pred_meta_dict",
                        output_dir="/output/images/cspca-detection-map/" ,
                        output_postfix="",
                        resample=False,
                        # mode="bilinear",
                        # padding_mode="zeros",
                        output_ext=".mha",
                        writer="ITKWriter",
                        separate_folder=False
                        )  
        self.save_im_fn.set_options(write_kwargs={"compression": True})
        
        
        ##### Setting up Models #####
        
        self.models = []
        for fold in [0, 1, 2, 3, 4]:
            
            

            model = SwinUNETR(
            img_size=(256,256,32),
            in_channels=3,
            out_channels=2,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
            use_v2=True
            )


            state_dict = torch.load(f"{self.algorithm_weights_dir}/f{fold}.ckpt", map_location=self.device)['state_dict']
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self.models.append(model)
    

        
        # display error/success message
        if len(self.models) == 0:
            raise Exception("No models have been found/initialized.")
        else:
            print(f"Success! {len(self.models)} model(s) have been initialized.")
            print("-"*100)
            
            

    # generate + save predictions, given images
    def predict(self):

        print("Preprocessing Images ...")
        x = self.monai_transforms(self.datalist)
        print("Complete.")

        print("Generating Predictions ...")
        outputs = []
        # for each member model in ensemble
        for p in range(len(self.models)):

            # switch model to evaluation mode
            self.models[p].eval()

            # scope to disable gradient updates
            with torch.no_grad():
                # aggregate predictions for all tta samples

                pred = torch.sigmoid(self.models[p](x["image"][None]))[:, 1, ...].detach().cpu().numpy()
                pred = pred * x["prostate"].detach().cpu().numpy()
                outputs += [pred[0]]

        # ensemble softmax predictions
        ensemble_output = np.mean(outputs, axis=0).astype('float32')
        
        # Inverse transform back to t2w space
        # Uses the same t2w transforms on the det_map and reverts them back to oiginal
        pred = x["t2w"].clone().set_array(torch.tensor(ensemble_output[None]))
        prob_map = {"t2w": pred}
        with allow_missing_keys_mode(self.monai_transforms):
            prob_map_orig = self.monai_transforms.inverse(prob_map)
        

        print("Generating Detection Map ...")
        # process softmax prediction to detection map
        det_map = extract_lesion_candidates(prob_map_orig["t2w"][0].cpu().numpy(), threshold='dynamic', num_lesions_to_extract=3,min_voxels_detection=72, dynamic_threshold_factor=3.0)[0]
        # det_map[det_map<(np.max(det_map)/5)] = 0
        print("Complete.")

        print("Saving detection map..")
        det_map = prob_map_orig["t2w"].clone().set_array(torch.tensor(det_map[None]))
        data = {"det_map": det_map}
        res = self.save_im_fn(data)

        # Change the name to the expected name
        old_name = res["det_map"].meta["filename_or_obj"].split("/")[-1]
        new_name = "cspca_detection_map.mha"
        base_dir = "/output/images/cspca-detection-map/"
        os.rename(base_dir + old_name, base_dir + new_name)
        print("Complete.")

        # save case-level likelihood
        with open("/output/cspca-case-level-likelihood.json", 'w') as f:
            json.dump(float(np.max(det_map)), f)

            

if __name__ == "__main__":
    csPCaAlgorithm().predict()
