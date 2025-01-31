import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from functools import partial
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    SaveImage
)
import warnings
from data_module import DataModule
from utils.plotting import plot_confusion, plot_difference, plot_metrics
from module_selector import get_network, get_logger, get_callbacks
from utils.utils import load_config, load_pretrained_model
from torch_metrics import PicaiMetric
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from einops import rearrange
from monai.losses import FocalLoss, DiceCELoss, DeepSupervisionLoss





torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=UserWarning)


class LitModel(pl.LightningModule):
    def __init__(self, config):
        """
        Inputs:
            config
        """

        super(LitModel, self).__init__()
        
        self.config = config
        self.lr = config.lr
        # self.save_hyperparameters()
        self.model = get_network(config)
        # self.model = load_pretrained_model(self.model)

        self.loss_fn_tumor = FocalLoss(include_background=True, gamma=config.loss.gamma)
        self.loss_fn_anatomy = DiceCELoss(sigmoid=True)
        
        if config.network.deep_supervision:
            self.loss_fn_tumor = DeepSupervisionLoss(FocalLoss(include_background=True, gamma=config.loss.gamma))
            self.loss_fn_anatomy= DeepSupervisionLoss(DiceCELoss(sigmoid=True))
            # self.loss_fn_tz = DeepSupervisionLoss(DiceCELoss(sigmoid=True))
            

        self.dsc_fn = DiceMetric(include_background=True, reduction="mean")
        self.picai_metric_fn = PicaiMetric()

        self.inferer = partial(
            sliding_window_inference,
            roi_size=config.transforms.spatial_size,
            sw_batch_size=config.sw_batch_size,
            predictor=self.model,
            overlap=config.infer_overlap,
        )
        self.post_pred = Compose(
            [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
        )
        self.post_logits = Compose(
            [Activations(sigmoid=True)]
        )

        self.validation_step_outputs = {'dsc': []}

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        lr_scheduler = WarmupCosineSchedule(optimizer=optimizer, warmup_steps=self.config.warmup_epochs, t_total=self.config.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        # img, tumor_gt, pz_gt, tz_gt = batch["image"], batch["tumor"], batch["pz"], batch["tz"]
        img, tumor_gt, tz_gt, pz_gt = batch["image"], batch["tumor"], batch["pz"], batch["tz"] # PZ TZ Inverted for picai!
        anatomy_gt = torch.concat([pz_gt, tz_gt], dim=1)

        logits = self(img)   
        loss_tumor = self.loss_fn_tumor(logits[:, 0:1, ...], tumor_gt) 
        loss_anatomy = self.loss_fn_anatomy(logits[:, 1:3, ...], anatomy_gt)

        loss = loss_tumor + 0.1 * loss_anatomy
        
        self.log_dict({
            "train/loss_tumor": loss_tumor,
            "train/loss_anatomy": loss_anatomy,
        }, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        img, tumor_gt, tz_gt, pz_gt = batch["image"], batch["tumor"], batch["pz"], batch["tz"] # PZ TZ Inverted for picai!
        all_gt = torch.concat([tumor_gt, pz_gt, tz_gt], dim=1)
        
        logits = self.inferer(img)
        tumor_probability = self.post_logits([logits[0, 0:1, ...]])[0]
        pred = [self.post_pred(i) for i in decollate_batch(logits)]
        
        prostate_pred = torch.where(pred[0][1][None] + pred[0][2][None] > 0, 1, 0)
        filtered_tumor_probability = (tumor_probability * prostate_pred)[None]
        filtered_tumor_probability = torch.nan_to_num(filtered_tumor_probability, nan=0.0)
        
        self.dsc_fn(y_pred=pred, y=all_gt)
        self.picai_metric_fn.update(preds=filtered_tumor_probability, target=tumor_gt)
    
    def on_validation_epoch_end(self):
        dsc = self.dsc_fn.aggregate("mean_batch")
        self.dsc_fn.reset()
            
        if self.trainer.global_step == 0:
            self.logger.experiment.define_metric("val/pi_cai_score", summary="max")
            self.logger.experiment.define_metric("val/dsc_tumor", summary="max")
            metrics = self.picai_metric_fn.reset()
            return
        
        # if self.current_epoch < 0:
        #     self.log_dict({
        #         "val/pi_cai_score": 0,
        #         "val/dsc_tumor": dsc[0],
        #         "val/dsc_pz": dsc[1],
        #         "val/dsc_tz": dsc[2],
        #     }, sync_dist=True, prog_bar=True, on_epoch=True)  
        #     metrics = self.picai_metric_fn.reset()
        #     return
        
        metrics = self.picai_metric_fn.compute()
        print(metrics)
        if self.global_rank == 0:
            print("\n")
            print(f"PI-CAI Score:\t\t {round(metrics.score, 4)}") 
            print(f"PI-CAI Metrics:\t\t {metrics}") 
            print("\n")
        
            images = [plot_fn(metrics, epoch=self.current_epoch) for plot_fn in [plot_metrics, plot_confusion, plot_difference]]
            self.logger.log_image(key=f"rank_{self.global_rank}", images=images)      
                  
        self.log_dict({
                "val/pi_cai_score": metrics.score,
                "val/ap": metrics.AP,
                "val/auroc": metrics.auroc,
                "val/dsc_tumor": dsc[0],
                "val/dsc_pz": dsc[1],
                "val/dsc_tz": dsc[2],
            }, sync_dist=True, prog_bar=True, on_epoch=True)  
            
        self.picai_metric_fn.reset()
        
    def test_step(self, batch, batch_idx):
        img, tumor_gt, pz_gt, tz_gt = batch["image"], batch["tumor"], batch["pz"], batch["tz"]
        all_gt = torch.concat([tumor_gt, pz_gt, tz_gt], dim=1)
        
        logits = self.inferer(img)
        tumor_probability = self.post_logits([logits[0, 0:1, ...]])[0]
        pred = [self.post_pred(i) for i in decollate_batch(logits)]
        
        prostate_pred = torch.where(pred[0][1][None] + pred[0][2][None] > 0, 1, 0)
        filtered_tumor_probability = (tumor_probability * prostate_pred)[None]
        filtered_tumor_probability = torch.nan_to_num(filtered_tumor_probability, nan=0.0)
        
        self.dsc_fn(y_pred=pred, y=all_gt)
        # self.picai_metric_fn.update(preds=tumor_probability[None], target=tumor_gt)
        self.picai_metric_fn.update(preds=filtered_tumor_probability, target=tumor_gt)
        
        sample_name = img._meta['filename_or_obj'][0].split("/")[-2]
        SaveImage(output_dir=f"./predictions/{self.config.network.name}/{sample_name}", output_postfix="pred", separate_folder=False)(pred[0])
        # SaveImage(output_dir=f"./predictions/{self.config.network.name}/{sample_name}", output_postfix="img", separate_folder=False)(img[0][0][None])
        # SaveImage(output_dir=f"./predictions/{self.config.network.name}/{sample_name}", output_postfix="gt", separate_folder=False)(all_gt[0])
        
        
    def on_test_epoch_end(self):
        dsc = self.dsc_fn.aggregate("mean_batch")
        self.dsc_fn.reset()
        
        metrics = self.picai_metric_fn.compute()
        print(metrics)
        if self.global_rank == 0:
            print("\n")
            print(f"PI-CAI Score:\t\t {round(metrics.score, 4)}") 
            print(f"PI-CAI Metrics:\t\t {metrics}") 
            print("\n")
        
            images = [plot_fn(metrics, epoch=self.current_epoch) for plot_fn in [plot_metrics, plot_confusion, plot_difference]]
            self.logger.log_image(key=f"rank_{self.global_rank}", images=images)      
                  
        self.log_dict({
                "test/pi_cai_score": metrics.score,
                "test/ap": metrics.AP,
                "test/auroc": metrics.auroc,
                "test/dsc_tumor": dsc[0],
                "test/dsc_pz": dsc[1],
                "test/dsc_tz": dsc[2],
            }, sync_dist=True, prog_bar=True, on_epoch=True)  
            
        self.picai_metric_fn.reset()


if __name__ == "__main__":
    config = load_config("config_swin_unetr_mtl_picai.yaml")

    data_module = DataModule(
        config,
    )

    logger = get_logger(config)
    model = LitModel(config)
    

    if config.checkpoint is not None and not config.test_mode:
        if config.checkpoint.split(".")[-1] == "ckpt":
            print("Fine tuning model from checkpoint")
            model = LitModel.load_from_checkpoint(config.checkpoint, config=config, strict=False)

    trainer = Trainer(
        accelerator="gpu",
        fast_dev_run=config.fast_dev_run,
        devices=config.gpus,
        strategy="ddp_find_unused_parameters_true",
        precision="16",
        num_sanity_val_steps=-1,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=config.val_every,
        enable_checkpointing=config.save_checkpoint,
        gradient_clip_val=1.0,
        logger=logger,
        use_distributed_sampler=False,
        callbacks=get_callbacks(config, logger),
    )
    

    if config.test_mode:
        # trainer.validate(model, data_module, ckpt_path=config.checkpoint, verbose=True)
        trainer.test(model, data_module, ckpt_path=config.checkpoint)
    else:
        if config.resume_ckpt:
            trainer.fit(model, data_module, ckpt_path=config.checkpoint)
        else:
            trainer.fit(model, data_module)
