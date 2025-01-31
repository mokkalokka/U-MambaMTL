import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from functools import partial
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
)
from monai.losses import DiceCELoss
import warnings
from shared_modules.data_module import DataModule
from shared_modules.module_selector import get_network, get_logger, get_callbacks
from shared_modules.utils import load_config
from monai.optimizers.lr_scheduler import WarmupCosineSchedule


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
        
        self.model = get_network(config)
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.dsc_fn = DiceMetric(include_background=False, reduction="mean")

        self.inferer = partial(
            sliding_window_inference,
            roi_size=config.transforms.spatial_size,
            sw_batch_size=config.sw_batch_size,
            predictor=self.model,
            overlap=config.infer_overlap,
        )

        self.validation_step_outputs = {'dsc': []}

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        lr_scheduler = WarmupCosineSchedule(optimizer=optimizer, warmup_steps=self.config.warmup_epochs, t_total=self.config.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        img, prostate_gt = batch["image"], batch["prostate"]

        logits = self(img)   
        loss = self.loss_fn(logits, prostate_gt) 
        
        self.log_dict({
            "train/loss": loss,
        }, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        img, tumor_gt, prostate_gt = batch["image"], batch["pca"], batch["prostate"]
        logits = self.inferer(img)
        
        pred = AsDiscrete(argmax=True, to_onehot=2)(logits[0])[None]
        label_onehot = AsDiscrete(to_onehot=2)(prostate_gt[0])[None]
        self.dsc_fn(y_pred=pred, y=label_onehot)
    
    def on_validation_epoch_end(self):
        dsc = self.dsc_fn.aggregate("mean_batch")
        self.dsc_fn.reset()
       
        self.log_dict({
                "val/dsc_prostate": dsc,
            }, sync_dist=True, prog_bar=True, on_epoch=True)  



if __name__ == "__main__":
    config = load_config()

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
        num_sanity_val_steps=0,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=config.val_every,
        enable_checkpointing=config.save_checkpoint,
        gradient_clip_val=1.0,
        logger=logger,
        use_distributed_sampler=False,
        callbacks=get_callbacks(config, logger),
    )
    

    if config.test_mode:
        trainer.test(model, data_module, ckpt_path=config.checkpoint)
    else:
        if config.resume_ckpt:
            trainer.fit(model, data_module, ckpt_path=config.checkpoint)
        else:
            trainer.fit(model, data_module)
