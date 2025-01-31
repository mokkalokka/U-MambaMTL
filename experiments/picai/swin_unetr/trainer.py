import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from functools import partial
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
)
import warnings
from shared_modules.data_module import DataModule
from shared_modules.module_selector import get_network, get_logger, get_callbacks, get_loss_fn
from shared_modules.utils import load_config
from shared_modules.plotting import plot_confusion, plot_difference, plot_metrics
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from shared_modules.torch_metrics import PicaiMetric


torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=UserWarning)



class LitModel(pl.LightningModule):
    def __init__(self, config):
        
        super(LitModel, self).__init__()
        
        self.config = config
        self.lr = config.lr
        
        self.model = get_network(config)
        self.loss_fn = get_loss_fn(config)
        self.dsc_fn = DiceMetric(include_background=False, reduction="mean")
        self.picai_metric_fn = PicaiMetric() #compute_on_cpu=True

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
        img, pca = batch["image"], batch["pca"]

        logits = self(img)   
        loss = self.loss_fn(logits, pca) 
        
        self.log_dict({
            "train/loss": loss,
        }, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        img, pca = batch["image"], batch["pca"]
        
        logits = self.inferer(img)
        
        tumor_probability = torch.sigmoid(logits[0,1])[None]
        
        pred = AsDiscrete(argmax=True, to_onehot=2)(logits[0])[None]
        label_onehot = AsDiscrete(to_onehot=2)(pca[0])[None]

        self.dsc_fn(y_pred=pred, y=label_onehot)
        self.picai_metric_fn.update(preds=tumor_probability, target=pca[0])
    
    def on_validation_epoch_end(self):
        dsc = self.dsc_fn.aggregate("mean_batch")
        self.dsc_fn.reset()
        metrics = self.picai_metric_fn.compute()
        # print(metrics)

        # if self.current_epoch < 15:
        #     metrics = PlaceholderMetrics()
        
        if self.global_rank == 0:
        
            images = [plot_fn(metrics, epoch=self.current_epoch) for plot_fn in [plot_metrics, plot_confusion, plot_difference]]
            self.logger.log_image(key=f"rank_{self.global_rank}", images=images)      
                  
        self.log_dict({
                "val/pi_cai_score": metrics.score,
                "val/ap": metrics.AP,
                "val/auroc": metrics.auroc,
                "val/dsc_tumor": dsc,
            }, sync_dist=True, prog_bar=True, on_epoch=True)  
        

        self.picai_metric_fn.reset()



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
