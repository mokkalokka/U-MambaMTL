"""Pre- and post-processing transforms for umamba_mtl_single inference.

Adapted from ``umamba_mtl.data_module_all``: each MRI sequence (T2w, ADC, HBV) is
loaded separately and ADC/HBV are resampled onto the T2w grid. This works for both
.mha and .nii/.nii.gz inputs, even when ADC/HBV were not resampled onto the T2w
grid beforehand. No prostate segmentation is used for cropping, so the pipeline
defaults to a center crop of the image.

The default values match the training configuration
(experiments/picai/umamba_mtl_single/config.yaml).
"""

from monai import transforms


def get_inference_transforms(
    spacing=(0.5, 0.5, 3.0),
    roi_size=(256, 256, 20),
    padding_mode="reflect",
    clip_lower=1,
    clip_upper=99,
):
    """Build the preprocessing transforms.

    Returns the spatial and intensity transforms separately: predictions are
    inverted back to the original T2w grid through the *spatial* part only.
    """
    prep_spatial = transforms.Compose([
        transforms.LoadImageD(keys="t2w", ensure_channel_first=True, reader="ITKReader"),
        transforms.LoadImageD(keys="adc", ensure_channel_first=True, reader="ITKReader"),
        transforms.LoadImageD(keys="hbv", ensure_channel_first=True, reader="ITKReader"),
        transforms.ResampleToMatchD(keys=["hbv", "adc"], key_dst="t2w"),
        transforms.ConcatItemsd(["t2w", "adc", "hbv"], name="image"),
        transforms.Spacingd(keys="image", pixdim=spacing, mode="bilinear"),
        transforms.Orientationd(keys="image", axcodes="RAS"),
        transforms.SpatialPadd(keys="image", spatial_size=roi_size, mode=padding_mode),
        transforms.CenterSpatialCropd(keys="image", roi_size=roi_size),
    ])
    prep_intensity = transforms.Compose([
        transforms.ClipIntensityPercentilesd(
            keys=["image"], lower=clip_lower, upper=clip_upper, channel_wise=True
        ),
        transforms.NormalizeIntensityd(keys=["image"], channel_wise=True),
    ])
    return prep_spatial, prep_intensity


def get_post_transforms(prep_spatial, output_dir, save_preds=True):
    """Build the post-processing transforms for the two model outputs.

    Predictions are inverted back to the original T2w grid. The zonal prediction
    is converted to labels (0=background, 1=PZ, 2=TZ). If ``save_preds``, both are
    saved to ``output_dir`` as ``*_pca_prob.nii.gz`` (float32 probabilities) and
    ``*_zonal_seg.nii.gz`` (uint8 labels).
    """

    def build(key, output_postfix, output_dtype, argmax=False):
        steps = [
            transforms.Invertd(
                keys=key, transform=prep_spatial, orig_keys="image",
                nearest_interp=False, to_tensor=True,
            ),
        ]
        if argmax:
            steps.append(transforms.AsDiscreted(keys=key, argmax=True, to_onehot=None))
        if save_preds:
            steps.append(
                transforms.SaveImaged(
                    keys=key,
                    output_dir=str(output_dir),
                    output_postfix=output_postfix,
                    output_dtype=output_dtype,
                    resample=False,
                    separate_folder=False,
                )
            )
        return transforms.Compose(steps)

    pca_post = build("pca", "pca_prob", "float32")
    zonal_post = build("zonal", "zonal_seg", "uint8", argmax=True)
    return pca_post, zonal_post