import pandas as pd
import wandb

model_list = [
    "resnet50.tv_in1k",
    "resnet101.tv_in1k",
    "resnet152.tv_in1k",
    "convnext_tiny.fb_in1k",
    "convnext_small.fb_in1k",
    "convnext_base.fb_in1k",
    "convnext_large.fb_in1k",
    "convnext_tiny.in12k_ft_in1k",
    "convnext_tiny.fb_in22k_ft_in1k",
    "vit_small_patch16_224.augreg_in1k",
    "vit_base_patch16_224.augreg_in1k",
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "vit_large_patch16_224.augreg_in21k_ft_in1k",
    "tf_efficientnet_b0.in1k",
    "tf_efficientnet_b1.in1k",
    "tf_efficientnet_b2.in1k",
    "tf_efficientnet_b3.in1k",
    "tf_efficientnet_b4.in1k",
    "tf_efficientnet_b5.in1k",
    "tf_efficientnet_b0.aa_in1k",
    "tf_efficientnet_b1.aa_in1k",
    "tf_efficientnet_b2.aa_in1k",
    "tf_efficientnet_b3.aa_in1k",
    "tf_efficientnet_b4.aa_in1k",
    "tf_efficientnet_b5.aa_in1k",
    "tf_efficientnet_b0.ap_in1k",
    "tf_efficientnet_b1.ap_in1k",
    "tf_efficientnet_b2.ap_in1k",
    "tf_efficientnet_b3.ap_in1k",
    "tf_efficientnet_b4.ap_in1k",
    "tf_efficientnet_b5.ap_in1k",
    "tf_efficientnet_b0.ns_jft_in1k",
    "tf_efficientnet_b1.ns_jft_in1k",
    "tf_efficientnet_b2.ns_jft_in1k",
    "tf_efficientnet_b3.ns_jft_in1k",
    "tf_efficientnet_b4.ns_jft_in1k",
    "tf_efficientnet_b5.ns_jft_in1k",
    "tf_efficientnet_b5.ra_in1k",
    "tf_efficientnet_lite0.in1k",
    "tf_efficientnet_lite1.in1k",
    "tf_efficientnet_lite2.in1k",
    "tf_efficientnet_lite3.in1k",
    "tf_efficientnet_lite4.in1k",
    "regnety_002.pycls_in1k",
    "regnety_004.pycls_in1k",
    "regnety_006.pycls_in1k",
    "regnety_008.pycls_in1k",
    "regnety_016.pycls_in1k",
    "regnety_032.pycls_in1k",
    "regnety_040.pycls_in1k",
    "regnety_064.pycls_in1k",
    "regnety_080.pycls_in1k",
    "regnety_120.pycls_in1k",
    "regnety_160.pycls_in1k",
    "regnety_320.pycls_in1k",
    "resnet50.a1_in1k",
    "resnet50.a1h_in1k",
    "resnet50.a2_in1k",
    "resnet50.a3_in1k",
    "resnet50.b1k_in1k",
    "resnet50.b2k_in1k",
    "resnet50.c1_in1k",
    "resnet50.c2_in1k",
    "resnet50.d_in1k",
    "resnet50.ra_in1k",
    "resnet50.ram_in1k",
    "resnet101.a1_in1k",
    "resnet101.a1h_in1k",
    "resnet101.a2_in1k",
    "resnet101.a3_in1k",
    "resnet152.a1_in1k",
    "resnet152.a1h_in1k",
    "resnet152.a2_in1k",
    "resnet152.a3_in1k",
    "vit_base_patch16_224.mae",
    "vit_huge_patch14_224.mae",
    "vit_large_patch16_224.mae",
    "vit_small_patch16_224.dino",
    "vit_base_patch16_224.dino",
    "vit_base_patch16_224.sam_in1k",
    "vit_base_patch16_224.orig_in21k_ft_in1k",
    "vgg19.tv_in1k",
    "vgg16.tv_in1k",
    "vgg13.tv_in1k",
    "vgg11.tv_in1k",
    "mixer_b16_224.goog_in21k_ft_in1k",
    "mixer_l16_224.goog_in21k_ft_in1k",
    "mixer_b16_224.goog_in21k",
    "mixer_l16_224.goog_in21k",
]


clean_model_names = {
    "resnet50.tv_in1k": "ResNet-50",
    "resnet101.tv_in1k": "ResNet-101",
    "resnet152.tv_in1k": "ResNet-152",
    "convnext_tiny.fb_in1k": "ConvNeXt-tiny",
    "convnext_small.fb_in1k": "ConvNeXt-small",
    "convnext_base.fb_in1k": "ConvNeXt-base",
    "convnext_large.fb_in1k": "ConvNeXt-large",
    "convnext_tiny.in12k_ft_in1k": "ConvNeXt-tiny (IN12k,ft)",
    "convnext_tiny.fb_in22k_ft_in1k": "ConvNeXt-tiny (IN22k,ft)",
    "vit_small_patch16_224.augreg_in1k": "ViT-small",
    "vit_base_patch16_224.augreg_in1k": "ViT-base",
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k": "ViT-tiny (IN21k,ft)",
    "vit_small_patch16_224.augreg_in21k_ft_in1k": "ViT-small (IN21k,ft)",
    "vit_base_patch16_224.augreg_in21k_ft_in1k": "ViT-base (IN21k,ft)",
    "vit_large_patch16_224.augreg_in21k_ft_in1k": "ViT-large (IN21k,ft)",
    "tf_efficientnet_b0.in1k": "EfficientNet-B0",
    "tf_efficientnet_b1.in1k": "EfficientNet-B1",
    "tf_efficientnet_b2.in1k": "EfficientNet-B2",
    "tf_efficientnet_b3.in1k": "EfficientNet-B3",
    "tf_efficientnet_b4.in1k": "EfficientNet-B4",
    "tf_efficientnet_b5.in1k": "EfficientNet-B5",
    "tf_efficientnet_b0.aa_in1k": "EfficientNet-B0-AA",
    "tf_efficientnet_b1.aa_in1k": "EfficientNet-B1-AA",
    "tf_efficientnet_b2.aa_in1k": "EfficientNet-B2-AA",
    "tf_efficientnet_b3.aa_in1k": "EfficientNet-B3-AA",
    "tf_efficientnet_b4.aa_in1k": "EfficientNet-B4-AA",
    "tf_efficientnet_b5.aa_in1k": "EfficientNet-B5-AA",
    "tf_efficientnet_b0.ap_in1k": "EfficientNet-B0-AP",
    "tf_efficientnet_b1.ap_in1k": "EfficientNet-B1-AP",
    "tf_efficientnet_b2.ap_in1k": "EfficientNet-B2-AP",
    "tf_efficientnet_b3.ap_in1k": "EfficientNet-B3-AP",
    "tf_efficientnet_b4.ap_in1k": "EfficientNet-B4-AP",
    "tf_efficientnet_b5.ap_in1k": "EfficientNet-B5-AP",
    "tf_efficientnet_b0.ns_jft_in1k": "EfficientNet-B0-NS (JFT)",
    "tf_efficientnet_b1.ns_jft_in1k": "EfficientNet-B1-NS (JFT)",
    "tf_efficientnet_b2.ns_jft_in1k": "EfficientNet-B2-NS (JFT)",
    "tf_efficientnet_b3.ns_jft_in1k": "EfficientNet-B3-NS (JFT)",
    "tf_efficientnet_b4.ns_jft_in1k": "EfficientNet-B4-NS (JFT)",
    "tf_efficientnet_b5.ns_jft_in1k": "EfficientNet-B5-NS (JFT)",
    "tf_efficientnet_b5.ra_in1k": "EfficientNet-B5-RA",
    "tf_efficientnet_lite0.in1k": "EfficientNet-lite0",
    "tf_efficientnet_lite1.in1k": "EfficientNet-lite1",
    "tf_efficientnet_lite2.in1k": "EfficientNet-lite2",
    "tf_efficientnet_lite3.in1k": "EfficientNet-lite3",
    "tf_efficientnet_lite4.in1k": "EfficientNet-lite4",
    "regnety_002.pycls_in1k": "RegNet-Y 002",
    "regnety_004.pycls_in1k": "RegNet-Y 004",
    "regnety_006.pycls_in1k": "RegNet-Y 006",
    "regnety_008.pycls_in1k": "RegNet-Y 008",
    "regnety_016.pycls_in1k": "RegNet-Y 016",
    "regnety_032.pycls_in1k": "RegNet-Y 032",
    "regnety_040.pycls_in1k": "RegNet-Y 040",
    "regnety_064.pycls_in1k": "RegNet-Y 064",
    "regnety_080.pycls_in1k": "RegNet-Y 080",
    "regnety_120.pycls_in1k": "RegNet-Y 120",
    "regnety_160.pycls_in1k": "RegNet-Y 160",
    "regnety_320.pycls_in1k": "RegNet-Y 320",
    "resnet50.a1_in1k": "ResNet-50 (A1)",
    "resnet50.a1h_in1k": "ResNet-50 (A1H)",
    "resnet50.a2_in1k": "ResNet-50 (A2)",
    "resnet50.a3_in1k": "ResNet-50 (A3)",
    "resnet50.b1k_in1k": "ResNet-50 (B1K)",
    "resnet50.b2k_in1k": "ResNet-50 (B2K)",
    "resnet50.c1_in1k": "ResNet-50 (C1)",
    "resnet50.c2_in1k": "ResNet-50 (C2)",
    "resnet50.d_in1k": "ResNet-50 (D)",
    "resnet50.ra_in1k": "ResNet-50 (RA)",
    "resnet50.ram_in1k": "ResNet-50 (AugMix)",
    "resnet101.a1_in1k": "ResNet-101 (A1)",
    "resnet101.a1h_in1k": "ResNet-101 (A1H)",
    "resnet101.a2_in1k": "ResNet-101 (A2)",
    "resnet101.a3_in1k": "ResNet-101 (A3)",
    "resnet152.a1_in1k": "ResNet-152 (A1)",
    "resnet152.a1h_in1k": "ResNet-152 (A1H)",
    "resnet152.a2_in1k": "ResNet-152 (A2)",
    "resnet152.a3_in1k": "ResNet-152 (A3)",
    "vit_base_patch16_224.mae": "ViT-base (patch 16, MAE)",
    "vit_huge_patch14_224.mae": "ViT-huge (patch 14, MAE)",
    "vit_large_patch16_224.mae": "ViT-large (patch 14, MAE)",
    "vit_small_patch16_224.dino": "ViT-small (DINO)",
    "vit_base_patch16_224.dino": "ViT-base (DINO)",
    "vit_base_patch16_224.sam_in1k": "ViT-base (SAM)",
    "vit_base_patch16_224.orig_in21k_ft_in1k": "ViT-base (IN21k orig,ft)",
    "vgg19.tv_in1k": "VGG19",
    "vgg16.tv_in1k": "VGG16",
    "vgg13.tv_in1k": "VGG13",
    "vgg11.tv_in1k": "VGG11",
    "mixer_b16_224.goog_in21k_ft_in1k": "MLP-MixereB (IN21k,ft)",
    "mixer_l16_224.goog_in21k_ft_in1k": "MLP-Mixer-L (IN21k,ft)",
    "mixer_b16_224.goog_in21k": "MLP-Mixer-B (IN21k)",
    "mixer_l16_224.goog_in21k": "MLP-Mixer-L (IN21k)",
}

imagenet_results = pd.read_csv("timm_results/results-imagenet.csv")
imagenet_results = (
    imagenet_results.loc[imagenet_results["model"].isin(model_list), ["model", "top1", "param_count"]]
    .astype({"top1": "Float32", "param_count": "Float32"})
    .reset_index(drop=True)
)
imagenet_a_results = pd.read_csv("timm_results/results-imagenet-a.csv")
imagenet_a_results = (
    imagenet_a_results.loc[imagenet_a_results["model"].isin(model_list), ["model", "top1", "param_count"]]
    .astype({"top1": "Float32", "param_count": "Float32"})
    .reset_index(drop=True)
)
imagenet_r_results = pd.read_csv("timm_results/results-imagenet-r.csv")
imagenet_r_results = (
    imagenet_r_results.loc[imagenet_r_results["model"].isin(model_list), ["model", "top1", "param_count"]]
    .astype({"top1": "Float32", "param_count": "Float32"})
    .reset_index(drop=True)
)
imagenet_real_results = pd.read_csv("timm_results/results-imagenet-real.csv")
imagenet_real_results = (
    imagenet_real_results.loc[imagenet_real_results["model"].isin(model_list), ["model", "top1", "param_count"]]
    .astype({"top1": "Float32", "param_count": "Float32"})
    .reset_index(drop=True)
)
imagenet_v2_results = pd.read_csv("timm_results/results-imagenet-v2.csv")
imagenet_v2_results = (
    imagenet_v2_results.loc[imagenet_v2_results["model"].isin(model_list), ["model", "top1", "param_count"]]
    .astype({"top1": "Float32", "param_count": "Float32"})
    .reset_index(drop=True)
)

num_params = dict(zip(imagenet_results["model"], imagenet_results["param_count"] * 10e6))
imagenet_results = dict(zip(imagenet_results["model"], imagenet_results["top1"]))
imagenet_a_results = dict(zip(imagenet_a_results["model"], imagenet_a_results["top1"]))
imagenet_r_results = dict(zip(imagenet_r_results["model"], imagenet_r_results["top1"]))
imagenet_real_results = dict(zip(imagenet_real_results["model"], imagenet_real_results["top1"]))
imagenet_v2_results = dict(zip(imagenet_v2_results["model"], imagenet_v2_results["top1"]))


# patch for SSL or IN21k models
ssl_num_params = {
    "vit_base_patch16_224.mae": 85.8 * 10e6,
    "vit_huge_patch14_224.mae": 630.8 * 10e6,
    "vit_large_patch16_224.mae": 303.3 * 10e6,
    "vit_small_patch16_224.dino": 21.7 * 10e6,
    "vit_base_patch16_224.dino": 85.8 * 10e6,
    "mixer_b16_224.goog_in21k": 59.88 * 10e6,
    "mixer_l16_224.goog_in21k": 208.2 * 10e6,
}
num_params.update(ssl_num_params)

nan_dict = {k: float("nan") for k in ssl_num_params.keys()}

imagenet_results.update(nan_dict)
imagenet_a_results.update(nan_dict)
imagenet_r_results.update(nan_dict)
imagenet_real_results.update(nan_dict)
imagenet_v2_results.update(nan_dict)


def wandb2pd(exp_runs):
    df = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
    summary_df = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
    config_df = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
    name_df = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)

    summary = []
    config = []
    name = []
    counter_of_run = 0
    num_sample = 1000
    for exp in exp_runs:
        if counter_of_run > num_sample:
            break
        counter_of_run += 1

        summary.append(exp.summary._json_dict)
        config.append({k: v for k, v in exp.config.items() if not k.startswith("_")})
        name.append(exp.name)

    summary_df = pd.DataFrame.from_records(summary)
    config_df = pd.DataFrame.from_records(config)

    name_df = pd.DataFrame({"name": name})
    df = pd.concat([name_df, config_df, summary_df], axis=1)
    return df


def get_wandb_logs(wandb_path):
    api = wandb.Api()

    exp_runs = api.runs(
        path=f"{wandb_path}",
        filters={"state": "finished"},
        order="-summary_metrics.avg_val_acc",
    )

    df = wandb2pd(exp_runs)
    return df
