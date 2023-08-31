import numpy as np
from scipy.stats import *
import seaborn as sns
import matplotlib.pyplot as plt
from appfl.misc import *
import torch
import torch.nn as nn
from math import *
import pandas as pd
import torchvision.models as models


sns.set_context("paper")
sns.set_style("whitegrid")
plt.figure(figsize=(10, 4))

mobilenet_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
alexnet_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

models_dict = {
    "MobileNet-V2": mobilenet_model,
    "ResNet50": resnet_model,
    "AlexNet": alexnet_model,
}
weights_data = []

for model_name, model in models_dict.items():
    state_dict = model.named_parameters()
    flattened_weights = np.concatenate(
        [v.flatten().detach().cpu().numpy() for _, v in state_dict]
    )
    weights_data.append(
        pd.DataFrame({"Model": model_name, "Weight": flattened_weights})
    )


weights_df = pd.concat(weights_data)
g = sns.FacetGrid(
    weights_df,
    col="Model",
    col_wrap=3,
    sharex=False,
    sharey=False,
)
for (model_name, ax), bins in zip(g.axes_dict.items(), [250, 500, 1500]):
    sns.histplot(
        weights_df[weights_df["Model"] == model_name]["Weight"],
        ax=ax,
        bins=bins,
        stat="density",
    )
    ax.set_title(model_name)
    ax.set_xlabel("")
ax = g.axes
ax[0].set_xlim(-0.25, 0.25)
ax[1].set_xlim(-0.075, 0.075)
ax[1].set_xticks([-0.05, 0, 0.05])
ax[2].set_xlim(-0.075, 0.075)
ax[2].set_xticks([-0.05, 0, 0.05])
plt.ylabel("Density")
plt.savefig("weight-distribution.pdf")
