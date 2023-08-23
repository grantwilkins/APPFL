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

mobilenet_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
alexnet_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

models_dict = {
    "MobileNet_V2": mobilenet_model,
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
sns.histplot(
    data=weights_df,
    x="Weight",
    hue="Model",
    bins="sqrt",
    kde=False,
    stat="density",
    palette="flare",
    legend=True,
)
plt.xlim(-0.25, 0.25)
plt.xlabel("Weight Value")
plt.ylabel("Density")
plt.savefig("histogram_difference.pdf")
