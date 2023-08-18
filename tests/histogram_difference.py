import numpy as np
from scipy.stats import *
import seaborn as sns
import matplotlib.pyplot as plt
from appfl.misc import *
import torch
import torch.nn as nn
from math import *
import pandas as pd


pre = flatten_primal_or_dual(torch.load("model.pth"))
post = flatten_primal_or_dual(torch.load("postmodel.pth"))
sns.set_context("paper")
sns.set_style("whitegrid")
sns.kdeplot(post - pre, fill=False)
plt.legend(["Post", "Pre"])
plt.show()


error_bounds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
differences_zfp_pruning = []
differences_sz_pruning = []
for error_bound in error_bounds:
    differences_sz_pruning.append(
        flatten_primal_or_dual(torch.load("post_SZ3_CNN_%s.pt" % (str(error_bound))))
        - flatten_primal_or_dual(torch.load("pre_SZ3_CNN_%s.pt" % (str(error_bound))))
    )
    differences_zfp_pruning.append(
        flatten_primal_or_dual(torch.load("post_ZFP_CNN_%s.pt" % (str(error_bound))))
        - flatten_primal_or_dual(torch.load("pre_ZFP_CNN_%s.pt" % (str(error_bound))))
    )
df = pd.DataFrame()
for i in range(len(error_bounds)):
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "Compressor": ["SZ3"] * len(differences_sz_pruning[i]),
                    "Error Bound": [error_bounds[i]] * len(differences_sz_pruning[i]),
                    "Difference": differences_sz_pruning[i],
                }
            ),
        ]
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "Compressor": ["ZFP"] * len(differences_zfp_pruning[i]),
                    "Error Bound": [error_bounds[i]] * len(differences_zfp_pruning[i]),
                    "Difference": differences_zfp_pruning[i],
                }
            ),
        ]
    )
sns.set_context("paper")
sns.set_style("whitegrid")
plt.figure()
g = sns.FacetGrid(df, col="Error Bound", row="Compressor", sharey=False, sharex=False)


def plot_kde_and_fits(*args, **kwargs):
    data = kwargs.pop("data")
    sns.kdeplot(x="Difference", data=data, multiple="stack", fill=False, **kwargs)

    # Gaussian MLE fitting
    gaussian_params = norm.fit(data["Difference"])
    x_gaussian = np.linspace(min(data["Difference"]), max(data["Difference"]), 100)
    y_gaussian = norm.pdf(x_gaussian, *gaussian_params)
    plt.plot(x_gaussian, y_gaussian, color="red", linestyle="--")

    # Laplacian MLE fitting
    laplace_params = laplace.fit(data["Difference"])
    x_laplace = np.linspace(min(data["Difference"]), max(data["Difference"]), 100)
    y_laplace = laplace.pdf(x_laplace, *laplace_params)
    plt.plot(x_laplace, y_laplace, color="green", linestyle="--")
    global i
    i = i + 1
    if i == 8:
        plt.subplots_adjust(bottom=0.2)
        plt.legend(
            legend_lines,
            ["Gaussian", "Laplace"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
        )


# Lines for legend
from matplotlib.lines import Line2D

legend_lines = [
    Line2D([0], [0], color="red", lw=1, linestyle="--"),
    Line2D([0], [0], color="green", lw=1, linestyle="--"),
]
i = 0
# Create legend

g.map_dataframe(plot_kde_and_fits)

plt.xlabel("Weights Value")
plt.ylabel("Percentage")
plt.savefig("cnn_sz_difference.pdf")

"""
error_bounds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
pruning_thresholds = [0.1, 0.5, 0.8, 0.9, 0.95]

differences_zfp_pruning = []
differences_sz_pruning = []
for error_bound, pruning_threshold in zip(error_bounds, pruning_thresholds):
    differences_sz_pruning.append(
        flatten_primal_or_dual(
            torch.load(
                "post_SZ3_AlexNet_%s_%s.pt" % (str(error_bound), str(pruning_threshold))
            )
        )
        - flatten_primal_or_dual(
            torch.load(
                "pre_SZ3_AlexNet_%s_%s.pt" % (str(error_bound), str(pruning_threshold))
            )
        )
    )
    differences_zfp_pruning.append(
        flatten_primal_or_dual(
            torch.load(
                "post_ZFP_AlexNet_%s_%s.pt" % (str(error_bound), str(pruning_threshold))
            )
        )
        - flatten_primal_or_dual(
            torch.load(
                "pre_ZFP_AlexNet_%s_%s.pt" % (str(error_bound), str(pruning_threshold))
            )
        )
    )


sns.set_context("paper")
sns.set_style("whitegrid")
# Define the labels
labels = ["1e-1", "1e-2", "1e-3"]

# Plot the histograms and fit a Gaussian distribution
for i in range(3):
    data = differences_sz_pruning[i][differences_sz_pruning[i] != 0]
    sns.histplot(data, bins=1000, kde=True, stat="percent")

    # Calculate the MLE parameters
    mu, std = stats.laplace.fit(data)
    # Perform the Kolmogorov-Smirnov test
    d, p_value = stats.kstest(data, "laplace", args=(mu, std))

    print(f"K-S statistic: {d}")
    print(f"p-value: {p_value}")

    # Generate a range of values over which to evaluate the fitted Gaussian
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)

    # Evaluate the Gaussian PDF at the x values
    p = stats.laplace.pdf(x, mu, std)

    # Plot the fitted Gaussian
    plt.plot(x, p, "k", linewidth=1.5)

plt.legend(labels)
plt.title("Parameter Difference SZ Lossy Compression")
plt.xlabel("Parameter Difference")
plt.ylabel("Percentage")
plt.savefig("histogram_difference_sz.pdf")

error_bounds = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"]
compressors = ["SZ3", "ZFP"]
differences_zfp = []
differences_sz = []
for i, error_bound in enumerate(error_bounds):
    differences_sz.append(
        flatten_primal_or_dual(torch.load("post_SZ3_AlexNet_" + error_bound + ".pt"))
        - flatten_primal_or_dual(torch.load("pre_SZ3_AlexNet_" + error_bound + ".pt"))
    )
    differences_zfp.append(
        flatten_primal_or_dual(torch.load("post_ZFP_AlexNet_" + error_bound + ".pt"))
        - flatten_primal_or_dual(torch.load("pre_ZFP_AlexNet_" + error_bound + ".pt"))
    )


sns.set_context("paper")
sns.set_style("whitegrid")
sns.histplot(
    differences_sz[0],
    bins=1000,
    kde=True,
    stat="percent",
)
plt.legend(["1e-3"])
plt.title("Parameter Difference SZ Lossy Compression")
plt.xlabel("Parameter Difference")
plt.ylabel("Percentage")
plt.savefig("histogram_difference_sz.pdf")


# Initialize a list to store DataFrames
data_frames = []

# Loop through differences and add to the list
for i in range(len(error_bounds)):
    for compressor, differences in zip(compressors, [differences_sz, differences_zfp]):
        temp_df = pd.DataFrame()
        temp_df["Difference"] = differences[i]
        temp_df["Compressor"] = compressor
        temp_df["Error Bound"] = error_bounds[i]
        data_frames.append(temp_df)

# Concatenate all the dataframes in the list into a single DataFrame
data = pd.concat(data_frames, ignore_index=True)

# Create FacetGrid with Seaborn
sns.set_context("paper")
sns.set_style("whitegrid")
g = sns.FacetGrid(data, row="Compressor", col="Error Bound", margin_titles=True)

for compressor in compressors:
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data[data["Compressor"] == compressor],
        x="Difference",
        hue="Error Bound",
        element="step",
        stat="density",
        common_norm=False,
        kde=True,
        bins=1000,
    )
    plt.title(f"Histogram of Differences for {compressor}")
    plt.savefig("%s_distributions.pdf" % (compressor), bbox_inches="tight")



sns.set_context("paper")
sns.set_style("whitegrid")
sns.histplot(
    difference_1,
    bins=1000,
    kde=True,
    stat="probability",
)
plt.title("Model Parameter Difference Pre- and Post-Compression")
plt.xlabel("Parameter Difference")
plt.ylabel("Percentage")
plt.savefig("histogram_difference_5e-2.pdf")

plt.clf()
sns.set_context("paper")
sns.set_style("whitegrid")
sns.histplot(
    flat_params_1,
    bins=1000,
    kde=True,
    stat="probability",
)
plt.title("Model Parameter Distribution")
plt.xlabel("Parameter")
plt.ylabel("Percentage")
plt.savefig("histogram_params_5e-2.pdf")

plt.clf()
sns.set_context("paper")
sns.set_style("whitegrid")
sns.histplot(
    difference_2,
    bins=1000,
    kde=True,
    stat="probability",
)
plt.title("Model Parameter Difference Pre- and Post-Compression No Pruning")
plt.xlabel("Parameter Difference")
plt.ylabel("Percentage")
plt.savefig("histogram_difference_5e-2_no_prune.pdf")

plt.clf()
sns.set_context("paper")
sns.set_style("whitegrid")
sns.histplot(
    flat_params_2,
    bins=1000,
    kde=True,
    stat="probability",
)
plt.title("Model Parameter Distribution 5e-2 No Pruning")
plt.xlabel("Parameter")
plt.ylabel("Percentage")
plt.savefig("histogram_params_5e-2_no_prune.pdf")

plt.clf()
sns.set_context("paper")
sns.set_style("whitegrid")
sns.histplot(
    difference_3,
    bins=round(math.sqrt(difference_3.size)),
    kde=True,
)
plt.title("Model Parameter Difference Pre- and Post-Compression")
plt.xlabel("Parameter Difference")
plt.ylabel("Frequency")
plt.savefig("histogram_difference_1e-1.pdf")

"""
