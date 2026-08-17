# -*- coding: utf-8 -*-
"""
Example: Partial Least-Squares (PLS) Regression with FoodomicsPy

Aim
---
This example demonstrates a complete but simple PLS regression workflow
using LF-NMR data.

The example shows how to:

1. Load predictor data and metadata
2. Align samples
3. Remove samples with missing response values
4. Define X and y
5. Define groups for grouped cross-validation
6. Crop the predictor range
7. Preprocess X
8. Run PLS regression
9. Inspect model statistics
10. Inspect component optimization
11. Plot PLS diagnostics
12. Plot measured vs predicted values
13. Plot residuals
14. Plot PLS coefficients
15. Plot PLS scores
16. Color samples using metadata


Expected data structure
-----------------------

Predictor data (df)
~~~~~~~~~~~~~~~~~~~

The predictor data should be a pandas DataFrame with:

    - rows = samples
    - columns = predictor variables
    - index = unique sample IDs

For spectral data, columns normally represent spectral variables such as
wavelength, wavenumber, chemical shift, or relaxation time.

Example:

                0.0     1.0     2.0     3.0     ...
    sample_01_a  2.31    2.45    2.67    2.82
    sample_01_b  2.29    2.43    2.65    2.80
    sample_02_a  3.14    3.28    3.41    3.52
    sample_02_b  3.11    3.25    3.39    3.50


Metadata (df_meta)
~~~~~~~~~~~~~~~~~~

The metadata should also be a pandas DataFrame with:

    - rows = samples
    - index = sample IDs matching df.index
    - columns = response variables and other sample information

Example:

                dry_matter    sample_type    batch
    sample_01_a    24.5       pig_manure       1
    sample_01_b    24.5       pig_manure       1
    sample_02_a    31.2       digestate        2
    sample_02_b    31.2       digestate        2


Sample IDs
~~~~~~~~~~

Sample IDs should be unique and should match between df and df_meta.

For example:

    df.index:
        sample_01_a
        sample_01_b
        sample_02_a
        sample_02_b

    df_meta.index:
        sample_01_a
        sample_01_b
        sample_02_a
        sample_02_b

The function:

    f.utils.align_data_metadata(df, df_meta)

can be used to align the two tables before analysis.


Replicates and groups
~~~~~~~~~~~~~~~~~~~~~

If several rows represent repeated measurements of the same underlying
sample, grouped cross-validation should be considered.

For example:

    sample_01_a
    sample_01_b
    sample_01_c

may represent three measurements of the same physical sample.

A group identifier can then be created by removing the replicate suffix:

    groups = X.index.str.rsplit("_", n=1).str[0]

which produces:

    sample_01
    sample_01
    sample_01

When these groups are supplied to perform_pls(), all measurements from
the same underlying sample are kept together in the same
cross-validation fold. This helps prevent data leakage.


Dataset
-------
LF-NMR data

Response
--------
Dry matter

Author
------
Umrbek Mavlanov
"""

# %% =========================================================================
# 1. IMPORT LIBRARIES
# =============================================================================

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import foodomicspy as f


# %% =========================================================================
# 2. SET PROJECT PATHS
# =============================================================================

# Main project folder
dir_project = Path(
    "/Users/pwk225/Library/CloudStorage/Dropbox/Research/Boost"
)

# Change the working directory to the project folder
os.chdir(dir_project)

# Predictor data
path_data = (
    dir_project
    / "data"
    / "LF-NMR"
    / "cleaned"
    / "LF-NMR_data.xlsx"
)

# Metadata
path_metadata = (
    dir_project
    / "data"
    / "metadata.xlsx"
)


# %% =========================================================================
# 3. LOAD DATA
# =============================================================================

df = pd.read_excel(
    path_data,
    index_col=0,
)

df_meta = pd.read_excel(
    path_metadata,
    index_col=0,
)


# %% =========================================================================
# 4. ALIGN DATA AND METADATA
# =============================================================================

# Make sure that:
#
# df.index == df_meta.index
#
# and that samples are in exactly the same order.
df, df_meta = f.utils.align_data_metadata(
    df,
    df_meta,
)


# %% =========================================================================
# 5. REMOVE SAMPLES WITH MISSING RESPONSE VALUES
# =============================================================================

# We want to predict dry matter.
#
# Samples without a measured dry_matter value cannot be used for
# supervised PLS regression. you can skip this if your data is complete

mask = df_meta["dry_matter"].notna()

df = df.loc[mask].copy()
df_meta = df_meta.loc[mask].copy()


# %% =========================================================================
# 6. CLEAN OPTIONAL METADATA --- if all your samples are labaled, skipp this 
# =============================================================================

# This column will later be used to color samples in plots.
#
# Replace missing sample types with the label "unknown".

df_meta["sample_type"] = (
    df_meta["sample_type"]
    .fillna("unknown")
)


# %% =========================================================================
# 7. EXPLORE RAW DATA - if you data is not spectra, skipp this part
# =============================================================================

# Plot a few raw LF-NMR profiles before preprocessing.

df.head().T.plot(
    figsize=(10, 6),
    legend=False,
    title="Raw LF-NMR data",
)

plt.xlabel("LF-NMR variable")
plt.ylabel("Signal intensity")
plt.tight_layout()


# %% =========================================================================
# 8. DEFINE X AND y
# =============================================================================

# X contains predictor variables.
X = df.copy()

# y contains the continuous response variable that we want to predict.
y = df_meta["dry_matter"].copy()

print("Number of samples:", X.shape[0])
print("Number of variables:", X.shape[1])

print()
print("Response:")
print(y.describe())


# %% =========================================================================
# 9. DEFINE GROUPS
# =============================================================================

# Grouped cross-validation is important when multiple measurements belong
# to the same physical/biological sample.
#
# Example sample IDs:
#
# sample01_1
# sample01_2
# sample01_3
#
# These measurements belong to the same underlying sample.
#
# The following line removes the final suffix and creates:
#
# sample01
# sample01
# sample01
#
# GroupKFold will then make sure that all measurements from the same
# physical sample stay together in either training or validation.

groups = (
    X.index
    .str.rsplit("_", n=1)
    .str[0]
)

print()
print("Number of unique groups:")
print(groups.nunique())


# %% =========================================================================
# 10. CROP THE DATA- if you dara is not spectra, skip this one
# =============================================================================

# Keep only the LF-NMR region relevant for the model.
#
# The limits depend on the dataset and analytical technique.

X = f.utils.crop_spectra(
    X,
    (0, 1000),
)

print()
print("Number of variables after cropping:")
print(X.shape[1])


# %% =========================================================================
# 11. PREPROCESS X
# =============================================================================


X_prep = f.preprocessing.scale(
    X, "auto"
) # or "pareto"


# %% =========================================================================
# 12. RUN PLS REGRESSION
# =============================================================================

result = f.chemometrics.perform_pls(
    X=X_prep,
    y=y,

    # Maximum number of latent variables to investigate
    max_components=15,

    # Five-fold cross-validation
    cv=5,

    # Keep related measurements together
    groups=groups,

    # X was already preprocessed above,
    # therefore internal sklearn scaling is disabled
    scale=False,

    # Select the model with minimum RMSECV
    selection_metric="rmsecv",
    selection_rule="minimum",
)


# %% =========================================================================
# 13. INSPECT MODEL SUMMARY
# =============================================================================

print()
print("=" * 60)
print("PLS MODEL SUMMARY")
print("=" * 60)

print(
    result["metrics"]
)


# Useful individual values can also be accessed directly.

print()
print("Selected number of latent variables:")
print(
    result["n_components"]
)

print()
print("RMSECV:")
print(
    result["rmsecv"]
)

print()
print("R²CV:")
print(
    result["r2cv"]
)

print()
print("Q²CV:")
print(
    result["q2cv"]
)


# %% =========================================================================
# 14. INSPECT COMPONENT OPTIMIZATION
# =============================================================================

# This table shows the cross-validation performance obtained using:
#
# 1 LV
# 2 LV
# 3 LV
# ...
#
# This allows us to understand why a particular number of latent
# variables was selected.

component_results = result[
    "component_results"
]

print()
print("=" * 60)
print("COMPONENT OPTIMIZATION")
print("=" * 60)

print(
    component_results
)


# %% =========================================================================
# 15. INSPECT SAMPLE-LEVEL PREDICTIONS
# =============================================================================

# Each sample has:
#
# measured response
# cross-validated prediction
# cross-validated residual
# calibration prediction
# calibration residual

predictions = result[
    "predictions"
]

print()
print("=" * 60)
print("PREDICTIONS")
print("=" * 60)

print(
    predictions.head(10)
)


# %% =========================================================================
# 16. MAIN PLS DIAGNOSTIC FIGURE
# =============================================================================

# plot_pls() creates three panels:
#
# A. RMSECV versus number of latent variables
# B. Measured versus cross-validated predicted values
# C. Cross-validated residuals

fig, axes = f.plotting.plot_pls(
    result,

    show_regression_line=False,

    # Label samples with large residuals
    label_residual_threshold=5,

    title="PLS prediction of dry matter",
)


# %% =========================================================================
# 17. MAIN PLS FIGURE COLORED BY SAMPLE TYPE
# =============================================================================

# Metadata can be used only for visualization.
#
# It does NOT affect model fitting.
#
# Because the sample IDs were preserved in perform_pls(), the plotting
# function aligns df_meta to the correct samples automatically.

fig, axes = f.plotting.plot_pls(
    result,

    metadata=df_meta,

    color_by="sample_type",

    title="PLS prediction of dry matter by sample type",
)


# %% =========================================================================
# 18. MEASURED VS PREDICTED PLOT
# =============================================================================

fig, ax = f.plotting.plot_pls_predictions(
    result,

    metadata=df_meta,

    color_by="sample_type",

    prediction="cv",

    show_identity_line=True,

    show_regression_line=False,

    show_metrics=True,

    title="Measured vs predicted dry matter",
)


# %% =========================================================================
# 19. RESIDUAL PLOT
# =============================================================================

# Residual:
#
# measured - predicted
#
# Ideally residuals should:
#
# - be distributed around zero
# - show no obvious trend
# - have approximately constant spread
#
# Large or structured residuals may indicate:
#
# - outliers
# - nonlinear relationships
# - heteroscedasticity
# - systematic differences between sample groups

fig, ax = f.plotting.plot_pls_residuals(
    result,

    metadata=df_meta,

    color_by="sample_type",

    prediction="cv",

    show_sd_lines=True,

    sd_threshold=2,

    label_residual_threshold=5,

    title="PLS residual diagnostics",
)


# %% =========================================================================
# 20. LATENT VARIABLE OPTIMIZATION PLOT
# =============================================================================

fig, ax = f.plotting.plot_pls_components(
    result,

    title="PLS latent variable selection",
)


# %% =========================================================================
# 21. REGRESSION COEFFICIENTS
# =============================================================================

# Regression coefficients indicate how strongly each X variable
# contributes to the PLS prediction.
#
# Large positive coefficients:
# variables positively associated with the response.
#
# Large negative coefficients:
# variables negatively associated with the response.
#
# Small coefficients:
# variables with relatively small direct influence in the fitted model.

fig, ax = f.plotting.plot_pls_coefficients(
    result,

    reverse_x=False,

    title="PLS regression coefficients",
)


# %% =========================================================================
# 22. PLS SCORES
# =============================================================================

# PLS scores describe sample positions in the latent-variable space.
#
# Coloring by metadata can help identify whether particular sample types
# occupy different regions of the PLS model.

# This requires at least two latent variables.

if result["n_components"] >= 2:

    fig, ax = f.plotting.plot_pls_scores(
        result,

        metadata=df_meta,

        color_by="sample_type",

        components=(1, 2),

        title="PLS scores",
    )

else:

    print()
    print(
        "PLS score plot skipped because the selected model "
        "contains fewer than two latent variables."
    )


# %% =========================================================================
# 23. ACCESS REGRESSION COEFFICIENTS DIRECTLY
# =============================================================================

coefficients = result[
    "coefficients"
]

print()
print("=" * 60)
print("REGRESSION COEFFICIENTS")
print("=" * 60)

print(
    coefficients.head(10)
)


# %% =========================================================================
# 24. OPTIONAL: IDENTIFY LARGEST RESIDUALS
# =============================================================================

largest_residuals = (
    predictions
    .assign(
        absolute_residual=lambda x:
        x["residual_cv"].abs()
    )
    .sort_values(
        "absolute_residual",
        ascending=False,
    )
)

print()
print("=" * 60)
print("SAMPLES WITH LARGEST CV RESIDUALS")
print("=" * 60)

print(
    largest_residuals.head(10)
)


# %% =========================================================================
# 25. SHOW ALL FIGURES
# =============================================================================

plt.show()


# %% =========================================================================
# 26. FINAL MESSAGE
# =============================================================================

print()
print("PLS analysis completed.")