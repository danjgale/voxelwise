# voxelwise

`voxelwise` enables easy analysis of voxel-wise data from fMRI, commonly known as multivariate pattern analysis (MVPA). In these analyses, voxel patterns are extracted from experimental trials and then analyzed using pattern classification and dissimilarity analyses, which attempt to quantify the differences, or lack thereof, in voxel activity between experimental conditions.

`voxelwise` implements GLM-based approaches for pattern extraction, along with easy-to-use interfaces for pattern classification and similarity analyses.

## Pattern Extraction

Voxel patterns are extracted in a few different ways. One way to do this is selecting timepoints in each trial (e.g.,  timepoints with peak activity, predefined timepoints of interest, etc) are chosen as patterns of interests that are fed into the analyses. Alternatively, hemodynamic response functions can be fit to each trial using general linear models, resulting in voxel-wise parameter estimates for each trial. This approach has been shown to be generally superior to timepoint selection methods. However, these approaches are much more difficult to implement, especially in existing fMRI software (e.g., SPM, FSL). `voxelwise` offers an easy solution to this by providing Python implementations of two commonly-discussed approaches, `LSS` (least-squares separate) and `LSA` (least-squares-all).


## Pattern Classification

Coming soon!


## Similarity Analyses

Coming soon!


## Nilearn and Nistats

Coming soon!
