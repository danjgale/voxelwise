# voxelwise

`voxelwise` enables easy analysis of voxel-wise data from fMRI, commonly known as multivariate pattern analysis (MVPA). In these analyses, voxel patterns are extracted from experimental trials and then analyzed using pattern classification and dissimilarity analyses, which attempt to quantify the differences, or lack thereof, in voxel activity between experimental conditions.

`voxelwise` implements GLM-based approaches for pattern extraction, along with easy-to-use interfaces for pattern classification and similarity analyses.

## Pattern Extraction

Voxel patterns are extracted in a few different ways. One way to do this is selecting timepoints in each trial (e.g.,  timepoints with peak activity, predefined timepoints of interest, etc) are chosen as patterns of interests that are fed into the analyses. Alternatively, hemodynamic response functions can be fit to each trial using general linear models, resulting in voxel-wise parameter estimates for each trial. This approach has been shown to be generally superior to timepoint selection methods. However, these approaches are much more difficult to implement, especially in existing fMRI software (e.g., SPM, FSL). `voxelwise` offers an easy solution to this by providing Python implementations of three commonly-discussed approaches:

| Model                  | Class Name | Approach                                                                           | Number of GLMs         | Number of Regressors*                       | Number of Patterns per Condition |
|------------------------|------------|------------------------------------------------------------------------------------|------------------------|---------------------------------------------|----------------------------------|
| Least-Squares Unitary  | `LSU`      | Each whole condition is estimated as a separate regressor for each run             | Number of runs         | Number of conditions                        | Number of runs                   |
| Least-Squares All      | `LSA`      | Each trial is estimated as a separate regressor for each run                       | Number of runs         | Number of trials                            | Number of trials                 |
| Least-Squares Separate | `LSS`      | Each trial of each condition is estimated in a separate GLM in a one-vs-all scheme | Number of total trials | 2 (trial of interest; all remaining trials) | Number of trials                 |
\* Not including extra regressors (e.g., temporal derivatives, nuissance regressors)

`LSA` and `LSS` are prefered for pattern classification analyses, which require many trials for training. However, the patterns are generally noisier than those estimated by the aggregated approach of `LSU` because each trial is indivually estimated. This is an acceptable cost because the improvement in classification accuracy from more trials is typically greater than fewer, but less noisier, trials. Meanwhile, `LSU` is recommended for similarity analyses because it is preferable to obtain a single, clean representation of a pattern for each condition that can be compared with others. For all approaches, trial or condition patterns from each run are estimated separately in order to provide separated data for model cross validation or reliability measures.

## Pattern Classification

Coming soon!


## Similarity Analyses

Coming soon!


## Nilearn and Nistats

Coming soon!
