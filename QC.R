# QC for Freesurfer versions
# Max Korbmacher, 03. April 2024
#
#
# Overview:
# 1) Remove +/-3SDs from the mean for each metric
# 2) check the smallest correlations
# 3) check the most important features closely
# 4) check age correlations at the feature level
# 5) PCA sorting into thickness, area, volume
# 6) check how version mixing influences predictions
#
# load pkgs
if (!require("pacman")) install.packages("pacman")
pacman::p_load(lme4, nlme, ggplot2, tidyverse, lm.beta, remotes, ggpubr, 
               grid, lmtest, car, lmtest,lmeInfo,lmerTest,sjstats,effsize,Rmpfr,
               ggrepel,PASWR2, reshape2, xgboost, confintr, factoextra, mgcv, 
               itsadug, Metrics, ggpointdensity, viridis, MuMIn,hrbrthemes,
               ggridges, egg, pheatmap, ggtext, RColorBrewer, pmsampsize,
               Metrics, dataPreparation, update = F)
# load data
FS5_train = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/train.csv")
FS5_test = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/test.csv")
FS7_train = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/train.csv")
FS7_test = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/test.csv")
#
# 1) Remove +/-5SDs from the mean for each metric
FS5_train = (remove_sd_outlier(FS5_train, n_sigmas = 5))
FS5_test = (remove_sd_outlier(FS5_test, n_sigmas = 5))


# 2) check the smallest correlations
# 3) check the most important features closely
# 4) check age correlations at the feature level
# 5) PCA sorting into thickness, area, volume
# 6) check how version mixing influences predictions