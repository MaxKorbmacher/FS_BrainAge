# QC for Freesurfer versions
# Max Korbmacher, 03. April 2024
#
#
# Overview:
# 1) check the smallest correlations
# 1.1) whole sample
# 1.2) age-stratification
# 2) check the most important features closely
# 3) check age correlations at the feature level
# 4) PCA sorting into thickness, area, volume
# 5) check how version mixing influences predictions
#
# load pkgs
if (!require("pacman")) install.packages("pacman")
pacman::p_load(lme4, nlme, ggplot2, tidyverse, lm.beta, remotes, ggpubr, 
               grid, lmtest, car, lmtest,lmeInfo,lmerTest,sjstats,effsize,Rmpfr,
               ggrepel,PASWR2, reshape2, xgboost, confintr, factoextra, mgcv, 
               itsadug, Metrics, ggpointdensity, viridis, MuMIn,hrbrthemes,
               ggridges, egg, pheatmap, ggtext, RColorBrewer, pmsampsize,
               Metrics, dataPreparation,cocor, update = F)
# load data
FS5_train = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/train.csv")
FS5_test = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/test.csv")
FS7_train = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/train.csv")
FS7_test = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/test.csv")
#
#
# 1) check the smallest correlations ####
#
#
# first remove demographics
FS5_train1 = FS5_train %>% select(-eid, -Age, -Sex, -Scanner)
FS5_test1 = FS5_test %>% select(-eid, -Age, -Sex, -Scanner)
FS7_train1 = FS7_train %>% select(-eid, -Age, -Sex, -Scanner)
FS7_test1 = FS7_test %>% select(-eid, -Age, -Sex, -Scanner)
#
#
# correlate
cor_vec_train = c()
cor_vec_test = c()
for (i in 1:ncol(FS5_train1)){
  cor_vec_train[i] = cor(FS5_train1[,i],FS7_train1[,i])
  cor_vec_test[i] = cor(FS5_test1[,i],FS7_test1[,i])
}
# estimate Pearson's correlation coefficients for feature estimates from FS5 and FS7
cor.dat = data.frame(Correlation = c(cor_vec_train,cor_vec_test), 
                     Data = c(replicate(length(cor_vec_train),"train"), 
                              replicate(length(cor_vec_test), "test")))
# make a vector which labels the features (which are correlated between FS5 & FS7)
cor.dat$Feature = names(FS5_train1)
# separate train from test data correlations
dt_train = cor.dat %>% filter(Data == "train")
dt_test = cor.dat %>% filter(Data == "test")
# sort the data frames
dt_train = dt_train[order(dt_train$Correlation),]
dt_test = dt_test[order(dt_test$Correlation),]
# display the smallest correlations
head(dt_train)
head(dt_test)
print("Enthorinal cortex as well as temporal and frontal poles' volume and surface area are least corresponding between versions.")
# write these data frames
write.csv(dt_train,"/Users/max/Documents/Projects/FS_brainage/results/feature_correlations_train.csv")
write.csv(dt_test,"/Users/max/Documents/Projects/FS_brainage/results/feature_correlations_test.csv")
# Now, make a plot for the distributions
cor.dat = data.frame(Correlation = c(cor_vec_train,cor_vec_test), 
                     Data = c(replicate(length(cor_vec_train),"train"), 
                              replicate(length(cor_vec_test), "test")))
# make a vector which labels the features (which are correlated between FS5 & FS7)
cor.dat$Data = as.factor(cor.dat$Data)
plot2 = cor.dat %>% 
  #rename("Data" = "Data", "uncorrected" = "BAGu") %>%
  melt(id.vars = "Data") %>% ggplot(aes(x = value, y = variable, fill = `Data`)) +
  geom_density_ridges(aes(fill = `Data`)) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  stat_density_ridges(quantile_lines = TRUE, quantiles = 0.5, alpha = .6) +
  theme_classic(base_size = 12) + theme(legend.position = "bottom") + ylab("") + xlab("Pearson's Correlation Coefficient") + 
  ggtitle("FreeSurfer v5 and v7 Metric Correlations per Data Set") + xlim(.5,1.1) +
  theme(axis.text.y=element_blank(), 
        axis.ticks.y=element_blank()) 
plot2 = plot2 + annotate("text", label = (paste("Training mean r = ", round(mean(cor_vec_train),3), sep = "")), x = .5, y = 12.5, size = 4, hjust = 0)
plot2 = plot2 + annotate("text", label = (paste("Test mean r = ", round(mean(cor_vec_test),3), sep = "")), x = .5, y = 10, size = 4, hjust = 0)
#
# Now, we can also check whether the correlation structure looks different when age-stratifying

corstrat = function(FS5_training_data, FS5_test_data, FS7_training_data, FS7_test_data){
  # correlate
  cor_vec_train = c()
  cor_vec_test = c()
  for (i in 1:ncol(FS5_training_data)){
    cor_vec_train[i] = cor(FS5_training_data[,i],FS7_training_data[,i])
    cor_vec_test[i] = cor(FS5_test_data[,i],FS7_test_data[,i])
  }
  # estimate Pearson's correlation coefficients for feature estimates from FS5 and FS7
  cor.dat = data.frame(Correlation = c(cor_vec_train,cor_vec_test), 
                       Data = c(replicate(length(cor_vec_train),"train"), 
                                replicate(length(cor_vec_test), "test")))
  # make a vector which labels the features (which are correlated between FS5 & FS7)
  cor.dat$Data = as.factor(cor.dat$Data)
  plot2 = cor.dat %>% 
    #rename("Data" = "Data", "uncorrected" = "BAGu") %>%
    melt(id.vars = "Data") %>% ggplot(aes(x = value, y = variable, fill = `Data`)) +
    geom_density_ridges(aes(fill = `Data`)) +
    scale_fill_manual(values = c("#E69F00","#56B4E9")) +
    stat_density_ridges(quantile_lines = TRUE, quantiles = 0.5, alpha = .6) +
    theme_classic(base_size = 12) + theme(legend.position = "bottom") + ylab("") + xlab("Pearson's Correlation Coefficient") + 
    xlim(.5,1.1) +
    theme(axis.text.y=element_blank(), 
          axis.ticks.y=element_blank()) 
  plot2 = plot2 + annotate("text", label = (paste("Training mean r = ", round(mean(cor_vec_train, na.rm=T),3), sep = "")), x = .5, y = 12.5, size = 4, hjust = 0)
  plot2 = plot2 + annotate("text", label = (paste("Test mean r = ", round(mean(cor_vec_test, na.rm=T),3), sep = "")), x = .5, y = 11.5, size = 4, hjust = 0)
  #plot2 = plot2 + annotate("text", label = (paste("training N = ", nrow(FS5_training_data),"; test N = ", nrow(FS5_test_data), sep = "")), x = .5, y = 10.5, size = 4, hjust = 0)
  return(plot2)
}
# Now, we can bin the data into 3 even age bins, which makes sense in an age range of around 50 to 80 (a bin for every decade)
# (Remember, age distributions are the same for FS5&7, as these are the same participants.)
FS5_train %>% mutate(Age_Groups = ntile(Age, n=3)) %>% group_by(Age_Groups) %>% summarise(M = mean(Age))
FS5_test %>% mutate(Age_Groups = ntile(Age, n=3)) %>% group_by(Age_Groups) %>% summarise(M = mean(Age))
# create lists of data frames based on age bins and remove variables which are not supposed to be correlated
FS5_train2 = FS5_train %>% mutate(Age_Groups = ntile(Age, n=3)) %>% group_by(Age_Groups) %>% select(-eid, -Age, -Sex, -Scanner, -Age_Groups)
FS5_test2 = FS5_test %>% mutate(Age_Groups = ntile(Age, n=3)) %>% group_by(Age_Groups) %>% select(-eid, -Age, -Sex, -Scanner, -Age_Groups)
FS7_train2 = FS7_train %>% mutate(Age_Groups = ntile(Age, n=3)) %>% group_by(Age_Groups) %>% select(-eid, -Age, -Sex, -Scanner, -Age_Groups)
FS7_test2 = FS7_test %>% mutate(Age_Groups = ntile(Age, n=3)) %>% group_by(Age_Groups) %>% select(-eid, -Age, -Sex, -Scanner, -Age_Groups)
FS5_train2 = group_split(FS5_train2)
FS7_train2 = group_split(FS7_train2)
FS5_test2 = group_split(FS5_test2)
FS7_test2 = group_split(FS7_test2)
# run the age splits though a loop
stratplots = list()
for (i in 1:length(FS5_train2)){
  stratplots[[i]] = corstrat(FS5_train2[[i]], FS5_test2[[i]], FS7_train2[[i]], FS7_test2[[i]])
}
cor_strat = ggpubr::ggarrange(plotlist = stratplots, common.legend = T, legend = "bottom", labels = c("Younger than 60", "60-70", "Older than 70"), hjust = -1)
cor_strat = annotate_figure(cor_strat, top = text_grob("Age-Stratefied FreeSurfer v5 and v7 Metric Correlations per Data Set", color = "black", face = "bold", size = 14))
ggsave(filename = "/Users/max/Documents/Projects/FS_brainage/results/CorStrat.pdf",plot = cor_strat, height = 8, width = 10)
rm(cor_strat, FS5_train2, FS5_test2, FS7_train2, FS7_test2)
#
#
#
#
#
# 2) check the most important features closely ####
print("Here we focus on Lasso feature importance and closely examine the most contributing features to the predictions based on permutation feature importance.")
print("Note: This concerns only training data.")
######### As reference, here are the feature rankings from the model output: 
# For FS7:
# rh_inferiorparietal_volume 0.488005 +/- 0.024075
# rh_inferiorparietal_area 0.446060 +/- 0.015865
# lh_superiorfrontal_volume 0.298037 +/- 0.008935
# lh_lateralorbitofrontal_area 0.188400 +/- 0.005097
# lh_superiorfrontal_area 0.118241 +/- 0.007743
# lh_insula_volume 0.103468 +/- 0.005793
# rh_superiortemporal_volume 0.098476 +/- 0.008758
# lh_superiorparietal_volume 0.092775 +/- 0.008547
# lh_rostralmiddlefrontal_volume 0.091777 +/- 0.004123
# lh_superiorparietal_area 0.088267 +/- 0.006056
#
# For FS5:
# rh_inferiorparietal_area 0.347284 +/- 0.009272
# rh_inferiorparietal_volume 0.336252 +/- 0.009485
# lh_superiorfrontal_volume 0.221772 +/- 0.008119
# lh_inferiortemporal_area 0.149656 +/- 0.006959
# lh_lateralorbitofrontal_area 0.089711 +/- 0.007426
# rh_superiortemporal_area 0.085009 +/- 0.005798
# rh_superiortemporal_volume 0.075111 +/- 0.004937
# lh_inferiortemporal_volume 0.071722 +/- 0.004952
# lh_superiortemporal_volume 0.064135 +/- 0.003839
# lh_superiorparietal_area 0.059248 +/- 0.003343
#
# we start with the top ranking features for the FS7 trained model
dt_train %>% filter(Feature %in% c("rh_inferiorparietal_volume","rh_inferiorparietal_area",
                                 "lh_superiorfrontal_volume","lh_lateralorbitofrontal_area",
                                 "lh_superiortemporal_volume","lh_insula_volume",
                                 "rh_superiortemporal_volume", "lh_superiorparietal_volume",
                                 "lh_rostralmiddlefrontal_volume","lh_superiorparietal_area"))
print("All features have a correspondance between versions of r > .9.")
# now, FS5 most important features
dt_train %>% filter(Feature %in% c("rh_inferiorparietal_volume","rh_inferiorparietal_area",
                                   "lh_superiorfrontal_volume", "lh_inferiortemporal_area",
                                   "lh_lateralorbitofrontal_area", "lh_inferiortemporal_volume",
                                   "rh_superiortemporal_area","rh_superiortemporal_volume",
                                   "rh_lateraloccipital_volume","lh_superiorfrontal_area"))
print("All features have a correspondance between versions of r > .9.")
#
#
#
#
#
#
#
# 3) check age correlations at the feature level ####
cor_vec_train_FS5 = c()
cor_vec_train_FS7 = c()
cor_vec_test_FS5 = c()
cor_vec_test_FS7 = c()
for (i in 1:ncol(FS5_train1)){
  cor_vec_train_FS5[i] = cor(FS5_train1[,i],FS5_train$Age)
  cor_vec_test_FS5[i] = cor(FS5_test1[,i],FS5_test$Age)
  cor_vec_test_FS7[i] = cor(FS5_train1[,i],FS7_train$Age)
  cor_vec_train_FS7[i] = cor(FS7_test1[,i],FS7_test$Age)
}
# make data frames for FS5 and FS7 train & test age-feature associations
cor.dat.FS5 = data.frame(Correlation = c(cor_vec_train_FS5,cor_vec_test_FS5), 
                     Data = as.factor(c(replicate(length(cor_vec_train_FS5),"train"), 
                              replicate(length(cor_vec_test_FS5), "test")))) # Feature = names(FS5_train1
cor.dat.FS7 = data.frame(Correlation = c(cor_vec_train_FS7,cor_vec_test_FS7), 
                         Data = as.factor(c(replicate(length(cor_vec_train_FS7),"train"), 
                                  replicate(length(cor_vec_test_FS7), "test")))) #Feature = names(FS7_train1)
# plots
plot3.1 = cor.dat.FS5 %>% 
  #rename("Data" = "Data", "uncorrected" = "BAGu") %>%
  reshape::melt(id.vars = "Data") %>% ggplot(aes(x = value, y = variable, fill = `Data`)) +
  geom_density_ridges(aes(fill = `Data`)) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  stat_density_ridges(quantile_lines = TRUE, quantiles = 0.5, alpha = .6) +
  theme_classic(base_size = 12) + theme(legend.position = "bottom") + ylab("") + xlab("Pearson's Correlation Coefficient") + 
  ggtitle("FS5 Metric-Age Correlations") + xlim(-.7,.3) +
  theme(axis.text.y=element_blank(), 
        axis.ticks.y=element_blank()) 
plot3.1 = plot3.1 + annotate("text", label = (paste("Training mean r = ", round(mean(cor_vec_train_FS5),3), sep = "")), x = -0.7, y = 6.5, size = 4, hjust = 0)
plot3.1 = plot3.1 + annotate("text", label = (paste("Test mean r = ", round(mean(cor_vec_test_FS5),3), sep = "")), x = -.7, y = 5.5, size = 4, hjust = 0)
plot3.2 = cor.dat.FS7 %>% 
  #rename("Data" = "Data", "uncorrected" = "BAGu") %>%
  reshape::melt(id.vars = "Data") %>% ggplot(aes(x = value, y = variable, fill = `Data`)) +
  geom_density_ridges(aes(fill = `Data`)) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  stat_density_ridges(quantile_lines = TRUE, quantiles = 0.5, alpha = .6) +
  theme_classic(base_size = 12) + theme(legend.position = "bottom") + ylab("") + xlab("Pearson's Correlation Coefficient") + 
  ggtitle("FS7 Metric-Age Correlations") + xlim(-.7,.3) +
  theme(axis.text.y=element_blank(), 
        axis.ticks.y=element_blank()) 
plot3.2 = plot3.2 + annotate("text", label = (paste("Training mean r = ", round(mean(cor_vec_train_FS7),3), sep = "")), x = -0.7, y = 6.5, size = 4, hjust = 0)
plot3.2 = plot3.2 + annotate("text", label = (paste("Test mean r = ", round(mean(cor_vec_test_FS7),3), sep = "")), x = -.7, y = 5.5, size = 4, hjust = 0)
plot3_top = ggpubr::ggarrange(plot3.1, plot3.2, ncol = 2, nrow = 1, common.legend = T, legend = "bottom")
#
# Finally compare the means of the correlations
# start with training data
t.test(cor.dat.FS5[cor.dat.FS5$Data %in% "train",]$Correlation,cor.dat.FS7[cor.dat.FS7$Data %in% "train",]$Correlation, paired = T)
# Cohen's d
(mean(cor.dat.FS7[cor.dat.FS7$Data %in% "train",]$Correlation) - mean(cor.dat.FS5[cor.dat.FS5$Data %in% "train",]$Correlation) ) / sd(cor.dat.FS7[cor.dat.FS7$Data %in% "train",]$Correlation-cor.dat.FS5[cor.dat.FS5$Data %in% "train",]$Correlation)
# then testing data
t.test(cor.dat.FS5[cor.dat.FS5$Data %in% "test",]$Correlation,cor.dat.FS7[cor.dat.FS7$Data %in% "test",]$Correlation, paired = T)
# Cohen's d
(mean(cor.dat.FS7[cor.dat.FS7$Data %in% "test",]$Correlation) - mean(cor.dat.FS5[cor.dat.FS5$Data %in% "test",]$Correlation) ) / sd(cor.dat.FS7[cor.dat.FS7$Data %in% "test",]$Correlation-cor.dat.FS5[cor.dat.FS5$Data %in% "test",]$Correlation)
#
#
#
#
# 4) PCA sorting into thickness, area, volume ####
# run PCA per data frame
res.pca1 <- prcomp(FS5_train1, scale = TRUE)
res.pca2 <- prcomp(FS5_test1, scale = TRUE)
res.pca3 <- prcomp(FS7_train1, scale = TRUE)
res.pca4 <- prcomp(FS7_test1, scale = TRUE)
# visualise
p1 = fviz_eig(res.pca1, geom="line", addlabels=F, hjust = -0.3,
         linecolor ="black", xlab = "Dimensions in FS5 Training Data", main = "", ylab = "Variance Explained") + theme_minimal()
p2 = fviz_eig(res.pca2, geom="line", addlabels=F, hjust = -0.3,
              linecolor ="black", xlab = "Dimensions in FS5 Test Data", main = "", ylab = "Variance Explained") + theme_minimal()
p3 = fviz_eig(res.pca3, geom="line", addlabels=F, hjust = -0.3,
              linecolor ="black", xlab = "Dimensions in FS7 Training Data", main = "", ylab = "Variance Explained") + theme_minimal()
p4 = fviz_eig(res.pca4, geom="line", addlabels=F, hjust = -0.3,
              linecolor ="black", xlab = "Dimensions in FS7 Test Data", main = "", ylab = "Variance Explained") + theme_minimal()
# another way to visualise:
# p1 = fviz_eig(res.pca1, addlabels=TRUE, hjust = -0.3,
#          linecolor ="red", xlab = "Dimensions in FS5 Training Data", main = "") + theme_minimal()
# p2 = fviz_eig(res.pca2, addlabels=TRUE, hjust = -0.3,
#               linecolor ="red", xlab = "Dimensions in FS5 Test Data", main = "") + theme_minimal()
# p3 = fviz_eig(res.pca3, addlabels=TRUE, hjust = -0.3,
#               linecolor ="red", xlab = "Dimensions in FS7 Training Data", main = "") + theme_minimal()
# p4 = fviz_eig(res.pca4, addlabels=TRUE, hjust = -0.3,
#               linecolor ="red", xlab = "Dimensions in FS7 Test Data", main = "") + theme_minimal()
# now, we got the first panel of plot 4
plot3_middle = ggarrange(p1,p3,p2,p4)
#
#
#
#
#

# add groups (vol,thick,area) to data and visualize weights for first two components per dataset
rel.w = function(DATASET, YLAB){
  conts = data.frame(Contribution_PC1 = get_pca_var(DATASET)$contrib[,1],Contribution_PC2 = get_pca_var(DATASET)$contrib[,2], Group = c("Area", "Thickness", "Volume"))
  #conts$Variable = row.names(conts) # variable names can be added, if wished
  cont_thickness = conts$Contribution_PC1
  conts_dat = data.frame(conts %>% group_by(Group) %>% summarise(PC1 = sum(Contribution_PC1)/100, PC2 = sum(Contribution_PC2)/100))
  conts_dat = reshape::melt(conts_dat, by = "Group")
  plot = ggplot(conts_dat, aes(x = variable, y = value, fill = Group)) +
    geom_col(colour = "black") +
    scale_fill_manual(values = brewer.pal(9,"Pastel1")[1:9]) + 
    ylab(YLAB) + theme_bw() +
    theme(legend.position='bottom', legend.title = element_blank()) + xlab("")
    #scale_x_discrete(labels=c("PC1" = "Principal Component 1", "PC2" = "Principal Component 2"))
  #return(plot)
}
panel1 = rel.w(res.pca1, "Relative Contribution FS5 Training")
panel2 = rel.w(res.pca2, "Relative Contribution FS5 Test")
panel3 = rel.w(res.pca1, "Relative Contribution FS7 Training")
panel4 = rel.w(res.pca2, "Relative Contribution FS7 Test")
plot3_bottom = ggpubr::ggarrange(panel1, panel2, panel3, panel4, nrow = 1, common.legend = T, legend = "bottom")
# add panels together into single fig
plot2 = ggpubr::ggarrange(plot2, plot3_top,plot3_middle, plot3_bottom, ncol = 1, labels = c("a","b","c", "d"))
ggsave(filename = "/Users/max/Documents/Projects/FS_brainage/results/Plot2.pdf",plot = plot2, height = 15, width = 10)
#
#
#
#
#

