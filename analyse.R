# Brain Age from FreeSurfer v5 vs v7
# Max Korbmacher, 27 March 2024
#
# Note: due to superior prediction performance, we use Lasso
# (other models used: LightGBM, XGBoost, Support Vector Machine Regression)
# All detailed in the code below.
#
#
# load pkgs
if (!require("pacman")) install.packages("pacman")
pacman::p_load(lme4, nlme, ggplot2, tidyverse, lm.beta, remotes, ggpubr, 
               grid, lmtest, car, lmtest,lmeInfo,lmerTest,sjstats,effsize,Rmpfr,
               ggrepel,PASWR2, reshape2, xgboost, confintr, factoextra, mgcv, 
               itsadug, Metrics, ggpointdensity, viridis, MuMIn,hrbrthemes,
               ggridges, egg, pheatmap, ggtext, RColorBrewer, pmsampsize,
               Metrics,cocor, emmeans, update = F)
# read data used for demographics
FS5_train = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/train.csv")
FS5_test = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/test.csv")
FS7_train = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/train.csv")
FS7_test = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/test.csv")
# read predictions in test data
## XGB
FS5preds_XGB = read.csv("/Users/max/Documents/Projects/FS_brainage/results/FS5predictions.csv")
FS7preds_XGB = read.csv("/Users/max/Documents/Projects/FS_brainage/results/FS7predictions.csv")
## SVM
FS5preds_SVM = read.csv("/Users/max/Documents/Projects/FS_brainage/results/FS5predictions_SVM.csv")
FS7preds_SVM = read.csv("/Users/max/Documents/Projects/FS_brainage/results/FS7predictions_SVM.csv")
## Light GBM
FS5preds_LGBM = read.csv("/Users/max/Documents/Projects/FS_brainage/results/FS5_LGBM_predictions.csv")
FS7preds_LGBM = read.csv("/Users/max/Documents/Projects/FS_brainage/results/FS7_LGBM_predictions.csv")
## Lasso
FS5preds_Lasso = read.csv("/Users/max/Documents/Projects/FS_brainage/results/FS5predictions_Lasso.csv")
FS7preds_Lasso = read.csv("/Users/max/Documents/Projects/FS_brainage/results/FS7predictions_Lasso.csv")
## LM
FS5preds_LM = read.csv("/Users/max/Documents/Projects/FS_brainage/results/FS5predictions_LM.csv")
FS7preds_LM = read.csv("/Users/max/Documents/Projects/FS_brainage/results/FS7predictions_LM.csv")

# put the predictions in a list
FS5preds = list(FS5preds_XGB, FS5preds_SVM, FS5preds_LGBM, FS5preds_Lasso, FS5preds_LM)
FS7preds = list(FS7preds_XGB, FS7preds_SVM, FS7preds_LGBM, FS7preds_Lasso, FS7preds_LM)

# read training predictions
## XGB
FS5train_preds_XGB = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/results/Brainage_XGB_train.csv")
FS7train_preds_XGB = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/results/Brainage_XGB_train.csv")
## SVM
FS5train_preds_SVM = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/results/Brainage_SVM_train.csv")
FS7train_preds_SVM = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/results/Brainage_SVR_train.csv")
## LGBM
FS5train_preds_LGBM = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/results/Brainage_LGBM_train.csv")
FS7train_preds_LGBM = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/results/Brainage_LGBM_train.csv")
## Lasso
FS5train_preds_Lasso = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/results/Brainage_Lasso_train.csv")
FS7train_preds_Lasso = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/results/Brainage_Lasso_train.csv")
## LM
FS5train_preds_LM = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/results/Brainage_LM_train.csv")
FS7train_preds_LM = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/results/Brainage_LM_train.csv")
# list the predictions
FS5train_preds = list(FS5train_preds_XGB, FS5train_preds_SVM, FS5train_preds_LGBM, FS5train_preds_Lasso, FS5train_preds_LM)
FS7train_preds = list(FS7train_preds_XGB, FS7train_preds_SVM, FS7train_preds_LGBM, FS7train_preds_Lasso, FS7train_preds_LM)
#
# Finally, read the bagged/mean LM predictions across i = 1,000 iterations
# these are made on the same test data which was randomly sampled from FS5 and FS7
version_mixing = read.csv("/Users/max/Documents/Projects/FS_brainage/version_mixing/mean_predictions.csv")
############################################################################ #
############################################################################ #
################# 1) Demographics #####################
############################################################################ #
############################################################################ #
# for sample descriptor, merge dfs
demo = rbind(FS5_test, FS5_train)
# number of participants
nrow(demo)
# sex distribution
table(demo$Sex)/nrow(demo)
# age distribution
summary(demo$Age)
sd(demo$Age)
# dist across scanner sites
table(demo$Scanner)/nrow(demo)
# rm demo df
rm(demo)
# sensitivity: we assume a conservative estimate of 40% shrinkage
pmsampsize(type = "c", parameters = ncol(FS5_train), rsquared = .6, intercept = mean(FS5_train$Age), sd = sd(FS5_train$Age))
#
############################################################################ #
############################################################################ #
################# 2) Assess brain ages #####################
############################################################################ #
############################################################################ #
print("#######################################################################")
print("Preliminary assessment: check how different algorithms perform when predicting on TRAINING data")
print("#######################################################################")
mods = c("XGBoost", "SVM", "LightGBM", "Lasso", "LM")
length(FS5train_preds)
for (i in 1:length(FS5train_preds)){
  FS5_tmp = FS5train_preds[[i]]%>% select(eid,Age,pred_age_train)
  FS7_tmp = FS7train_preds[[i]]%>% select(eid,pred_age_train)
  training = merge(FS5_tmp,FS7_tmp, by = "eid")
  names(training) = c("eid", "Age", "FS5", "FS7")
  print(mods[i])
  print(cor(training))
}
print("#######################################################################")
print("We can see that LM produces the best training results.")
print("We can double check whether this holds true when predicting in independent test data.")
print("#######################################################################")
for (i in 1:length(FS5preds)){
  FS5_tmp = FS5preds[[i]]%>% select(eid,Age,FS5_FS5,FS5_FS7)
  FS7_tmp = FS7preds[[i]]%>% select(eid,FS7_FS7, FS7_FS5)
  test = merge(FS5_tmp,FS7_tmp, by = "eid")
  print(mods[i])
  print(cor(test))
}
print("#######################################################################")
print("Based on the prediction-label correlations, LM performs best.")
print("Hence, we report the results for LM.")
print("#######################################################################")
#
#
print("We plot the LM predictions")
# scatter
# make a function for plotting with flexible data frames, y-var, and labels
scatter.plot = function(data, Metric, xtext, ytext){
  ggplot(data = data, mapping = aes(x = Age, y = Metric)) +
    geom_pointdensity() +
    scale_color_viridis() +
    labs(x = xtext, y = ytext)+
    stat_cor(method = "pearson", label.x = 45, label.y = 90)+
    stat_cor(aes(label = paste(..rr.label.., ..p.label.., sep = "~`,`~")),label.x = 45, label.y = 85)+
    stat_smooth(method = "gam",formula = y ~ s(x, k = 4), col = "red")+
    labs(color="Number of neighbouring points") +
    theme_bw()
}
# training scatter plots
train_plot_FS5 = scatter.plot(FS5train_preds_LM, FS5train_preds_LM$pred_age_train, c("Training Sample Age"), c("FS5 Predicted Age"))
train_plot_FS7 = scatter.plot(FS7train_preds_LM, FS7train_preds_LM$pred_age_train, c("Training Sample Age"), c("FS7 Predicted Age"))
# test scatter plots
test_plot_FS5_FS5 = scatter.plot(FS5preds_LM, FS5preds_LM$FS5_FS5, c("Test Sample Age"), c("Predicted Age FS5 > FS5"))
test_plot_FS5_FS7 = scatter.plot(FS5preds_LM, FS5preds_LM$FS5_FS7, c("Test Sample Age"), c("Predicted Age FS5 > FS7"))
test_plot_FS7_FS5 = scatter.plot(FS7preds_LM, FS7preds_LM$FS7_FS5, c("Test Sample Age"), c("Predicted Age FS7 > FS5"))
test_plot_FS7_FS7 = scatter.plot(FS7preds_LM, FS7preds_LM$FS7_FS7, c("Test Sample Age"), c("Predicted Age FS7 > FS7"))
# arrange the panels
plot = ggpubr::ggarrange(train_plot_FS5, train_plot_FS7,
          test_plot_FS5_FS5, test_plot_FS5_FS7,
          test_plot_FS7_FS7, test_plot_FS7_FS5,
          ncol = 2,nrow = 3,
          labels = c("a", "b","c","d","e","f"),
          common.legend = T)
ggsave(filename = "/Users/max/Documents/Projects/FS_brainage/results/Plot.pdf",plot = plot, height = 9, width = 11)
#
print("#######################################################################")
print("#######################################################################")
print("Finally, we estimate a few error metrics for another layer of comparison of the predictions.")
print("#######################################################################")
print("We start with the training performance.")
print("#######################################################################")
Sex = FS5_train$Sex
Scanner = FS5_train$Scanner
FS5_train_performance = data.frame(matrix(ncol = 5,nrow = length(FS5train_preds)))
FS7_train_performance = data.frame(matrix(ncol = 5,nrow = length(FS5train_preds)))
for (i in 1:length(FS5train_preds)){
  FS5_train_performance[i,] = c(cor(FS5train_preds[[i]]$Age,FS5train_preds[[i]]$pred_age_train),
    summary(lm(FS5train_preds[[i]]$Age~FS5train_preds[[i]]$pred_age_train))$adj.r.squared,
    summary(lm(FS5train_preds[[i]]$Age~FS5train_preds[[i]]$pred_age_train+Sex+Scanner))$adj.r.squared,
    mae(FS5train_preds[[i]]$Age,FS5train_preds[[i]]$pred_age_train),
    rmse(FS5train_preds[[i]]$Age,FS5train_preds[[i]]$pred_age_train))
  FS7_train_performance[i,] = c(cor(FS7train_preds[[i]]$Age,FS7train_preds[[i]]$pred_age_train),
              summary(lm(FS7train_preds[[i]]$Age~FS7train_preds[[i]]$pred_age_train))$adj.r.squared,
              summary(lm(FS7train_preds[[i]]$Age~FS7train_preds[[i]]$pred_age_train+Sex+Scanner))$adj.r.squared,
              mae(FS7train_preds[[i]]$Age,FS7train_preds[[i]]$pred_age_train),
              rmse(FS7train_preds[[i]]$Age,FS7train_preds[[i]]$pred_age_train))
}
train_performance = rbind(FS5_train_performance, FS7_train_performance)
names(train_performance) = c("R", "adjR2","adjR2corrected","MAE","RMSE")
train_performance$FSversion = c(5,5,5,5,5,7,7,7,7,7)
train_performance$Model = c("XGBoost", "SVM", "LightGBM", "Lasso", "LM")
LM = train_performance %>% filter(Model == "LM")
#train_performance = train_performance %>% filter(!Model == "LM")
write.csv(file = "/Users/max/Documents/Projects/FS_brainage/results/train_performance.csv", train_performance)
print("#######################################################################")
print("#######################################################################")
print("Then we look at the test performance across models.")
print("#######################################################################")
Sex = FS5_test$Sex
Scanner = FS5_test$Scanner
FS5toFS5 = data.frame(matrix(ncol = 5,nrow = length(FS5preds)))
FS5toFS7 = FS5toFS5
FS7toFS7 = FS5toFS5
FS7toFS5 = FS5toFS5
for (i in 1:length(FS5preds)){
  FS5toFS5[i,] = c(cor(FS5preds[[i]]$Age,FS5preds[[i]]$FS5_FS5),
                                summary(lm(FS5preds[[i]]$Age~FS5preds[[i]]$FS5_FS5))$adj.r.squared,
                                summary(lm(FS5preds[[i]]$Age~FS5preds[[i]]$FS5_FS5+Sex+Scanner))$adj.r.squared,
                                mae(FS5preds[[i]]$Age,FS5preds[[i]]$FS5_FS5),
                                rmse(FS5preds[[i]]$Age,FS5preds[[i]]$FS5_FS5))
  FS5toFS7[i,] = c(cor(FS5preds[[i]]$Age,FS5preds[[i]]$FS5_FS7),
                   summary(lm(FS5preds[[i]]$Age~FS5preds[[i]]$FS5_FS7))$adj.r.squared,
                   summary(lm(FS5preds[[i]]$Age~FS5preds[[i]]$FS5_FS7+Sex+Scanner))$adj.r.squared,
                   mae(FS5preds[[i]]$Age,FS5preds[[i]]$FS5_FS7),
                   rmse(FS5preds[[i]]$Age,FS5preds[[i]]$FS5_FS7))
  FS7toFS5[i,] = c(cor(FS7preds[[i]]$Age,FS7preds[[i]]$FS7_FS5),
                                summary(lm(FS7preds[[i]]$Age~FS7preds[[i]]$FS7_FS5))$adj.r.squared,
                                summary(lm(FS7preds[[i]]$Age~FS7preds[[i]]$FS7_FS5+Sex+Scanner))$adj.r.squared,
                                mae(FS7preds[[i]]$Age,FS7preds[[i]]$FS7_FS5),
                                rmse(FS7preds[[i]]$Age,FS7preds[[i]]$FS7_FS5))
  FS7toFS7[i,] = c(cor(FS7preds[[i]]$Age,FS7preds[[i]]$FS7_FS7),
                   summary(lm(FS7preds[[i]]$Age~FS7preds[[i]]$FS7_FS7))$adj.r.squared,
                   summary(lm(FS7preds[[i]]$Age~FS7preds[[i]]$FS7_FS7+Sex+Scanner))$adj.r.squared,
                   mae(FS7preds[[i]]$Age,FS7preds[[i]]$FS7_FS7),
                   rmse(FS7preds[[i]]$Age,FS7preds[[i]]$FS7_FS7))
}
test_performance = rbind(FS5toFS5, FS5toFS7, FS7toFS5, FS7toFS7)
names(test_performance) = c("R", "adjR2","adjR2corrected","MAE","RMSE")
test_performance$Predictions = c(replicate(length(FS5preds),"FS5toFS5"), replicate(length(FS5preds),"FS5toFS7"), replicate(length(FS5preds),"FS7toFS5"), replicate(length(FS5preds),"FS7toFS7"))
test_performance$Model = c("XGBoost", "SVM", "LightGBM", "Lasso", "LM")
LM_test = test_performance %>% filter(Model == "LM")
#test_performance = test_performance %>% filter(Model == "LM")
write.csv(file = "/Users/max/Documents/Projects/FS_brainage/results/test_performance.csv", test_performance)
l1 = train_performance %>% filter(Model=="Lasso")
l2 = test_performance %>% filter(Model=="Lasso")
names(l2) = names(l1)
lasso_perf = rbind(l1,l2)
write.csv(file = "/Users/max/Documents/Projects/FS_brainage/results/lasso_perf.csv", lasso_perf)
#
# write an extra file for LM (included post hoc, based on an excellen reviewer comment)
names(LM) = names(LM_test)
LM = rbind(LM,LM_test)
LM$Model = c("train", "train", "test", "test", "test", "test")



print("#######################################################################")
print("#######################################################################")
#
print("Compare correlations")
# start comparing the predictions in the training data
a = FS7train_preds_Lasso %>% select(-Age)
all_preds1 = merge(FS5train_preds_LM, a, by = "eid")
cocor(~Age + pred_age_train.x | Age + pred_age_train.y, all_preds1)

# then compare the predictions in the test set
a = FS7preds_LM %>% select(-Age)
all_preds = merge(FS5preds_LM, a, by = "eid")
# predicting within-version data
cocor(~Age + FS5_FS5 | Age + FS7_FS7, all_preds)
# predicting between versions
cocor(~Age + FS5_FS7 | Age + FS7_FS5, all_preds)
print("The evidence suggests that the differences in age-prediction associations are statistically significant.")
print("These differences show that FS5 does predict better within version, and FS7 between versions.")
#
#
#
#
print("#######################################################################")
print("#######################################################################")
print("Finally, we examine whether version mixing produces better predictions.")
print("#######################################################################")
# merge the bagged mean predictions with the single shot predictions
a = version_mixing %>% select(-age)
all_preds = merge(all_preds, a, by = "eid")
#
# We build a table which we save as a sorted object
## for that we prep a function producing a vector of all required values
cctab = function(cocor_object){
  rdiff = cocor_object@diff
  lowCI = cocor_object@zou2007$conf.int[1]
  upCI = cocor_object@zou2007$conf.int[2]
  z = cocor_object@hittner2003$statistic
  pval = cocor_object@hittner2003$p.value
  vec = c(rdiff, lowCI, upCI, z, pval)
  return(vec)
}
## which we then apply 
# print("Which model can predict FS5 and FS7 data best?")
# FS5 data
ob1 = cocor(~Age + mix2FS5 | Age + FS5_FS5, all_preds)
ob2 = cocor(~Age + mix2FS5 | Age + FS7_FS5, all_preds)
ob3 = cocor(~Age + FS5_FS5 | Age + FS7_FS5, all_preds)
# FS7 data
ob4 = cocor(~Age + mix2FS7 | Age + FS5_FS7, all_preds)
ob5 = cocor(~Age + mix2FS7 | Age + FS7_FS7, all_preds)
ob6 = cocor(~Age + FS5_FS7 | Age + FS7_FS7, all_preds)
#
# print("Extra: Which model can predict best in mixed data?")
ob7 = cocor(~Age + mix_test | Age + FS52mix, all_preds)
ob8 = cocor(~Age + mix_test | Age + FS72mix, all_preds)
ob9 = cocor(~Age + FS52mix | Age + FS72mix, all_preds)
#
# within version predictions
ob10 = cocor(~Age + mix_test | Age + FS5_FS5, all_preds)
ob11 = cocor(~Age + mix_test | Age + FS7_FS7, all_preds)
ob12 = cocor(~Age + FS7_FS7 | Age + FS5_FS5, all_preds)

# put all the vectors together
corcomp = data.frame(rbind(cctab(ob1),cctab(ob2), cctab(ob3), cctab(ob4), cctab(ob5), cctab(ob6), cctab(ob7), cctab(ob8), cctab(ob9),cctab(ob10),cctab(ob11),cctab(ob11)))
names(corcomp) = c("rdiff", "lowCI", "upCI", "Z", "p")
rownames(corcomp) = c("Mix to FS5 & FS5 to FS5", "Mix to FS5 & FS7 to FS5", "FS5 to FS5 & FS7 to FS5",
                      "Mix to FS7 & FS5 to FS7", "Mix to FS7 & FS7 to FS7", "FS5 to FS7 & FS7 to FS7",
                      "Mix to Mix & FS5 to Mix", "Mix to Mix & FS7 to Mix", "FS5 to Mix & FS7 to Mix",
                      "Mix to Mix & FS5 to FS5", "Mix to Mix & FS7 to FS7", "FS5 to FS5 & FS7 to FS7")
corcomp$p.adj = p.adjust(corcomp$p, method = "fdr")
write.csv(corcomp, "/Users/max/Documents/Projects/FS_brainage/results/corr_comp.csv")
############################################################################ #
############################################################################ #
################# 3) Assess correlation structure in the data ################
############################################################################ #
############################################################################ #
# first remove demographics
FS5_train = FS5_train %>% select(-eid, -Age, -Sex, -Scanner)
FS5_test = FS5_test %>% select(-eid, -Age, -Sex, -Scanner)
FS7_train = FS7_train %>% select(-eid, -Age, -Sex, -Scanner)
FS7_test = FS7_test %>% select(-eid, -Age, -Sex, -Scanner)
#
# just to be sure that the column names are the same
setequal(names(FS5_train), names(FS7_train))
setequal(names(FS5_test), names(FS7_test))
#
# correlate
cor_vec_train = c()
cor_vec_test = c()
for (i in 1:ncol(FS5_train)){
  cor_vec_train[i] = cor(FS5_train[,i],FS7_train[,i])
  cor_vec_test[i] = cor(FS5_test[,i],FS7_test[,i])
}
# plot the distribution of Pearson's correlation coefficients
cor.dat = data.frame(Correlation = c(cor_vec_train,cor_vec_test), 
                     Data = c(replicate(length(cor_vec_train),"train"), 
                              replicate(length(cor_vec_test), "test")))
cor.dat$Data = as.factor(cor.dat$Data)

plot2 = cor.dat %>% 
  #rename("Data" = "Data", "uncorrected" = "BAGu") %>%
  melt(id.vars = "Data") %>% ggplot(aes(x = value, y = variable, fill = `Data`)) +
  geom_density_ridges(aes(fill = `Data`)) +
  scale_fill_manual(values = c("#E69F00","#56B4E9")) +
  stat_density_ridges(quantile_lines = TRUE, quantiles = 0.5, alpha = .6) +
  theme_classic(base_size = 15) + theme(legend.position = "bottom") + ylab("") + xlab("Pearson's Correlation Coefficient") + 
  ggtitle("FreeSurfer v5 and v7 Metric Correlations per Data Set") + xlim(.5,1.1) +
  theme(axis.text.y=element_blank(), 
      axis.ticks.y=element_blank()) 
plot2 = plot2 + annotate("text", label = (paste("Training mean r = ", round(mean(cor_vec_train),3), sep = "")), x = .5, y = 12.5, size = 6, hjust = 0)
plot2 = plot2 + annotate("text", label = (paste("Test mean r = ", round(mean(cor_vec_test),3), sep = "")), x = .5, y = 11.5, size = 6, hjust = 0)
ggsave(filename = "/Users/max/Documents/Projects/FS_brainage/results/Plot2.pdf",plot = plot2, height = 6, width = 6)
#
############################################################################ #
############################################################################ #
################# 4) Site and sex effects ####################################
############################################################################ #
############################################################################ #
demog = FS5_test %>% select(eid, Sex, Scanner)
coveff = merge(all_preds, demog, by = "eid")
#
########## RANDOM EFFECTS (investigating site effects)
#
# produce 3 long data frames reflecting predictions into FS5, FS7, mixed versions
pred_v5 = rbind(coveff %>% select(eid, FS5_FS5, Sex, Scanner, Age) %>% rename(BrainAge = FS5_FS5), coveff %>% 
                  select(eid, FS7_FS5, Sex, Scanner, Age) %>% rename(BrainAge = FS7_FS5), coveff %>% 
                  select(eid, mix2FS5, Sex, Scanner, Age) %>% rename(BrainAge = mix2FS5))
pred_v7 = rbind(coveff %>% select(eid, FS5_FS7, Sex, Scanner, Age) %>% rename(BrainAge = FS5_FS7), coveff %>% 
                  select(eid, FS7_FS7, Sex, Scanner, Age) %>% rename(BrainAge = FS7_FS7), coveff %>% 
                  select(eid, mix2FS7, Sex, Scanner, Age) %>% rename(BrainAge = mix2FS7))
pred_mix = rbind(coveff %>% select(eid, FS52mix, Sex, Scanner, Age) %>% rename(BrainAge = FS52mix), coveff %>% 
                  select(eid, FS72mix, Sex, Scanner, Age) %>% rename(BrainAge = FS72mix), coveff %>% 
                  select(eid, mix_test, Sex, Scanner, Age) %>% rename(BrainAge = mix_test))
pred_within = rbind(coveff %>% select(eid, FS5_FS5, Sex, Scanner, Age) %>% rename(BrainAge = FS5_FS5), coveff %>% 
                      select(eid, FS7_FS7, Sex, Scanner, Age) %>% rename(BrainAge = FS7_FS7), coveff %>% 
                      select(eid, mix_test, Sex, Scanner, Age) %>% rename(BrainAge = mix_test))
# add dummies for the version
pred_v5$Version = c(replicate(nrow(coveff),"FS5_FS5"),replicate(nrow(coveff),"FS7_FS5"), replicate(nrow(coveff),"mix_FS5"))
pred_v7$Version = c(replicate(nrow(coveff),"FS5_FS7"),replicate(nrow(coveff),"FS7_FS7"), replicate(nrow(coveff),"mix_FS7"))
pred_mix$Version = c(replicate(nrow(coveff),"FS5_mix"),replicate(nrow(coveff),"FS7_mix"), replicate(nrow(coveff),"mix_mix"))
pred_within$Version = c(replicate(nrow(coveff),"FS5_FS5"),replicate(nrow(coveff),"FS7_FS7"), replicate(nrow(coveff),"mix_mix"))
# run models to check for an interaction effect of scanner and version
summary(lmer(BrainAge ~ Scanner + Sex + Age + (1|Version), data = pred_v5))
summary(lmer(BrainAge ~ Scanner + Sex + Age + (1|Version), data = pred_v7))
summary(lmer(BrainAge ~ Scanner + Sex + Age + (1|Version), data = pred_mix))
summary(lmer(BrainAge ~ Scanner + Sex + Age + (1|Version), data = pred_within))
#
# check the results in terms of mean BrainAges
pred_v7 %>% group_by(Scanner) %>% summarize(M = mean(BrainAge))

############ FIXED EFFECTS
#
# investigating first the effect of sex
#
namelist = names(coveff[3:11])
statsframe = data.frame(matrix(ncol = 6, nrow = length(namelist)))
for (i in 1:length(namelist)){
  F1 = formula(paste(namelist[i]," ~ Sex + Age + Scanner", sep = ""))
  m1 = (lm(formula = F1, data = coveff))
  l1 = emmeans(m1, specs = "Sex")
  statsframe[i,] = data.frame(pairs(l1))
  ## below a version using the raw regression estimates = naiive estimates
  #b = (summary(lm(formula = F1, data = coveff))$coeff[2])
  #se = (summary(lm(formula = F1, data = coveff))$coeff[2,2])
  #t = (summary(lm(formula = F1, data = coveff))$coeff[2,3])
  #p = (summary(lm(formula = F1, data = coveff))$coeff[2,4])
  #statsframe[i] = c(namelist[i], b, se, t, p)
}
#statsframe = data.frame(t(statsframe))
names(statsframe) = c("Name", "Beta", "SE","df", "T", "p")
statsframe[2:5] = statsframe[2:5] %>% mutate_all(as.numeric)
statsframe$Name = namelist
# Show p-corrected significant 
statsframe$p.adj = p.adjust(statsframe$p, method = "fdr")
pvec = ifelse(statsframe$p.adj < .05, "*","")
# If wanting to check which are sig and which are not:
#statsframe %>% filter(p.adj < .05)
#statsframe %>% filter(p.adj > .05)

# add start for UNCORRECTED findings
sex_plot = ggplot(statsframe, aes(x=Name, y=Beta)) + 
  geom_point(size = 2.5)+
  geom_errorbar(aes(ymin=Beta-SE, ymax=Beta+SE), width=.2,
                position=position_dodge(0.05)) +
  theme_bw() + xlab("Training and test data") + 
  geom_text(data = statsframe, aes(x = Name, y = Beta+SE+.1, label = pvec)) +
  ylab("Beta coefficients (in years) ± standard error") +
  scale_x_discrete(limits = c("FS5_FS5","FS7_FS5","mix2FS5",  "FS5_FS7","FS7_FS7", "mix2FS7" , "FS52mix",  "FS72mix", "mix_test"),
                   labels=c("FS5 to FS5", "FS7 to FS5", "Mix to FS5", "FS5 to FS7", "FS7 to FS7", "Mix to FS7", "FS5 to Mix", "FS7 to Mix", "Mix to Mix"))
ggsave(filename = "/Users/max/Documents/Projects/FS_brainage/results/Sex_Plot.pdf",plot = sex_plot, height = 4, width = 8)
#
#
# 
# then investigating version differences as fixed effects
#
m1 = (lm(BrainAge ~ Scanner + Sex + Age + Version, data = pred_v5))
m2 = (lm(BrainAge ~ Scanner + Sex + Age + Version, data = pred_v7))
m3 = (lm(BrainAge ~ Scanner + Sex + Age + Version, data = pred_mix))
m4 = (lm(BrainAge ~ Scanner + Sex + Age + Version, data = pred_within))
l1 = emmeans(m1, specs = "Version")
l2 = emmeans(m2, specs = "Version")
l3 = emmeans(m3, specs = "Version")
l4 = emmeans(m4, specs = "Version")
marginal_differences = rbind(data.frame(pairs(l1)),data.frame(pairs(l2)),data.frame(pairs(l3)),data.frame(pairs(l4)))
marginal_differences$p.adj = p.adjust(marginal_differences$p.value, method = "fdr")
write.csv(file = "/Users/max/Documents/Projects/FS_brainage/results/marginal_differences.csv", marginal_differences)
#
############################################################################ #
############################################################################ #
################# THE END ################################################## #
############################################################################ #
############################################################################ #
# That's it.
print("The end.")
#

