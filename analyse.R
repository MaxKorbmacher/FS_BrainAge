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
               Metrics, update = F)
# read data used for demographics
FS5_train = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/train.csv")
FS5_test = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/test.csv")
FS7_train = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/train.csv")
FS7_test = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/test.csv")
# read predictions in test data
## XGB
FS5preds_XGB = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5predictions.csv")
FS7preds_XGB = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7predictions.csv")
## SVM
FS5preds_SVM = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5predictions_SVM.csv")
FS7preds_SVM = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7predictions_SVM.csv")
## Light GBM
FS5preds_LGBM = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5_LGBM_predictions.csv")
FS7preds_LGBM = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7_LGBM_predictions.csv")
## Lasso
FS5preds_Lasso = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5predictions_Lasso.csv")
FS7preds_Lasso = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7predictions_Lasso.csv")
# put the predictions in a list
FS5preds = list(FS5preds_XGB, FS5preds_SVM, FS5preds_LGBM, FS5preds_Lasso)
FS7preds = list(FS7preds_XGB, FS7preds_SVM, FS7preds_LGBM, FS7preds_Lasso)

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
# list the predictions
FS5train_preds = list(FS5train_preds_XGB, FS5train_preds_SVM, FS5train_preds_LGBM, FS5train_preds_Lasso)
FS7train_preds = list(FS7train_preds_XGB, FS7train_preds_SVM, FS7train_preds_LGBM, FS7train_preds_Lasso)
#
############################################################################ #
############################################################################ #
################# 1) Demographics #####################
############################################################################ #
############################################################################ #
# for sample descriptor, merge dfs
demo = rbind(FS5_test, FS5_train)
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
for (i in 1:length(FS5train_preds)){
  FS5_tmp = FS5train_preds[[i]]%>% select(eid,Age,pred_age_train)
  FS7_tmp = FS7train_preds[[i]]%>% select(eid,pred_age_train)
  training = merge(FS5_tmp,FS7_tmp, by = "eid")
  names(training) = c("eid", "Age", "FS5", "FS7")
  print(cor(training))
}
print("#######################################################################")
print("We can see that Lasso produces the best training results.")
print("We can double check whether this holds true when predicting in independent test data.")
print("#######################################################################")
for (i in 1:length(FS5preds)){
  FS5_tmp = FS5preds[[i]]%>% select(eid,Age,FS5_FS5,FS5_FS7)
  FS7_tmp = FS7preds[[i]]%>% select(eid,FS7_FS7, FS7_FS5)
  test = merge(FS5_tmp,FS7_tmp, by = "eid")
  print(cor(test))
}
print("#######################################################################")
print("Based on the prediction-label correlations, Lasso performs best.")
print("Hence, we report the results for Lasso.")
print("#######################################################################")
#
#
print("We plot the Lasso predictions")
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
train_plot_FS5 = scatter.plot(FS5train_preds_Lasso, FS5train_preds_Lasso$pred_age_train, c("Training Sample Age"), c("FS5 Predicted Age"))
train_plot_FS7 = scatter.plot(FS7train_preds_Lasso, FS7train_preds_Lasso$pred_age_train, c("Training Sample Age"), c("FS7 Predicted Age"))
# test scatter plots
test_plot_FS5_FS5 = scatter.plot(FS5preds_Lasso, FS5preds_Lasso$FS5_FS5, c("Test Sample Age"), c("Predicted Age FS5 > FS5"))
test_plot_FS5_FS7 = scatter.plot(FS5preds_Lasso, FS5preds_Lasso$FS5_FS7, c("Test Sample Age"), c("Predicted Age FS5 > FS7"))
test_plot_FS7_FS5 = scatter.plot(FS7preds_Lasso, FS7preds_Lasso$FS7_FS5, c("Test Sample Age"), c("Predicted Age FS7 > FS5"))
test_plot_FS7_FS7 = scatter.plot(FS7preds_Lasso, FS7preds_Lasso$FS7_FS7, c("Test Sample Age"), c("Predicted Age FS7 > FS7"))
# arrange the panels
plot = ggpubr::ggarrange(train_plot_FS5, train_plot_FS7,
          test_plot_FS5_FS5, test_plot_FS5_FS7,
          test_plot_FS7_FS7, test_plot_FS7_FS5,
          ncol = 2,nrow = 3,
          labels = c("a", "b","c","d","e","f"),
          common.legend = T)
ggsave(filename = "/Users/max/Documents/Projects/FS_brainage/Plot.pdf",plot = plot, height = 9, width = 11)
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
train_performance$FSversion = c(5,5,5,5,7,7,7,7)
train_performance$Model = c("XGBoost", "SVM", "LightGBM", "Lasso")
write.csv(file = "/Users/max/Documents/Projects/FS_brainage/train_performance.csv", train_performance)
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
test_performance$Predictions = c(replicate(4,"FS5toFS5"), replicate(4,"FS5toFS7"), replicate(4,"FS7toFS5"), replicate(4,"FS7toFS7"))
test_performance$Model = c("XGBoost", "SVM", "LightGBM", "Lasso")
write.csv(file = "/Users/max/Documents/Projects/FS_brainage/test_performance.csv", test_performance)
print("#######################################################################")
print("#######################################################################")
#
#
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
ggsave(filename = "/Users/max/Documents/Projects/FS_brainage/Plot2.pdf",plot = plot2, height = 6, width = 6)
#
# That's it.
print("The end.")
#

