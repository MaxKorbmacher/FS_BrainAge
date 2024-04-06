# Synthetic data
# Max Korbmacher, 03. April 2024
# 
# get package
install.packages("synthpop")
library(synthpop)

# load data
FS5test = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/test.csv")
FS5train = read.csv("/Users/max/Documents/Projects/FS_brainage/FS5/train.csv")
FS7test = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/test.csv")
FS7train = read.csv("/Users/max/Documents/Projects/FS_brainage/FS7/train.csv")

# create synth versions of the data
FS5_test_syn = syn(FS5test)
FS5_train_syn = syn(FS5train)
FS7_test_syn = syn(FS7test)
FS7_train_syn = syn(FS7train)

setwd("/Users/max/Documents/Projects/FS_brainage/")

# write the new (synthetic!) data frames
write.syn(FS5_test_syn, file = "FS5_test_synth", filetype = "csv")
write.syn(FS5_train_syn, "FS5_train_synth", filetype = "csv")
write.syn(FS7_test_syn, "FS7_test_synth", filetype = "csv")
write.syn(FS7_train_syn, "FS7_train_synth", filetype = "csv")
