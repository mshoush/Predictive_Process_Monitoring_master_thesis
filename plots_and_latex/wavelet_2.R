#source("http://bioconductor.org/biocLite.R")
#biocLite("Rgraphviz")
#biocLite("scmamp")

#install.packages("BiocManager")
#BiocManager::install("Rgraphviz")
#BiocManager::install("scmamp")


library(scmamp)
library(ggplot2)
library(Rgraphviz)
library(reshape2)
library(plyr)
library(dplyr)


setwd("/home/kamel/utcs/master_courses/4th/thesis/results_all/catboost/code")

# toy example
data(data_blum_2015)
data(data_gh_2008)

head(data.gh.2008)
dim(data.gh.2008)

test <- nemenyiTest (data.gh.2008, alpha=0.05)
test

test$diff.matrix

abs(test$diff.matrix) > test$statistic

plotCD (data.gh.2008, alpha=0.05, cex=1.25)


# actual data
data_fscore <- read.csv("./final_data/nemenyi/data_wavelet_fscore_nemenyi.csv", sep=";", header=T)

data_fscore$method <- as.character(data_fscore$method)
data_fscore$dataset <- as.character(data_fscore$dataset)
data_fscore$cls <- sapply(data_fscore$method, function(x) strsplit(x, "_")[[1]][3])

data <- read.csv("./final_data/nemenyi/data_wavelet_nemenyi.csv", sep=";", header=T)

data

data$method <- as.character(data$method)
data$dataset <- as.character(data$dataset)

# best scores from each classifier
data$cls <- sapply(data$method, function(x) strsplit(x, "_")[[1]][3])

dt_best <- ddply(data, .(cls, dataset), summarize, score=max(score))

dt_best_cast <- dcast(dt_best, dataset~cls, value.var="score")
rownames(dt_best_cast) <- dt_best_cast[,1]
dt_best_cast <- dt_best_cast[,-1]

png("./images_cd/wavelet/cd_classifiers.png", width=1000, height=500)
plotCD (dt_best_cast, alpha=0.05, cex=2)
dev.off()
pdf("./images_cd/wavelet/rplot_cd_classifiers.pdf", width=100, height=40) 
plotCD (dt_best_cast, alpha=0.05, cex=15)
dev.off() 


# catboost only
data$bucket_encoding <- sapply(data$method, function(x) paste(strsplit(x, "_")[[1]][1:2], collapse="_"))

dt_xgboost <- subset(data, cls=="catboost")

dt_xgb_cast <- dcast(dt_xgboost, dataset~bucket_encoding, value.var="score")
rownames(dt_xgb_cast) <- dt_xgb_cast[,1]
dt_xgb_cast <- dt_xgb_cast[,-1]

png("./images_cd/wavelet/cd_catboost.png", width=1000, height=500)
plotCD (dt_xgb_cast, alpha=0.05, cex=1.25)
dev.off()
pdf("./images_cd/wavelet/rplot_cd_catboost.pdf", width=150, height=40) 
plotCD (dt_xgb_cast, alpha=0.05, cex=15)
dev.off() 



# all methods
data_cast <- dcast(data, dataset~method, value.var="score")
rownames(data_cast) <- data_cast[,1]
data_cast <- data_cast[,-1]

test <- nemenyiTest (data_cast, alpha=0.05)
test

test$diff.matrix

abs(test$diff.matrix) > test$statistic

png("images_cd/wavelet/cd_all.png", width=2000, height=900)
plotCD (data_cast, alpha=0.05, cex=0.8)
dev.off()
pdf("./images_cd/wavelet/rplot_cd_all.pdf", width=550, height=90) 
plotCD (data_cast, alpha=0.05, cex=15)
dev.off() 


library(dplyr)
# count, in how many datasets is best
dt_best <- ddply(data, .(cls, dataset), summarize, score=max(score))
dt_best <- dt_best %>% 
  group_by(dataset) %>%
  filter(round(score,2) == round(max(score),2)) %>%
  arrange(dataset, score, cls)
table(dt_best$cls)
data.frame(dt_best)

dt_best <- ddply(data_fscore, .(cls, dataset), summarize, score=max(score))
dt_best_fscore <- dt_best %>% 
  group_by(dataset) %>%
  filter(round(score,2) == round(max(score),2)) %>%
  arrange(dataset, score, cls)
table(dt_best_fscore$cls)
data.frame(dt_best_fscore)


dt_best <- subset(data, cls=="catboost") %>% 
  group_by(dataset) %>%
  filter(round(score,2) == round(max(score),2)) %>%
  arrange(dataset, method, score, cls)
table(dt_best$method)
sum(table(dt_best$method))
data.frame(dt_best)
subset(data.frame(dt_best), bucket_encoding=="prefix_index")
subset(data.frame(dt_best), grepl("knn", bucket_encoding))

dt_best_fscore <- subset(data_fscore, cls=="catboost") %>% 
  group_by(dataset) %>%
  filter(round(score,2) == round(max(score),2)) %>%
  arrange(dataset, method, score, cls)
table(dt_best_fscore$method)
sum(table(dt_best_fscore$method))
data.frame(dt_best_fscore)

