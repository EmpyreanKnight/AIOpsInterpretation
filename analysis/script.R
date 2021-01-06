require(ggpubr)
require(effsize)
require(xtable)
require(ScottKnott)
require(gtools)
require(stringi)
require(stringr)
require(scales)
require(tidyr)
require(ggplot2)
require(dplyr)
require(corrplot)


datasets <- c('google', 'disk')
periods <- c(28, 36)
names(periods) <- datasets
named_data <- c('Google', 'Backblaze')
names(named_data) <- datasets
models <- c('lda', 'qda', 'lr', 'cart', 'gbdt', 'nn', 'rf', 'rgf')


# generate feature importance ranking with Scott-Knott test in each time period
# the output is combined into one data frame with columns for the dataset, model, time period, feature and its ranking
output <- NA
for (dataset in datasets) {
  for (model in models) {
    df <- read.csv(paste('preliminary_results/importance_', dataset, '_', model, '.csv', sep=''))
    
    # on each time period, calculate the feature importance ranking
    for (i in 1:periods[dataset]) {
      dfi <- df[df$Period == i,][colnames(df)[10:length(colnames(df))]]
      sk <- with(gather(dfi), SK(x=key, y=value, model='y~x', which='x'))
      capture.output(sk <- summary(sk), file='NUL')
      sk <- data.frame(Feature=sk$Levels, Rank=as.integer(sk$`SK(5%)`))
      sk$Feature <- as.integer(gsub('.*_(.*)', '\\1', sk$Feature))
      sk$Dataset <- named_data[dataset]
      sk$Model <- toupper(model)
      sk$Period <- i

      # combine individual feature ranking data frames
      if (typeof(output) != 'list') {
        output <- sk
      } else {
        output <- rbind(output, sk)
      }
    }
  }
}
output <- output[c('Dataset', 'Model', 'Period', 'Feature', 'Rank')]
write.csv(output, file='ranking_in_period.csv', quote=F, row.names=F)


# calculate the "overlapping value" of feature importance ranking in top n
# the overlapping value is defined as the intersection devided by the union of features\
# occurred in top n in two groups of observations (e.g., 10 iterations of feature importance ranking in one time period)
# For example, the follwing code create heatmaps for overlapping values of ranking between different time periods
df <- read.csv('ranking_in_period.csv')
require(corrplot)
for (dataset in datasets) {
  for (model in models) {
    top1_mat <- matrix(nrow=periods[dataset], ncol=periods[dataset])
    top3_mat <- matrix(nrow=periods[dataset], ncol=periods[dataset])
    top5_mat <- matrix(nrow=periods[dataset], ncol=periods[dataset])

    for (i in 1:(periods[dataset]-1)) {
      dfi <- df %>% filter(Period==i & Dataset==named_data[dataset] & Model==toupper(model))
      for (j in (i+1):periods[dataset]) {
        dfj <- df %>% filter(Period==j & Dataset==named_data[dataset] & Model==toupper(model))
        
        res1 <- dfi %>% filter(Rank %in% 1:1) %>% pull(Feature) 
        res2 <- dfj %>% filter(Rank %in% 1:1) %>% pull(Feature)
        top1_mat[i, j] <- length(intersect(res1, res2))/length(union(res1, res2))

        res1 <- dfi %>% filter(Rank %in% 1:3) %>% pull(Feature) 
        res2 <- dfj %>% filter(Rank %in% 1:3) %>% pull(Feature)
        top3_mat[i, j] <- length(intersect(res1, res2))/length(union(res1, res2))

        res1 <- sk1 %>% filter(Rank %in% 1:5) %>% pull(Feature) 
        res2 <- sk2 %>% filter(Rank %in% 1:5) %>% pull(Feature)
        top5_mat[i, j] <- length(intersect(res1, res2))/length(union(res1, res2))
      }
    }

    pdf(file = paste('periods_top1_', dataset, '_', model, '.pdf', sep=''))
    top1_mat[!upper.tri(top1_mat)] <- 0
    corrplot(top1_mat, method='color', type='upper', is.corr=FALSE, diag=FALSE, cl.lim=c(0,1), tl.col='black', tl.srt=0, tl.offset=.6, tl.pos='td', mar=c(0,1,0,0), outline='gray')
    dev.off()

    pdf(file = paste('periods_top3_', dataset, '_', model, '.pdf', sep=''))
    top3_mat[!upper.tri(top1_mat)] <- 0
    corrplot(top3_mat, method='color', type='upper', is.corr=FALSE, diag=FALSE, cl.lim=c(0,1), tl.col='black', tl.srt=0, tl.offset=.6, tl.pos='td', mar=c(0,1,0,0), outline='gray')
    dev.off()

    pdf(file = paste('periods_top5_', dataset, '_', model, '.pdf', sep=''))
    top5_mat[!upper.tri(top5_mat)] <- 0
    corrplot(top5_mat, method='color', type='upper', is.corr=FALSE, diag=FALSE, cl.lim=c(0,1), tl.col='black', tl.srt=0, tl.offset=.6, tl.pos='td', mar=c(0,1,0,0), outline='gray')
    dev.off()
  }
}
