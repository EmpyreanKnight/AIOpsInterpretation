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


# ------------------------------------------------------------ General code ------------------------------------------------------------
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
topn_ranking <- c(1, 3, 5)
output <- matrix(nrow=length(models)*(length(models)-1)*mean(periods)*length(topn_ranking), ncol=6)
colnames(output) <- c('Dataset', 'Model1', 'Model2', 'Period', 'Topn', 'Overlap')
idx <- 1
for (dataset in datasets) {
  for (model_idx1 in 1:(length(models)-1)) {
    model1 <- toupper(models[model_idx1])
    for (model_idx2 in (model_idx1+1):length(models)) {
      model2 <- toupper(models[model_idx2])
      for (i in 1:periods[dataset]) {
        for (j in 1:length((topn_ranking))) {
          model1_features <- df %>% filter(Dataset==named_data[dataset] & Model==model1 & Period==i & Rank %in% 1:topn_ranking[j]) %>% pull(Feature)
          model2_features <- df %>% filter(Dataset==named_data[dataset] & Model==model2 & Period==i & Rank %in% 1:topn_ranking[j]) %>% pull(Feature)
          output[idx, 1] <- named_data[dataset]
          output[idx, 2] <- model1
          output[idx, 3] <- model2
          output[idx, 4] <- i
          output[idx, 5] <- topn_ranking[j]
          output[idx, 6] <- length(intersect(model1_features, model2_features))/length(union(model1_features, model2_features))
          idx <- idx + 1
        }
      }
    }
  }
}
df <- data.frame(output)
write.csv(df, file='overlapping_in_period.csv', quote=FALSE, row.names=FALSE)


# ------------------------------------------------------------ Code for Report 1 ------------------------------------------------------------
# Fig. 1: AUC performance curve in each time period for each model
datasets <- c('google', 'disk')
periods <- c(28, 36)
names(periods) <- datasets
named_data <- c('Google', 'Backblaze')
names(named_data) <- datasets
models <- c('lda', 'qda', 'lr', 'cart', 'gbdt', 'nn', 'rf', 'rgf')
df <- NA
for (dataset in datasets) {
  for (model in models) {
    dfi <- read.csv(paste('preliminary_results/importance_', dataset, '_', model, '.csv', sep=''))
    dfi <- dfi[c('Period', 'Model', 'Test.AUC')]
    dfi <- dfi[dfi$Period != periods[dataset], ]
    dfi$Dataset <- named_data[dataset]

    if (typeof(df) != 'list') {
      df <- dfi
    } else {
      df <- rbind(df, dfi)
    }
  }
}
ggplot(df %>% group_by(Period, Dataset, Model) %>% summarize(AUC=mean(Test.AUC)), aes(x=Period, y=AUC, color=Model)) + 
  geom_line() + geom_point() +
  facet_grid(.~Dataset, scales='free_x')
ggsave('compare_performace.pdf', width=190, height=80, unit='mm')


# Statistically categorize models into differently performed groups
# Fig 2: double SK of grouping models by the ranking of their AUC performance in each time period
for (dataset in datasets) {
  # read and combine data frames
  df <- NA
  for (model in models) {
    dfi <- read.csv(paste('preliminary_results/importance_', dataset, '_', model, '.csv', sep=''))
    dfi <- dfi[c('Model', 'Period', 'Test.AUC')]

    if (typeof(df) != 'list') {
      df <- dfi
    } else {
      df <- rbind(df, dfi)
    }
  }

  # First SK ranks the AUC performance of models in each time period
  sk <- NA  # format: [Model, Rank, Period]
  for (i in 1:(periods[dataset]-1)) {  # skip the last period as no testing data for it
    dfi <- df[df$Period == i,]
    dfi <- data.frame(variable=dfi$Model, value=dfi$Test.AUC)
    ski <- with(dfi, SK(x=variable, y=value, model='y~x', which='x'))
    capture.output(ski <- summary(ski), file='NUL')
    ski <- data.frame(Model=ski$Levels, Rank=as.integer(ski$`SK(5%)`))
    ski$Period <- i
    
    if (typeof(sk) != 'list') {
      sk <- ski
    } else {
      sk <- rbind(sk, ski)
    }
  }
  write.csv(sk, file=paste('auc_ranking_', dataset, '.csv', sep=''), quote=FALSE, row.names=FALSE)

  # Double SK ranks the ranking of the model AUC performane in each period
  sk2 <- with(sk, SK(x=Model, y=Rank, model='y~x', which='x'))
  sk2 <- summary(sk2)
  sk2 <- data.frame(Model=sk2$Levels, Group=as.integer(sk2$`SK(5%)`))
  sk2 <- merge(sk, sk2, by='Model')
  sk2$Group <- max(sk2$Group) - sk2$Group + 1
  
  ggplot(sk2, aes(x=Model, y=Rank)) + geom_boxplot() + facet_grid(.~Group, scales='free_x', space = "free_x") +
      labs(x='Model', y='Ranking of performance') + scale_y_reverse()
  ggsave(paste('double_sk_', dataset, '.pdf', sep=''), width=190, height=80, unit='mm')     
}


# Fig 3, 4, 5: overlapping value related heatmaps
topn_ranking <- c(1, 3, 5)
for (dataset in datasets) {
  df_auc <- read.csv(paste('auc_ranking_', dataset, '.csv', sep=''))  # data frame for AUC ranking in each period 
  df_overlap <- read.csv('overlapping_in_period.csv')  # data frame for overlapping values in each period
  df_overlap <- df_overlap[df_overlap$Dataset==named_data[dataset], ]

  # count on how many periods two models are in the same performance group (how similar their performance)
  output <- matrix(nrow=length(models), ncol=length(models))
  rownames(output) <- toupper(models)
  colnames(output) <- toupper(models)
  # count the mean overlapping in different conditions (on all periods, on periods two models have the same perf group, on periods two models have different perf group)
  top1_all <- matrix(nrow=length(models), ncol=length(models))
  top3_all <- matrix(nrow=length(models), ncol=length(models))
  top5_all <- matrix(nrow=length(models), ncol=length(models))
  top1_same <- matrix(nrow=length(models), ncol=length(models))
  top3_same <- matrix(nrow=length(models), ncol=length(models))
  top5_same <- matrix(nrow=length(models), ncol=length(models))
  top1_diff <- matrix(nrow=length(models), ncol=length(models))
  top3_diff <- matrix(nrow=length(models), ncol=length(models))
  top5_diff <- matrix(nrow=length(models), ncol=length(models))

  for (model_idx1 in 1:(length(models)-1)) {
    model1 <- toupper(models[model_idx1])
    for (model_idx2 in (model_idx1+1):length(models)) {
      model2 <- toupper(models[model_idx2])
      count_same <- 0
      count_diff <- 0
      top_ranks <- c(0, 0, 0, 0, 0, 0, 0, 0, 0)
      for (i in 1:(periods[dataset]-1)) {
        model1_rank <- df_auc %>% filter(Model==model1 & Period==i) %>% pull(Rank)
        model2_rank <- df_auc %>% filter(Model==model2 & Period==i) %>% pull(Rank)
        for (j in 1:length(topn_ranking)) {
          overlap <- df_overlap %>% filter(Model1==model1 & Model2==model2 & Period==i & Topn==topn_ranking[j]) %>% pull(Overlap)
          top_ranks[j] <- top_ranks[j] + overlap
          if (model1_rank == model2_rank) {
            count_same <- count_same + 1
            top_ranks[3+j] <- top_ranks[3+j] + overlap
          } else{
            count_diff <- count_diff + 1
            top_ranks[6+j] <- top_ranks[6+j] + overlap
          }
        }
      }
      output[model_idx1, model_idx2] <- count_same / length(topn_ranking)
      top1_all[model_idx1, model_idx2]  <- length(topn_ranking)*top_ranks[1]/(count_same+count_diff)
      top3_all[model_idx1, model_idx2]  <- length(topn_ranking)*top_ranks[2]/(count_same+count_diff)
      top5_all[model_idx1, model_idx2]  <- length(topn_ranking)*top_ranks[3]/(count_same+count_diff)
      top1_same[model_idx1, model_idx2] <- length(topn_ranking)*top_ranks[4]/count_same
      top3_same[model_idx1, model_idx2] <- length(topn_ranking)*top_ranks[5]/count_same
      top5_same[model_idx1, model_idx2] <- length(topn_ranking)*top_ranks[6]/count_same
      top1_diff[model_idx1, model_idx2] <- length(topn_ranking)*top_ranks[7]/count_diff
      top3_diff[model_idx1, model_idx2] <- length(topn_ranking)*top_ranks[8]/count_diff
      top5_diff[model_idx1, model_idx2] <- length(topn_ranking)*top_ranks[9]/count_diff
    }
  }

  plot_names <- c('grouping_', 'all_top1_', 'all_top3_', 'all_top5_', 'same_top1_', 'same_top3_', 'same_top5_', 'diff_top1_', 'diff_top3_', 'diff_top5_')
  tbl_names <- list(output, top1_all, top3_all, top5_all, top1_same, top3_same, top5_same, top1_diff, top3_diff, top5_diff)
  rownames(output) <- c('', '', '', '', '', '', '', '')
  colnames(output) <- toupper(models)
  diag(output) <- 0
  for (i in 1:10) {
    mat <- tbl_names[[i]]
    rownames(mat) <- c('', '', '', '', '', '', '', '')
    colnames(mat) <- toupper(models)
    diag(mat) <- 0
    mat[is.nan(mat)] <- 0
    #mat <- mat[1:7, 2:8]

    pdf(file = paste(plot_names[i], dataset, '.pdf', sep=''))
    corrplot(output, method='color', type='upper', is.corr=FALSE, tl.col='black', tl.srt=0, tl.offset=.6, p.mat=output, insig='p-value', sig.level=-1, tl.pos='td', mar=c(0,1,0,0), outline='gray')
    corrplot(t(mat), add=T, method='number', type='lower', is.corr=FALSE, tl.col='black', tl.srt=0, tl.pos='ld', outline='gray')
    dev.off()
  }
}


# ------------------------------------------------------------ Code for Report 2 ------------------------------------------------------------
# Fig. 1, 2, 7, 8, 9, and 10: Top n overlapping value heatmap
df <- read.csv('ranking_in_period.csv')
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

        res1 <- dfi %>% filter(Rank %in% 1:5) %>% pull(Feature) 
        res2 <- dfj %>% filter(Rank %in% 1:5) %>% pull(Feature)
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


# Fig 3, 4, and 5: Top-N vs Original
require(corrplot)
for (dataset in datasets) {
  next_period_mat <- matrix(nrow=length(models), ncol=periods[dataset]-1)
  for (i in 1:length(models)) {
    model <- models[i]
    full_mat <- matrix(nrow=periods[dataset], ncol=periods[dataset])
    df1 <- read.csv(paste('preliminary_results/importance_', dataset, '_', model, '.csv', sep=''))
    df2 <- read.csv(paste('preliminary_results/top_', dataset, '_', model, '.csv', sep=''))
    for (j in 1:(periods[dataset]-1)) {
      obs1 <- mean(df1 %>% filter(Period == j) %>% pull(Test.AUC))
      obs2 <- mean(df2 %>% filter(Training.Period == j & Testing.Period == j + 1) %>% pull(Test.AUC))
      next_period_mat[i, j] <- (obs1 - obs2) / obs1

      for (k in (j+1):periods[dataset]) {
        obs1 <- mean(df1 %>% filter(Period == k-1) %>% pull(Test.AUC))
        obs2 <- mean(df2 %>% filter(Training.Period == j & Testing.Period == k) %>% pull(Test.AUC))
        full_mat[j, k] <- (obs1 - obs2) / obs1
      }
    }
    pdf(file = paste('topn_all_', dataset, '_', model, '.pdf', sep=''))
    corrplot(full_mat, method='color', type='upper', is.corr=FALSE, diag=FALSE, tl.col='black', tl.srt=0, tl.offset=.6, tl.pos='td', mar=c(0,1,0,0), outline='gray')
    dev.off()
  }
  rownames(next_period_mat) <- toupper(models)
  pdf(file = paste('topn_next_', dataset, '.pdf', sep=''), height=2)
  corrplot(next_period_mat, tl.col='black', tl.srt=0, tl.offset=.6, tl.cex=0.7, mar=c(0,0,0,0), outline='gray')
  dev.off()

  df <- read.csv('ranking_in_period.csv')
  mat <- data.frame(spread(df %>% filter(Dataset == named_data[dataset] & Rank %in% 1:3 & Period != periods[dataset]) %>% group_by(Model, Period) %>% summarize(N=n()), Period, N))
  mat <- data.matrix(mat[, 2:ncol(mat)])
  rownames(mat) <- toupper(models)
  colnames(mat) <- 1:(periods[dataset]-1)
  pdf(file = paste('topn_next_num_', dataset, '.pdf', sep=''), height=2)
  corrplot(mat, method='number', is.corr=F, tl.col='black', cl.lim=c(0, max(mat)), tl.srt=0, tl.offset=.6, tl.cex=0.7, mar=c(0,0,0,0), outline='gray')
  dev.off()
}
