
##Second Plot - Cluster Ligand-Receptor Pairs Interactions
suppressPackageStartupMessages(library(tidyverse))
library(tidyverse)
suppressPackageStartupMessages(library(reshape))
library(reshape)
library(optparse)

option_list = list(
  make_option(c("-f", "--lr_file"), type="character", default="new_clusters_lr.csv", 
              help="dataset file name", metavar="character"),
  make_option(c("-c","--pvalues"), type="character", default="new_pvalues2.csv", 
              help="dataset file name", metavar="character"),
  make_option(c("-o", "--out"), type="character", default="dotplot.pdf", 
              help="output file name [default= %default]", metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);



#Cluster and L-R selection

data3 <- read.csv(opt$lr_file)
#data3

data4 <- read.csv(opt$pvalues)
#data4

data3 = melt(data3, id = c("X"))
colnames(data3) = c("LR", "Cluster", "value")
#data3


data4 = melt(data4, id = c("X"))
colnames(data4) = c("LR", "Cluster", "pvalue")
#data4

data5 = merge(data3, data4, by = c("LR", "Cluster"))
#data5

data5[data5==0] <- NA
#data5

data5 = na.omit(data5)
#data5

data5$Cluster <- gsub('_', '-', data5$Cluster)
#data5

data5$pvalue <- gsub('Smallest', 'p < 0.01 ', data5$pvalue)
#data5
data5$pvalue <- gsub('Smaller', '0.01 < p < 0.025', data5$pvalue)
#data5
data5$pvalue <- gsub('Small', '0.025 < p < 0.05 ', data5$pvalue)
#data5

values2 <- c(1,2,3); names(values2) <- c('0.025 < p < 0.05', '0.01 < p < 0.025 ','p < 0.01')

g2 = ggplot(data5, aes(x = Cluster, y = LR, color = value, size = pvalue)) + 
  theme_linedraw() + theme(panel.grid.major = element_blank()) +
  scale_color_gradientn(colours = rainbow(5),
                        breaks= c(min(data5$value), max(data5$value)), labels = c("min", "max"),
                        limits=c(min(data5$value), max(data5$value))) +
  geom_vline(xintercept=seq(0.5, 100.5, 1), linewidth = 0.01) +
  geom_hline(yintercept=seq(0.5, 500.5, 1), linewidth = 0.01) +
  geom_point(pch = 16) + 
  scale_x_discrete(name="Cell Types") +
  scale_y_discrete(name="Ligand-Receptor Pairs") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

g2 = g2 + scale_size_discrete(name = "pvalue", range=c(1, 3),
                              labels= c('0.025 < p < 0.05', '0.01 < p < 0.025 ','p < 0.01'))


ggsave(opt$out, g2, height=5, width = 8)

