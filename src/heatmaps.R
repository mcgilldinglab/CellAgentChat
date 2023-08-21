
suppressPackageStartupMessages(library(tidyverse))
library(tidyverse)
suppressPackageStartupMessages(library(ComplexHeatmap))
library(ComplexHeatmap)
library(utils)
library(optparse)

option_list = list(
  make_option(c("-f", "--interaction_file"), type="character", default="interaction_matrix.csv", 
              help="dataset file name", metavar="character"),
  make_option(c("-c","--cluster_names"), type="character", default="cluster_names.csv", 
              help="dataset file name", metavar="character"),
  make_option(c("-o", "--out"), type="character", default="heatmap.pdf", 
              help="output file name [default= %default]", metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);


data1 <- read.csv(opt$interaction_file)

clusters = read.csv(opt$cluster_names)
clusters = clusters$X0

colnames(data1) = clusters
rownames(data1) = clusters[2:length(clusters)]

#length(clusters)

#clusters[2]

data1 =  data1 %>% select(c(clusters[2]:clusters[length(clusters)]))


data2 = t(data.matrix(data1))
#data2


ggPalette <- function(n) {
  hues = seq(15, 375, length = n + 1)
  grDevices::hcl(h = hues, l = 65, c = 100)[1:n]
}

scPalette <- function(n) {
  colorSpace <- c('#E41A1C','#377EB8','#4DAF4A','#984EA3','#F29403','#F781BF','#BC9DCC','#A65628','#54B0E4','#222F75','#1B9E77','#B2DF8A',
                  '#E3BE00','#FB9A99','#E7298A','#910241','#00CDD1','#A6CEE3','#CE1261','#5E4FA2','#8CA77B','#00441B','#DEDC00','#B3DE69','#8DD3C7','#999999')
  if (n <= length(colorSpace)) {
    colors <- colorSpace[1:n]
  } else {
    colors <- grDevices::colorRampPalette(colorSpace)(n)
  }
  return(colors)
}

color = scPalette((ncol(data2)))
names(color) = colnames(data2)

df<- data.frame(group = colnames(data2)); rownames(df) <- colnames(data2)

ra = rowAnnotation(Strength = anno_barplot(rowSums(abs(data2)), border = FALSE,
                                           gp = gpar(fill = color, col=color)), show_annotation_name = FALSE)
ca = HeatmapAnnotation(Strength = anno_barplot(colSums(abs(data2)), border = FALSE,
                                               gp = gpar(fill = color, col=color)), show_annotation_name = FALSE)

col_annotation = HeatmapAnnotation(df = df, col = list(group = color),
                                   show_legend = FALSE, show_annotation_name = FALSE,
                                   simple_anno_size = grid::unit(0.5, "cm"))

row_annotation = rowAnnotation(df = df, col = list(group = color),
                               show_legend = FALSE, show_annotation_name = FALSE,
                               simple_anno_size = grid::unit(0.5, "cm"))


pdf(opt$out)

Heatmap(data2, cluster_rows = FALSE,cluster_columns = FALSE, 
        col = c("White","pink","Red", "Brown"),
        right_annotation = ra, top_annotation = ca, 
        bottom_annotation = col_annotation, left_annotation =row_annotation,
        row_names_side = "left",row_names_rot = 0, row_title = "Source (Sender)",
        column_title = "Number of Interactions",
        heatmap_legend_param = list(title = "Num of Interactions", title_position = "leftcenter-rot"),
        row_names_gp = gpar(fontsize = 10),column_names_gp = gpar(fontsize = 10),
        width = ncol(data2)*unit(8, "mm"), height = nrow(data2)*unit(8, "mm"))

dev.off()
