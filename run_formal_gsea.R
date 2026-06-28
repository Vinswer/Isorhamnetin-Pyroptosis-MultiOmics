args <- commandArgs(trailingOnly=TRUE)
rank_file <- args[1]
gmt_dir <- args[2]
out_csv <- args[3]
out_json <- args[4]
suppressPackageStartupMessages({
  library(fgsea)
  library(msigdbr)
  library(jsonlite)
})
rank_df <- read.csv(rank_file, check.names = FALSE)
rank_vec <- rank_df$ranking_score
names(rank_vec) <- rank_df$gene_name
rank_vec <- rank_vec[!is.na(rank_vec)]
rank_vec <- sort(rank_vec, decreasing = TRUE)
read_gmt_simple <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  out <- list()
  for (ln in lines) {
    parts <- strsplit(ln, "\t")[[1]]
    if (length(parts) >= 3) {
      term <- parts[1]
      genes <- unique(parts[3:length(parts)])
      out[[term]] <- genes
    }
  }
  out
}
reactome <- read_gmt_simple(file.path(gmt_dir, "Reactome_2022.gmt"))
kegg <- read_gmt_simple(file.path(gmt_dir, "KEGG_2013.gmt"))
go_bp <- read_gmt_simple(file.path(gmt_dir, "GO_Biological_Process_2021.gmt"))
go_cc <- read_gmt_simple(file.path(gmt_dir, "GO_Cellular_Component_2021.gmt"))
pick_one <- function(lib, patterns) {
  nm <- names(lib)
  hit <- nm[sapply(tolower(nm), function(x) any(sapply(patterns, function(p) grepl(p, x, fixed = TRUE))))]
  if (length(hit) == 0) return(NULL)
  lib[[hit[1]]]
}
pathways <- list(
  GO_PYROPTOSIS = pick_one(go_bp, c("pyroptosis")),
  REACTOME_INFLAMMASOMES = pick_one(reactome, c("inflammasome")),
  REACTOME_INNATE_IMMUNE_SYSTEM = pick_one(reactome, c("innate immune system")),
  KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY = pick_one(kegg, c("nod-like receptor signaling pathway", "nod like receptor signaling pathway")),
  REACTOME_INTERLEUKIN_1_SIGNALING = pick_one(reactome, c("interleukin-1", "interleukin 1"))
)
pathways <- pathways[!sapply(pathways, is.null)]
if (length(pathways) == 0) stop("No target pathways were found in the local GMT resources.")
fg <- fgsea(pathways = pathways, stats = rank_vec, minSize = 5, maxSize = 5000, nperm = 5000)
fg <- as.data.frame(fg)
fg$leadingEdge_str <- vapply(fg$leadingEdge, function(x) paste(x, collapse = ";"), character(1))
write.csv(fg, out_csv, row.names = FALSE)
detail_list <- list()
for (i in seq_len(nrow(fg))) {
  term <- fg$pathway[i]
  ps <- pathways[[term]]
  prep <- fgsea::preparePathwaysAndStats(list(tmp = ps), rank_vec, minSize = 1, maxSize = 5000, gseaParam = 1)
  res <- fgsea::calcGseaStat(
    stats = prep$stats,
    selectedStats = prep$selectedStats[[1]],
    returnAllExtremes = TRUE,
    returnLeadingEdge = TRUE,
    scoreType = "std"
  )
  detail_list[[term]] <- list(
    pathway = term,
    NES = fg$NES[i],
    ES = fg$ES[i],
    pval = fg$pval[i],
    padj = fg$padj[i],
    leadingEdge = fg$leadingEdge[[i]],
    selectedStats = names(prep$stats)[prep$selectedStats[[1]]],
    selectedIdx = as.integer(prep$selectedStats[[1]]),
    runningScore = as.numeric(res$res)
  )
}
writeLines(jsonlite::toJSON(detail_list, auto_unbox = TRUE, pretty = TRUE, null = "null"), out_json)
