#' @file GRIN_sliding_window.R
#'
#' @desc This file uses several functions from GRIN.R at https://github.com/sullivanka/GRIN
#'       The RWR based rankings in GRIN are replaced with RWR rankings from this repo. The
#'       leave-one-out ranks of the input gene set and null distribution are provided to
#'       this R script, which in turns calculates a Mann-Whitney U test with sliding window
#'       and computes an elbow point to filter gene into retained and removed sets. 

parse_arguments <- function() {
  suppressPackageStartupMessages(library(optparse))

  option_list = list(
    make_option(c("-g", "--gene_ranks"),
                action="store",
                default=NULL,
                type='character',
                help="path to the gene_ranks.tsv file from GRIN++"),
    make_option(c("-n", "--null_ranks"),
                action="store",
                default=NULL,
                type='character',
                help="path to the null_ranks.tsv file from GRIN++"),
    make_option(c("-m", "--modname"),
                action="store",
                default="default",
                type='character',
                help="alias for this run. Useful for output."),
    make_option(c("-p", "--plot"),
                action="store_true",
                default=FALSE,
                help="Include this parameter if you want to output PNG plots of results. [default %default]"),
    make_option(c("-o", "--outdir"),
                action="store",
                default=NULL,
                type='character',
                help="path to the output directory"),
    make_option(c("--threads"),
                action="store",
                default=parallel::detectCores()-1,
                type='numeric',
                help="number of threads to use. default for your system is all cores - 1 [default %default]"),
    make_option(c("-s", "--simple-filenames"),
                action="store_true",
                default=FALSE,
                help="Use simple filenames."),
    make_option(c("-v", "--verbose"),
                action="store_true",
                default=FALSE,
                help="log more stuff")
  )

  desc <- "GRIN_sliding_window.R"
  opt <- parse_args(OptionParser(option_list=option_list,
                                description=desc),
                   convert_hyphens_to_underscores=TRUE)
  
  errors <- 0
  # Check whether all necessary arguments have been setby the user
  # Check opt$gene_ranks
  if (is.null(opt$gene_ranks)) {
    message("ERROR:: --gene_ranks is required but is not set.")
    errors <- errors + 1
  } else if (!file.exists(opt$gene_ranks)) {
    message("ERROR:: --gene_ranks must be an existing TSV file.")
    errors <= errors + 1
  }
  # Checl opt$null_ranks
  if (is.null(opt$null_ranks)) {
    message("ERROR:: --null_ranks is required but is not set.")
    errors <= errors + 1
  } else if (!file.exists(opt$null_ranks)) {
    message("ERROR:: --null_ranks must be an existing TSV file.")
    errors <= errors + 1
  }
  # Check outputs: opt$outdir.
  if(is.null(opt$outdir)) {
    message("ERROR:: --outdir is required but is not set.")
    errors <- errors+1
  }

  if (opt$verbose) {
    print(opt)
  }

  if (errors > 0) {
    quit()
  }

  return(opt)
}

load_ranks <- function(ranks_file) {

  ranks <- read.table(ranks_file, header = T, sep = "\t", fill=TRUE)

  return(ranks)
}

write_table = function(table, path) {
  # Create out_dir if it doesn't exist (avoid warning message if out_dir exists).
  out_dir = dirname(path)
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive=TRUE)
  }
  # Save the table.
  write.table(table, 
              path,
              sep = "\t",
              quote=F,
              col.names = T,
              row.names = F)
}

get_file_path = function(..., outdir=NULL, ext='.tsv') {
  filename = paste(..., sep='_')
  filename = paste0(filename, ext)
  if (!is.null(outdir)) {
    filename = file.path(outdir, filename)
  }
  return(filename)
}

# Calculate sliding window Mann-Whitney U test p-values and elbow of resulting
# curve to determine cutoff point
mannWhitneyWindow <- function(nullRanks, scores) {
  # Make sliding window of size 0.15 * number of genes
  numGenes <- nrow(scores)
  windowSize <- round(numGenes * 0.15, digits = 0)
  
  windowMatrix <- foreach(i = 1:(numGenes-windowSize), .combine = 'rbind') %do% {
    winstart  <- i
    winend    <- winstart+windowSize
    window    <- dplyr::slice(scores, winstart:winend)
    window.null <- dplyr::slice(nullRanks, winstart:winend)
    # Calculate Mann-Whitney U test (two-sample Wilcoxon rank sum test)
    test <- wilcox.test(window$rank, window.null$rank, alternative = "less", paired = F)
    df <- data.frame(window = i, p = test$p.value)
    colnames(df) <- c("Window", "p")
    df
  }
  return(windowMatrix)
}

elbowFilter <- function(scores, windowMatrix) {
  # Calculate elbow from sliding window and round to nearest whole number 
  elbow <- KneeArrower::findCutoff(windowMatrix$Window, windowMatrix$p, method = "first")
  elbowRound <- round(elbow$x, digits = 0)
  
  # Now filter genes into retained and removed gene sets
  retainedGenes <- dplyr::filter(scores, rank_position < elbowRound) %>%
    dplyr::mutate(set = "Retained") %>% dplyr::select(-rank_position)
  removedGenes <- dplyr::filter(scores, rank_position >= elbowRound) %>%
    dplyr::mutate(set = "Removed") %>% dplyr::select(-rank_position)
  
  filteredGenes <- list(retainedGenes, removedGenes, elbowRound)
  names(filteredGenes) <- c("Retained_Genes", "Removed_Genes", "Elbow")
  return(filteredGenes)
}

save_plot <- function(scores, windowMatrix, elbow, opt) {
  message("Generating and saving elbow plot...")
  
  # Elbow plots for ranked and randomly removed genes
  outplot <- ggplot(windowMatrix) + 
    geom_line(aes(x=Window, y=p), size=1) + 
    geom_vline(xintercept=elbow, linetype="dashed") + 
    labs(title=scores$setid[1],
         subtitle="Sliding window Mann-Whitney U Test") + 
    theme_light() + theme(axis.text.x = element_text(size=6))
    
  if (opt$simple_filenames) {
    filepath = get_file_path('GRIN-elbow-plots',opt$outdir, ext='.png')
  } else {
    filepath = get_file_path("GRIN", opt$modname, "_elbow_plot", 
                             outdir=opt$outdir, ext='.png')
  }

  png(filename=filepath, width=800, height=800)
  print(outplot)
  dev.off()
}

main <- function() {
  # parse commnd line arguments
  opt <- parse_arguments()

  suppressPackageStartupMessages({
    library(foreach)
    library(dplyr)
    library(KneeArrower)
  })

  # Load input gene set ranks and null distribution ranks
  gene_ranks <- load_ranks(opt$gene_ranks)
  null_ranks <- load_ranks(opt$null_ranks)

  # Throw error if geneset contains less than 4 genes
  if (nrow(gene_ranks) < 4) {
    message('ERROR: At least 4 genes must be in the gene_ranks for GRIN to work.')
    message('ERROR: Your gene_ranks:')
    print(head(gene_ranks))
    message(paste0("Number of genes in gene_ranks: ", nrow(gene_ranks)))
    return(0)
  }

  if (nrow(gene_ranks) != nrow(null_ranks)) {
    message('ERROR: gene_ranks and null_ranks are not the same size.')
    message(paste0("gene_ranks size: ", nrow(gene_ranks), " null_ranks size: ", nrow(null_ranks)))
    return(0)
  }

  # Break ties in rank for sliding window comparison to null distribution
  gene_ranks <- gene_ranks %>% dplyr::mutate(rank = rank + if_else(duplicated(rank), runif( n(),0,1 ), 0)) %>%
    dplyr::select(rank, INDEX) %>% dplyr::relocate(INDEX, .before = rank)

  # Add rank position next to each gene from input gene set
  for (i in 1:(nrow(gene_ranks))) gene_ranks$rank_position[i] <- i
  
  # Compute Mann-Whitney U test with sliding window
  windowMatrix <- mannWhitneyWindow(null_ranks, gene_ranks)
  
  # Compute elbow and filter genes into retained and removed gene sets
  filteredGenes <- elbowFilter(gene_ranks, windowMatrix)

  # Write retained and removed gene sets to file
  retainedPath <- get_file_path(opt$modname, "GRIN", 
                                "Retained_Genes", outdir=opt$outdir, ext='.tsv')
  removedPath <- get_file_path(opt$modname, "GRIN",
                               "Removed_Genes", outdir=opt$outdir, ext='.tsv')
  write_table(filteredGenes$Retained_Genes,retainedPath)
  write_table(filteredGenes$Removed_Genes,removedPath)
  
  # If flag present, plot elbow plot and save sliding window matrix
  if(opt$plot) {
    suppressPackageStartupMessages(library(ggplot2))
    message("Making plot and saving sliding window matrix...")
    windowPath <- get_file_path("GRIN", opt$modname,
                                "Window_Matrix", outdir=opt$outdir, ext='.txt')
    write_table(windowMatrix, windowPath)
    save_plot(gene_ranks, windowMatrix, filteredGenes$Elbow, opt)
  }

  message(paste0("COMPLETED GRIN++: ",opt$modname))
  return(0)
}

status = main()
quit(save='no', status=status)