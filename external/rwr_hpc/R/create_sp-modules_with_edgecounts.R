
parse_arguments <- function() {
  suppressPackageStartupMessages(library(optparse))

  option_list <- list(
    make_option(c("--shortest_paths"),
                action = "store",
                type = "character",
                help = "directory containing shortest paths or .tsv file containing list of directories"),
    make_option(c("--runtags"),
                action = "store",
                type = "character",
                help = "file containing a runtag per subnetwork. Each runtag is on a separate line"),
    make_option(c("--num_layers"),
                action = "store",
                type = "numeric",
                help = "number of layers in the multiplex"),
    make_option(c("--map"),
                action = "store",
                type = "character",
                help = "path to ensemble id and gene symbo map"),
    make_option(c("-o", "--outdir"),
                action = "store",
                default = "./",
                type = "character",
                help = "path to the output directory. A processed/ subdirectory will be created in outdir"),
    make_option(c("-v", "--verbose"),
                action = "store_true",
                default = FALSE,
                help = "log more stuff")
  )

  desc <- "create_sp-modules_with_edgecounts.R"
  opt <- parse_args(OptionParser(option_list = option_list,
                                 description = desc),
                    convert_hyphens_to_underscores = TRUE)

  is.wholenumber <- function(x, tol = .Machine$double.eps^0.5) {
    abs(x - round(x)) < tol
  }

  errors <- 0
  # Verify `shortest_paths` exists either as a directory or a file
  if (is.null(opt$shortest_paths)) {
    message("ERROR:: --shortest_paths is required but is not set.")
    errors <- errors + 1
  } else if (!(file.exists(opt$shortest_paths))) {
    message("ERROR:: --shortest_paths does not exist.")
    errors <- errors + 1
  }
  # Verify `runtag` is provided
  if (is.null(opt$runtags)) {
    message(("ERROR: --runtags is required but is not set."))
    errors <- errors + 1
  } else if (!file.exists((opt$shortest_paths))) {
    message(("ERROR:: --runtags does not exist."))
    errors <- errors + 1
  }

  # Verify `num_layers` is provided and valid
  if (is.null(opt$num_layers)) {
    message("ERROR:: --num_layers is required but is not set.")
    errors <- errors + 1
  } else if (opt$num_layers < 0 || !is.wholenumber(opt$num_layers)) {
    message("ERROR:: --num_layers must be positive integer.")
    errors <- errors + 1
  }

  # Verify map is provided and valid
  if (is.null(opt$map)) {
    message("ERROR:: --map is required but is not set.")
    errors <- errors + 1
  } else if (!(file.exists(opt$map))) {
    message("ERROR:: --map does not exist.")
    errors <- errors + 1
  }

  if (opt$verbose) {
    print(opt)
  }

  if (errors > 0) {
    quit()
  }

  return(opt)
}

# create shortest paths input data per clade

merged_with_edgecounts <- function(subnet, inv = FALSE, verbose = FALSE) {

  edges <- read.table(
    subnet,
    header = TRUE,
    sep = "\t",
    stringsAsFactors = FALSE
  ) %>% dplyr::select(from, to, type) %>% unique() %>% data.frame()

  g <- graph_from_data_frame(edges, directed = FALSE)
  E(g)$weight <- 1
  g_simpl <- simplify(
    g,
    remove.multiple = TRUE,
    remove.loops = TRUE,
    edge.attr.comb = list(type = function(x) paste(x, sep = '|', collapse = '|'), weight = "sum")
  )

  return(g_simpl)
}

main <- function() {
  # parse commnd line arguments
  opt <- parse_arguments()

  suppressPackageStartupMessages({
    library(igraph)
    library(data.table)
    library(dplyr)
  })

  # Create output directory
  processed_dir <- file.path(opt$outdir, "processed")
  if (!dir.exists(processed_dir)) {
    dir.create(processed_dir, recursive = TRUE)
  }

  # Check if shortest paths input is a file of a directory
  subnets <- c()
  if (file_test("-f", opt$shortest_paths)) {
    dirs_to_check <- read.table(
      opt$shortest_paths,
      header = FALSE,
      stringsAsFactors = FALSE
    )

    for (dir in dirs_to_check$V1) {
      # print(sp)
      sp = list.files(dir, full.names = TRUE)
      sp <- sp[which(grepl("shortest_paths.tsv", sp))]

      subnets <- c(subnets, sp)
    }
    print(dirs_to_check)

  } else {
    subnets = list.files(opt$shortest_paths, full.names = TRUE)
    
    subnets <- subnets[which(grepl("shortest_paths.tsv",subnets))]
  }

  print("read subnetworks")

  # Read in runtags
  runtags <- read.table(
    opt$runtags,
    sep = "\t",
    header = FALSE,
    stringsAsFactors = FALSE
  )
  print("read runtags")

  # Verify unique runtags
  unique_runtags <- unique(runtags$V1)
  
  if (nrow(runtags) != length(unique_runtags)) {
    stop("ERROR:: runtags are not unique")
  }
  runtags <- unique_runtags

  # Verify unique subnetwork
  if (length(subnets) != length(unique(subnets))) {
    stop("ERROR:: subnetworks are not unique")
  }
  # Verfiy the number of runtags match the number of subnetworks
  if (length(runtags) != length(subnets)) {
    print(runtags)
    print(subnets)
    stop("ERROR:: each subnetwork must have a runtag")
  }

  # Verify each subnetwork has a runtag
  # read in map
  map <- read.table(
    opt$map,
    header = TRUE,
    sep = "\t",
    stringsAsFactors = FALSE
  )
  print(head(map))

  for (i in seq_len(length(subnets))) {
    net <- subnets[i]
    runtag <- runtags[i]

    print(paste0("working on network: ", net))
    subgraph <- merged_with_edgecounts(net)
    net_name <- paste0("sp-", runtag, ".tsv")
    subgraph_name <- file.path(processed_dir, net_name)

    as_df <- igraph::as_data_frame(subgraph)
    as_df$weight <- as_df$weight / opt$num_layers
    as_df <- as_df[rev(order(as_df$weight)), ]
    as_df <- as_df[, c("from", "to", "type", "weight")]
    as_df$row_order <- seq_len(nrow(as_df))
    as_df <- merge(
      as_df,
      map,
      by.x = "from",
      by.y = "ensembl",
      all.x = TRUE
    ) %>% dplyr::select(symbol, to, type, weight, row_order)
    names(as_df) <- c("from", "to", "type", "weight", "row_order")
    as_df <- merge(
      as_df,
      map,
      by.x = "to",
      by.y = "ensembl",
      all.x = TRUE
    ) %>% dplyr::select(from, symbol, type, weight, row_order)
    names(as_df) <- c("from", "to", "type", "weight", "row_order")
    as_df <- as_df[order(as_df$row_order),] %>% dplyr::select(from,to,type,weight)
    as_df <- as_df %>% filter(!is.na(from) & !is.na(to)) %>% data.frame()
    data.table::fwrite(as_df, subgraph_name, quote = FALSE, sep = "\t")
  }

  return(0)
}

status <- main()
quit(save = "no", status = status)