#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(BayesPrism))

parse_args <- function(argv) {
  args <- list()
  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (!startsWith(key, "--")) {
      stop(paste("Unexpected argument:", key))
    }
    name <- substring(key, 3)
    if (i == length(argv) || startsWith(argv[[i + 1]], "--")) {
      args[[name]] <- TRUE
      i <- i + 1
    } else {
      args[[name]] <- argv[[i + 1]]
      i <- i + 2
    }
  }
  args
}

require_arg <- function(args, name) {
  value <- args[[name]]
  if (is.null(value) || value == "") {
    stop(paste("Missing required argument --", name, sep = ""))
  }
  value
}

read_list <- function(path) {
  values <- readLines(path, warn = FALSE)
  values[nzchar(values)]
}

as_bool <- function(value, default) {
  if (is.null(value)) {
    return(default)
  }
  tolower(as.character(value)) %in% c("1", "true", "yes", "y")
}

align_expression <- function(path, id_col, genes, ids = NULL) {
  frame <- read.csv(path, check.names = FALSE)
  if (!(id_col %in% colnames(frame))) {
    stop(paste(path, "is missing id column", id_col))
  }
  missing_genes <- setdiff(genes, colnames(frame))
  if (length(missing_genes) > 0) {
    stop(paste("Expression table is missing genes:", paste(head(missing_genes, 10), collapse = ",")))
  }
  frame[[id_col]] <- as.character(frame[[id_col]])
  if (!is.null(ids)) {
    frame <- frame[match(ids, frame[[id_col]]), , drop = FALSE]
    if (any(is.na(frame[[id_col]]))) {
      stop("Expression table does not contain all requested ids")
    }
  }
  mat <- as.matrix(frame[, genes, drop = FALSE])
  storage.mode(mat) <- "double"
  mat <- round(mat)
  mat[mat < 0] <- 0
  rownames(mat) <- frame[[id_col]]
  mat
}

normalize_and_align <- function(weights, spot_ids, cell_types) {
  weights <- as.matrix(weights)
  if (nrow(weights) != length(spot_ids) && ncol(weights) == length(spot_ids)) {
    weights <- t(weights)
  }
  if (is.null(rownames(weights))) {
    rownames(weights) <- spot_ids[seq_len(nrow(weights))]
  }
  missing_spots <- setdiff(spot_ids, rownames(weights))
  if (length(missing_spots) > 0) {
    missing_weights <- matrix(0, nrow = length(missing_spots), ncol = ncol(weights))
    rownames(missing_weights) <- missing_spots
    colnames(missing_weights) <- colnames(weights)
    weights <- rbind(weights, missing_weights)
  }
  weights <- weights[spot_ids, , drop = FALSE]
  for (cell_type in cell_types) {
    if (!(cell_type %in% colnames(weights))) {
      weights <- cbind(weights, rep(0, nrow(weights)))
      colnames(weights)[ncol(weights)] <- cell_type
    }
  }
  weights <- weights[, cell_types, drop = FALSE]
  weights[is.na(weights)] <- 0
  weights[weights < 0] <- 0
  row_sums <- rowSums(weights)
  empty <- row_sums <= 1e-12
  if (any(empty)) {
    weights[empty, ] <- 1 / length(cell_types)
    row_sums <- rowSums(weights)
  }
  weights / row_sums
}

downsample_cells <- function(labels, cell_id_col, cell_type_col, max_cells_per_type, seed) {
  if (is.null(max_cells_per_type) || is.na(max_cells_per_type) || max_cells_per_type <= 0) {
    return(labels)
  }
  set.seed(seed)
  selected <- unlist(lapply(split(labels[[cell_id_col]], labels[[cell_type_col]]), function(ids) {
    if (length(ids) <= max_cells_per_type) {
      ids
    } else {
      sample(ids, max_cells_per_type)
    }
  }), use.names = FALSE)
  labels[labels[[cell_id_col]] %in% selected, , drop = FALSE]
}

make_reference <- function(sc_count, labels, cell_id_col, cell_type_col, input_type, cell_types) {
  if (input_type == "GEP") {
    labels <- labels[match(rownames(sc_count), labels[[cell_id_col]]), , drop = FALSE]
    gep <- rowsum(sc_count, group = labels[[cell_type_col]], reorder = FALSE)
    gep <- gep[cell_types, , drop = FALSE]
    return(list(
      reference = gep,
      cell.type.labels = rownames(gep),
      cell.state.labels = rownames(gep),
      num_reference_rows = nrow(gep),
      num_reference_cells = nrow(sc_count)
    ))
  }
  list(
    reference = sc_count,
    cell.type.labels = as.character(labels[[cell_type_col]]),
    cell.state.labels = as.character(labels[[cell_type_col]]),
    num_reference_rows = nrow(sc_count),
    num_reference_cells = nrow(sc_count)
  )
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  spatial_expression <- require_arg(args, "spatial-expression")
  scrna_expression <- require_arg(args, "scrna-expression")
  scrna_labels <- require_arg(args, "scrna-labels")
  genes_path <- require_arg(args, "genes")
  cell_types_path <- require_arg(args, "cell-types")
  output_dir <- require_arg(args, "output-dir")

  spot_id_col <- ifelse(is.null(args[["spot-id-col"]]), "spot_id", args[["spot-id-col"]])
  cell_id_col <- ifelse(is.null(args[["cell-id-col"]]), "cell_id", args[["cell-id-col"]])
  cell_type_col <- ifelse(is.null(args[["cell-type-col"]]), "cell_type", args[["cell-type-col"]])
  input_type <- ifelse(is.null(args[["input-type"]]), "GEP", args[["input-type"]])
  if (!(input_type %in% c("GEP", "count.matrix"))) {
    stop("--input-type must be GEP or count.matrix")
  }

  max_cells_per_type <- as.integer(ifelse(is.null(args[["max-cells-per-type"]]), "0", args[["max-cells-per-type"]]))
  max_spots <- as.integer(ifelse(is.null(args[["max-spots"]]), "0", args[["max-spots"]]))
  n_cores <- as.integer(ifelse(is.null(args[["n-cores"]]), "1", args[["n-cores"]]))
  chain_length <- as.integer(ifelse(is.null(args[["chain-length"]]), "1000", args[["chain-length"]]))
  burn_in <- as.integer(ifelse(is.null(args[["burn-in"]]), "500", args[["burn-in"]]))
  thinning <- as.integer(ifelse(is.null(args[["thinning"]]), "2", args[["thinning"]]))
  seed <- as.integer(ifelse(is.null(args[["seed"]]), "123", args[["seed"]]))
  alpha <- as.numeric(ifelse(is.null(args[["alpha"]]), "1", args[["alpha"]]))
  outlier_cut <- as.numeric(ifelse(is.null(args[["outlier-cut"]]), "1", args[["outlier-cut"]]))
  outlier_fraction <- as.numeric(ifelse(is.null(args[["outlier-fraction"]]), "1", args[["outlier-fraction"]]))
  pseudo_min <- as.numeric(ifelse(is.null(args[["pseudo-min"]]), "1e-8", args[["pseudo-min"]]))
  optimizer <- ifelse(is.null(args[["optimizer"]]), "MLE", args[["optimizer"]])
  maxit <- as.integer(ifelse(is.null(args[["maxit"]]), "100000", args[["maxit"]]))
  update_gibbs <- as_bool(args[["update-gibbs"]], TRUE)
  which_theta <- ifelse(is.null(args[["which-theta"]]), ifelse(update_gibbs, "final", "first"), args[["which-theta"]])
  key_arg <- ifelse(is.null(args[["key"]]), "", args[["key"]])
  key <- if (key_arg == "" || tolower(key_arg) == "null" || tolower(key_arg) == "none") NULL else key_arg

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  genes <- read_list(genes_path)
  cell_types <- read_list(cell_types_path)

  mixture_all <- align_expression(spatial_expression, spot_id_col, genes)
  spot_ids_all <- rownames(mixture_all)
  mixture <- mixture_all
  if (!is.null(max_spots) && !is.na(max_spots) && max_spots > 0 && nrow(mixture) > max_spots) {
    mixture <- mixture[seq_len(max_spots), , drop = FALSE]
  }
  zero_spots <- rowSums(mixture) <= 0
  if (any(zero_spots)) {
    mixture <- mixture[!zero_spots, , drop = FALSE]
  }
  if (nrow(mixture) == 0) {
    stop("No nonzero-expression spots remain for BayesPrism")
  }

  labels <- read.csv(scrna_labels, check.names = FALSE)
  if (!(cell_id_col %in% colnames(labels)) || !(cell_type_col %in% colnames(labels))) {
    stop("scRNA label table is missing cell id or cell type columns")
  }
  labels[[cell_id_col]] <- as.character(labels[[cell_id_col]])
  labels[[cell_type_col]] <- as.character(labels[[cell_type_col]])
  labels <- labels[labels[[cell_type_col]] %in% cell_types, , drop = FALSE]
  labels <- downsample_cells(labels, cell_id_col, cell_type_col, max_cells_per_type, seed)

  sc_count_all <- align_expression(scrna_expression, cell_id_col, genes)
  shared_cells <- intersect(rownames(sc_count_all), labels[[cell_id_col]])
  if (length(shared_cells) == 0) {
    stop("No scRNA cells overlap labels")
  }
  labels <- labels[match(shared_cells, labels[[cell_id_col]]), , drop = FALSE]
  sc_count <- sc_count_all[shared_cells, , drop = FALSE]

  reference_info <- make_reference(sc_count, labels, cell_id_col, cell_type_col, input_type, cell_types)
  prism <- new.prism(
    reference = reference_info$reference,
    input.type = input_type,
    cell.type.labels = reference_info$cell.type.labels,
    cell.state.labels = reference_info$cell.state.labels,
    key = key,
    mixture = mixture,
    outlier.cut = outlier_cut,
    outlier.fraction = outlier_fraction,
    pseudo.min = pseudo_min
  )
  bp <- run.prism(
    prism,
    n.cores = n_cores,
    update.gibbs = update_gibbs,
    gibbs.control = list(
      chain.length = chain_length,
      burn.in = burn_in,
      thinning = thinning,
      seed = seed,
      alpha = alpha
    ),
    opt.control = list(
      optimizer = optimizer,
      maxit = maxit
    )
  )
  theta <- get.fraction(bp, which.theta = which_theta, state.or.type = "type")
  weights <- normalize_and_align(theta, spot_ids_all, cell_types)
  proportions_path <- file.path(output_dir, "bayesprism_proportions.csv")
  out <- data.frame(spot_id = rownames(weights), weights, check.names = FALSE)
  write.csv(out, proportions_path, row.names = FALSE)

  manifest <- list(
    method = "BayesPrism",
    package = "BayesPrism",
    package_version = as.character(utils::packageVersion("BayesPrism")),
    spatial_expression = spatial_expression,
    scrna_expression = scrna_expression,
    scrna_labels = scrna_labels,
    genes = genes_path,
    cell_types = cell_types_path,
    input_type = input_type,
    key = if (is.null(key)) NA_character_ else key,
    update_gibbs = update_gibbs,
    which_theta = which_theta,
    optimizer = optimizer,
    chain_length = chain_length,
    burn_in = burn_in,
    thinning = thinning,
    alpha = alpha,
    seed = seed,
    n_cores = n_cores,
    maxit = maxit,
    outlier_cut = outlier_cut,
    outlier_fraction = outlier_fraction,
    pseudo_min = pseudo_min,
    max_cells_per_type = max_cells_per_type,
    max_spots = max_spots,
    num_spots_input = nrow(mixture),
    num_zero_expression_spots_skipped = sum(zero_spots),
    num_spots_output = nrow(weights),
    num_genes = length(genes),
    num_reference_cells = reference_info$num_reference_cells,
    num_reference_rows = reference_info$num_reference_rows,
    proportions_path = proportions_path
  )
  jsonlite::write_json(manifest, file.path(output_dir, "bayesprism_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
}

main()
