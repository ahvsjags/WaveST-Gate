#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(CARD))

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
  rownames(mat) <- frame[[id_col]]
  t(mat)
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

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  spatial_expression <- require_arg(args, "spatial-expression")
  spatial_coords <- require_arg(args, "spatial-coords")
  scrna_expression <- require_arg(args, "scrna-expression")
  scrna_labels <- require_arg(args, "scrna-labels")
  genes_path <- require_arg(args, "genes")
  cell_types_path <- require_arg(args, "cell-types")
  output_dir <- require_arg(args, "output-dir")

  spot_id_col <- ifelse(is.null(args[["spot-id-col"]]), "spot_id", args[["spot-id-col"]])
  cell_id_col <- ifelse(is.null(args[["cell-id-col"]]), "cell_id", args[["cell-id-col"]])
  cell_type_col <- ifelse(is.null(args[["cell-type-col"]]), "cell_type", args[["cell-type-col"]])
  sample_col <- ifelse(is.null(args[["sample-col"]]), "sample_id", args[["sample-col"]])
  x_col <- ifelse(is.null(args[["x-col"]]), "x", args[["x-col"]])
  y_col <- ifelse(is.null(args[["y-col"]]), "y", args[["y-col"]])
  min_count_gene <- as.integer(ifelse(is.null(args[["min-count-gene"]]), "1", args[["min-count-gene"]]))
  min_count_spot <- as.integer(ifelse(is.null(args[["min-count-spot"]]), "1", args[["min-count-spot"]]))

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  genes <- read_list(genes_path)
  cell_types <- read_list(cell_types_path)

  coords <- read.csv(spatial_coords, check.names = FALSE)
  if (!(spot_id_col %in% colnames(coords)) || !(x_col %in% colnames(coords)) || !(y_col %in% colnames(coords))) {
    stop("Coordinate table is missing spot/x/y columns")
  }
  coords[[spot_id_col]] <- as.character(coords[[spot_id_col]])
  spot_ids <- coords[[spot_id_col]]
  spatial_location <- data.frame(
    x = as.numeric(coords[[x_col]]),
    y = as.numeric(coords[[y_col]]),
    row.names = spot_ids
  )

  spatial_count <- align_expression(spatial_expression, spot_id_col, genes, ids = spot_ids)

  labels <- read.csv(scrna_labels, check.names = FALSE)
  if (!(cell_id_col %in% colnames(labels)) || !(cell_type_col %in% colnames(labels))) {
    stop("scRNA label table is missing cell id or cell type columns")
  }
  labels[[cell_id_col]] <- as.character(labels[[cell_id_col]])
  labels[[cell_type_col]] <- as.character(labels[[cell_type_col]])
  labels <- labels[labels[[cell_type_col]] %in% cell_types, , drop = FALSE]

  sc_count_all <- align_expression(scrna_expression, cell_id_col, genes)
  shared_cells <- intersect(colnames(sc_count_all), labels[[cell_id_col]])
  if (length(shared_cells) == 0) {
    stop("No scRNA cells overlap labels")
  }
  labels <- labels[match(shared_cells, labels[[cell_id_col]]), , drop = FALSE]
  sc_count <- sc_count_all[, shared_cells, drop = FALSE]

  sc_meta <- data.frame(
    cellType = as.character(labels[[cell_type_col]]),
    sampleInfo = if (sample_col %in% colnames(labels)) as.character(labels[[sample_col]]) else "Sample",
    row.names = shared_cells,
    check.names = FALSE
  )

  card <- createCARDObject(
    sc_count = sc_count,
    sc_meta = sc_meta,
    spatial_count = spatial_count,
    spatial_location = spatial_location,
    ct.varname = "cellType",
    ct.select = cell_types,
    sample.varname = "sampleInfo",
    minCountGene = min_count_gene,
    minCountSpot = min_count_spot
  )
  card <- CARD_deconvolution(CARD_object = card)

  weights <- normalize_and_align(card@Proportion_CARD, spot_ids, cell_types)
  proportions_path <- file.path(output_dir, "card_proportions.csv")
  out <- data.frame(spot_id = rownames(weights), weights, check.names = FALSE)
  write.csv(out, proportions_path, row.names = FALSE)

  manifest <- list(
    method = "CARD",
    package = "CARD",
    package_version = as.character(utils::packageVersion("CARD")),
    spatial_expression = spatial_expression,
    spatial_coords = spatial_coords,
    scrna_expression = scrna_expression,
    scrna_labels = scrna_labels,
    genes = genes_path,
    cell_types = cell_types_path,
    min_count_gene = min_count_gene,
    min_count_spot = min_count_spot,
    num_spots = length(spot_ids),
    num_genes = length(genes),
    num_reference_cells = length(shared_cells),
    num_card_spots_after_qc = nrow(card@Proportion_CARD),
    selected_phi = card@info_parameters$phi,
    proportions_path = proportions_path
  )
  jsonlite::write_json(manifest, file.path(output_dir, "card_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
}

main()
