#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(spacexr))

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
  x_col <- ifelse(is.null(args[["x-col"]]), "x", args[["x-col"]])
  y_col <- ifelse(is.null(args[["y-col"]]), "y", args[["y-col"]])
  doublet_mode <- ifelse(is.null(args[["doublet-mode"]]), "multi", args[["doublet-mode"]])
  max_cores <- as.integer(ifelse(is.null(args[["max-cores"]]), "8", args[["max-cores"]]))
  n_max_cells <- as.integer(ifelse(is.null(args[["n-max-cells"]]), "30000", args[["n-max-cells"]]))
  min_umi_reference <- as.integer(ifelse(is.null(args[["min-umi-reference"]]), "1", args[["min-umi-reference"]]))
  umi_min_spatial <- as.integer(ifelse(is.null(args[["umi-min-spatial"]]), "1", args[["umi-min-spatial"]]))
  max_multi_types <- as.integer(ifelse(is.null(args[["max-multi-types"]]), "4", args[["max-multi-types"]]))

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  genes <- read_list(genes_path)
  cell_types <- read_list(cell_types_path)

  coords <- read.csv(spatial_coords, check.names = FALSE)
  if (!(spot_id_col %in% colnames(coords)) || !(x_col %in% colnames(coords)) || !(y_col %in% colnames(coords))) {
    stop("Coordinate table is missing spot/x/y columns")
  }
  coords[[spot_id_col]] <- as.character(coords[[spot_id_col]])
  spot_ids <- coords[[spot_id_col]]
  coord_mat <- data.frame(
    x = as.numeric(coords[[x_col]]),
    y = as.numeric(coords[[y_col]]),
    row.names = spot_ids
  )

  spatial_counts <- align_expression(spatial_expression, spot_id_col, genes, ids = spot_ids)
  spatial_numi <- colSums(spatial_counts)

  labels <- read.csv(scrna_labels, check.names = FALSE)
  if (!(cell_id_col %in% colnames(labels)) || !(cell_type_col %in% colnames(labels))) {
    stop("scRNA label table is missing cell id or cell type columns")
  }
  labels[[cell_id_col]] <- as.character(labels[[cell_id_col]])
  labels[[cell_type_col]] <- as.character(labels[[cell_type_col]])
  labels <- labels[labels[[cell_type_col]] %in% cell_types, , drop = FALSE]

  scrna_counts_all <- align_expression(scrna_expression, cell_id_col, genes)
  shared_cells <- intersect(colnames(scrna_counts_all), labels[[cell_id_col]])
  if (length(shared_cells) == 0) {
    stop("No scRNA cells overlap labels")
  }
  labels <- labels[match(shared_cells, labels[[cell_id_col]]), , drop = FALSE]
  scrna_counts <- scrna_counts_all[, shared_cells, drop = FALSE]
  scrna_cell_types <- factor(labels[[cell_type_col]], levels = cell_types)
  names(scrna_cell_types) <- shared_cells
  scrna_numi <- colSums(scrna_counts)

  reference <- Reference(
    counts = scrna_counts,
    cell_types = scrna_cell_types,
    nUMI = scrna_numi,
    require_int = TRUE,
    n_max_cells = n_max_cells,
    min_UMI = min_umi_reference
  )
  spatial <- SpatialRNA(
    coords = coord_mat,
    counts = spatial_counts,
    nUMI = spatial_numi,
    require_int = TRUE
  )
  rctd <- create.RCTD(
    spatial,
    reference,
    max_cores = max_cores,
    UMI_min = umi_min_spatial,
    CELL_MIN_INSTANCE = 1,
    MAX_MULTI_TYPES = max_multi_types,
    keep_reference = FALSE
  )
  rctd <- run.RCTD(rctd, doublet_mode = doublet_mode)

  weights <- NULL
  rctd_spot_ids <- colnames(rctd@spatialRNA@counts)
  if (!is.null(rctd@results$weights)) {
    weights <- rctd@results$weights
  } else if (!is.null(rctd@results$weights_unconfident)) {
    weights <- rctd@results$weights_unconfident
  } else if (is.list(rctd@results) && length(rctd@results) == length(rctd_spot_ids)) {
    weights <- matrix(0, nrow = length(rctd_spot_ids), ncol = length(cell_types))
    rownames(weights) <- rctd_spot_ids
    colnames(weights) <- cell_types
    for (i in seq_along(rctd@results)) {
      result <- rctd@results[[i]]
      if (!is.null(result$sub_weights) && !is.null(result$cell_type_list)) {
        values <- as.numeric(result$sub_weights)
        names(values) <- as.character(result$cell_type_list)
      } else if (!is.null(result$all_weights)) {
        values <- as.numeric(result$all_weights)
        if (!is.null(names(result$all_weights))) {
          names(values) <- names(result$all_weights)
        } else {
          names(values) <- cell_types[seq_along(values)]
        }
      } else {
        values <- numeric(0)
      }
      shared_types <- intersect(names(values), cell_types)
      weights[i, shared_types] <- values[shared_types]
    }
  } else {
    stop(paste("RCTD result does not contain an extractable weight matrix. Result keys:", paste(names(rctd@results), collapse = ",")))
  }
  weights <- as.matrix(weights)
  if (nrow(weights) != length(spot_ids) && ncol(weights) == length(spot_ids)) {
    weights <- t(weights)
  }
  if (is.null(rownames(weights))) {
    rownames(weights) <- spot_ids
  }
  missing_rctd_spots <- setdiff(spot_ids, rownames(weights))
  if (length(missing_rctd_spots) > 0) {
    missing_weights <- matrix(0, nrow = length(missing_rctd_spots), ncol = ncol(weights))
    rownames(missing_weights) <- missing_rctd_spots
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
  weights <- weights / row_sums

  proportions_path <- file.path(output_dir, "rctd_proportions.csv")
  out <- data.frame(spot_id = rownames(weights), weights, check.names = FALSE)
  write.csv(out, proportions_path, row.names = FALSE)

  manifest <- list(
    method = "RCTD",
    package = "spacexr",
    package_version = as.character(utils::packageVersion("spacexr")),
    spatial_expression = spatial_expression,
    spatial_coords = spatial_coords,
    scrna_expression = scrna_expression,
    scrna_labels = scrna_labels,
    genes = genes_path,
    cell_types = cell_types_path,
    doublet_mode = doublet_mode,
    max_cores = max_cores,
    n_max_cells = n_max_cells,
    min_umi_reference = min_umi_reference,
    umi_min_spatial = umi_min_spatial,
    max_multi_types = max_multi_types,
    num_spots = length(spot_ids),
    num_genes = length(genes),
    num_reference_cells = length(shared_cells),
    proportions_path = proportions_path
  )
  jsonlite::write_json(manifest, file.path(output_dir, "rctd_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
}

main()
