#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(Giotto))

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

select_marker_genes <- function(sc_count, labels, cell_id_col, cell_type_col, cell_types, top_markers, pseudo_count) {
  labels <- labels[match(rownames(sc_count), labels[[cell_id_col]]), , drop = FALSE]
  means <- sapply(cell_types, function(cell_type) {
    cells <- labels[[cell_type_col]] == cell_type
    if (!any(cells)) {
      rep(0, ncol(sc_count))
    } else {
      colMeans(sc_count[cells, , drop = FALSE])
    }
  })
  rownames(means) <- colnames(sc_count)
  marker_rows <- list()
  marker_genes <- character()
  for (cell_type in cell_types) {
    type_mean <- means[, cell_type]
    other_types <- setdiff(cell_types, cell_type)
    other_mean <- if (length(other_types) == 1) means[, other_types] else rowMeans(means[, other_types, drop = FALSE])
    score <- log2(type_mean + pseudo_count) - log2(other_mean + pseudo_count)
    ord <- order(score, decreasing = TRUE)
    chosen <- rownames(means)[ord][seq_len(min(top_markers, length(ord)))]
    marker_genes <- c(marker_genes, chosen)
    marker_rows[[cell_type]] <- data.frame(
      cell_type = cell_type,
      gene = chosen,
      score = as.numeric(score[chosen]),
      mean_in_type = as.numeric(type_mean[chosen]),
      mean_other = as.numeric(other_mean[chosen]),
      check.names = FALSE
    )
  }
  list(markers = unique(marker_genes), marker_table = do.call(rbind, marker_rows), means = means)
}

make_spot_clusters <- function(spot_ids, cluster_size) {
  n <- length(spot_ids)
  cluster_size <- max(2, as.integer(cluster_size))
  clusters <- paste0("dwls_cluster_", ceiling(seq_len(n) / cluster_size))
  tab <- table(clusters)
  if (length(tab) > 1 && tail(tab, 1) == 1) {
    clusters[clusters == names(tail(tab, 1))] <- names(tab)[length(tab) - 1]
  }
  clusters
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
  top_markers <- as.integer(ifelse(is.null(args[["top-markers"]]), "25", args[["top-markers"]]))
  n_cell <- as.numeric(ifelse(is.null(args[["n-cell"]]), "50", args[["n-cell"]]))
  cutoff <- as.numeric(ifelse(is.null(args[["cutoff"]]), "2", args[["cutoff"]]))
  cluster_size <- as.integer(ifelse(is.null(args[["cluster-size"]]), "64", args[["cluster-size"]]))
  pseudo_count <- as.numeric(ifelse(is.null(args[["pseudo-count"]]), "1e-6", args[["pseudo-count"]]))
  eps <- as.numeric(ifelse(is.null(args[["eps"]]), "1e-8", args[["eps"]]))
  max_spots <- as.integer(ifelse(is.null(args[["max-spots"]]), "0", args[["max-spots"]]))

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  genes <- read_list(genes_path)
  cell_types <- read_list(cell_types_path)

  spatial_count <- align_expression(spatial_expression, spot_id_col, genes)
  spot_ids_all <- rownames(spatial_count)
  spatial_run <- spatial_count
  if (!is.null(max_spots) && !is.na(max_spots) && max_spots > 0 && nrow(spatial_run) > max_spots) {
    spatial_run <- spatial_run[seq_len(max_spots), , drop = FALSE]
  }

  labels <- read.csv(scrna_labels, check.names = FALSE)
  if (!(cell_id_col %in% colnames(labels)) || !(cell_type_col %in% colnames(labels))) {
    stop("scRNA labels must contain cell id and cell type columns")
  }
  labels[[cell_id_col]] <- as.character(labels[[cell_id_col]])
  labels[[cell_type_col]] <- as.character(labels[[cell_type_col]])
  sc_count_all <- align_expression(scrna_expression, cell_id_col, genes)
  shared_cells <- intersect(rownames(sc_count_all), labels[[cell_id_col]])
  if (length(shared_cells) == 0) {
    stop("No scRNA cells overlap labels")
  }
  labels <- labels[match(shared_cells, labels[[cell_id_col]]), , drop = FALSE]
  sc_count <- sc_count_all[shared_cells, , drop = FALSE]

  marker_info <- select_marker_genes(sc_count, labels, cell_id_col, cell_type_col, cell_types, top_markers, pseudo_count)
  signature_genes <- intersect(marker_info$markers, colnames(spatial_count))
  if (length(signature_genes) < 2) {
    stop("Giotto SpatialDWLS requires at least two signature genes")
  }
  signature <- marker_info$means[signature_genes, cell_types, drop = FALSE]
  signature <- as.matrix(signature)
  signature[signature < eps] <- eps
  expression_matrix <- t(spatial_run[, signature_genes, drop = FALSE])

  spot_ids_run <- rownames(spatial_run)
  spatial_locs <- data.frame(cell_ID = spot_ids_run, sdimx = seq_along(spot_ids_run), sdimy = 0)
  cell_metadata <- data.frame(
    cell_ID = spot_ids_run,
    dwls_cluster = make_spot_clusters(spot_ids_run, cluster_size),
    check.names = FALSE
  )
  gobject <- createGiottoObject(
    expression = expression_matrix,
    spatial_locs = spatial_locs,
    cell_metadata = cell_metadata,
    verbose = FALSE
  )
  gobject <- normalizeGiotto(
    gobject = gobject,
    expression_values = "raw",
    name = "normalized",
    verbose = FALSE
  )
  enrichment <- runDWLSDeconv(
    gobject = gobject,
    expression_values = "normalized",
    cluster_column = "dwls_cluster",
    sign_matrix = signature,
    n_cell = n_cell,
    cutoff = cutoff,
    name = "DWLS",
    return_gobject = FALSE
  )
  raw <- as.data.frame(enrichment@enrichDT, check.names = FALSE)
  rownames(raw) <- raw[["cell_ID"]]
  raw_results <- raw[, setdiff(colnames(raw), "cell_ID"), drop = FALSE]
  weights <- normalize_and_align(raw_results, spot_ids_all, cell_types)

  proportions_path <- file.path(output_dir, "spatialdwls_giotto_proportions.csv")
  out <- data.frame(spot_id = rownames(weights), weights, check.names = FALSE)
  write.csv(out, proportions_path, row.names = FALSE)

  signature_path <- file.path(output_dir, "spatialdwls_giotto_signature_matrix.csv")
  write.csv(data.frame(gene = rownames(signature), signature, check.names = FALSE), signature_path, row.names = FALSE)
  marker_path <- file.path(output_dir, "spatialdwls_giotto_marker_genes.csv")
  write.csv(marker_info$marker_table, marker_path, row.names = FALSE)

  manifest <- list(
    method = "SpatialDWLS/Seurat",
    implementation = "Giotto_runDWLSDeconv_R4.1.0_branch",
    spatial_expression = spatial_expression,
    scrna_expression = scrna_expression,
    scrna_labels = scrna_labels,
    genes = genes_path,
    cell_types = cell_types_path,
    top_markers = top_markers,
    n_cell = n_cell,
    cutoff = cutoff,
    cluster_size = cluster_size,
    pseudo_count = pseudo_count,
    eps = eps,
    max_spots = max_spots,
    num_spots_input = nrow(spatial_run),
    num_spots_output = nrow(weights),
    num_genes = length(genes),
    num_signature_genes = length(signature_genes),
    num_reference_cells = length(shared_cells),
    num_cell_types = length(cell_types),
    package_versions = list(
      Seurat = as.character(utils::packageVersion("Seurat")),
      Giotto = as.character(utils::packageVersion("Giotto")),
      GiottoClass = as.character(utils::packageVersion("GiottoClass")),
      GiottoUtils = as.character(utils::packageVersion("GiottoUtils")),
      GiottoVisuals = as.character(utils::packageVersion("GiottoVisuals")),
      Rfast = as.character(utils::packageVersion("Rfast")),
      quadprog = as.character(utils::packageVersion("quadprog"))
    ),
    proportions_path = proportions_path,
    signature_path = signature_path,
    marker_path = marker_path
  )
  jsonlite::write_json(manifest, file.path(output_dir, "spatialdwls_giotto_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
}

main()
