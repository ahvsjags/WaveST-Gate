#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(quadprog))

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

solve_qp_nonnegative <- function(D, d, names_out) {
  D <- as.matrix(D)
  D <- (D + t(D)) / 2
  d <- as.numeric(d)
  A <- diag(nrow(D))
  bzero <- rep(0, nrow(D))
  out <- tryCatch(
    quadprog::solve.QP(Dmat = D, dvec = d, Amat = A, bvec = bzero)$solution,
    error = function(cond) NULL
  )
  if (is.null(out)) {
    D <- as.matrix(Matrix::nearPD(D, corr = FALSE)$mat)
    out <- tryCatch(
      quadprog::solve.QP(Dmat = D, dvec = d, Amat = A, bvec = bzero)$solution,
      error = function(cond) rep(0, length(d))
    )
  }
  out[out < 0] <- 0
  names(out) <- names_out
  out
}

solve_ols_internal <- function(S, B) {
  D <- t(S) %*% S
  d <- t(S) %*% B
  solve_qp_nonnegative(D, d, colnames(S))
}

solve_dampened_wls_j <- function(S, B, gold_standard, j, eps) {
  pred <- as.vector(S %*% gold_standard)
  pred[pred < eps] <- eps
  ws <- (1 / pred)^2
  finite <- is.finite(ws)
  if (!any(finite)) {
    ws <- rep(1, length(pred))
  } else {
    ws[!finite] <- max(ws[finite])
  }
  ws_scaled <- ws / max(min(ws[ws > 0]), eps)
  multiplier <- 2^(j - 1)
  ws_dampened <- pmin(ws_scaled, multiplier)
  SW <- S * sqrt(ws_dampened)
  BW <- B * sqrt(ws_dampened)
  D <- t(SW) %*% SW
  d <- t(SW) %*% BW
  sc <- norm(D, "2")
  if (!is.finite(sc) || sc <= eps) {
    sc <- 1
  }
  solve_qp_nonnegative(D / sc, d / sc, colnames(S))
}

solve_dampened_wls <- function(S, B, j, eps, max_iter, tol) {
  solution <- solve_ols_internal(S, B)
  if (sum(solution) <= eps) {
    solution <- rep(1 / ncol(S), ncol(S))
    names(solution) <- colnames(S)
  }
  for (iter in seq_len(max_iter)) {
    new_solution <- solve_dampened_wls_j(S, B, solution, j, eps)
    averaged <- rowMeans(cbind(new_solution, matrix(solution, nrow = length(solution), ncol = 4)))
    change <- norm(as.matrix(averaged - solution))
    solution <- averaged
    if (!is.finite(change) || change <= tol) {
      break
    }
  }
  solution[solution < 0] <- 0
  if (sum(solution) <= eps) {
    solution <- rep(1 / ncol(S), ncol(S))
  } else {
    solution <- solution / sum(solution)
  }
  names(solution) <- colnames(S)
  solution
}

deconvolve_spot <- function(signature, spot_expr, j, eps, max_iter, tol, n_cell) {
  if (sum(spot_expr) <= eps) {
    out <- rep(1 / ncol(signature), ncol(signature))
    names(out) <- colnames(signature)
    return(out)
  }
  first <- solve_dampened_wls(signature, spot_expr, j, eps, max_iter, tol)
  present <- names(first)[first >= (1 / n_cell)]
  if (length(present) == 0) {
    present <- names(first)[which.max(first)]
  }
  if (length(present) == 1) {
    out <- rep(0, ncol(signature))
    names(out) <- colnames(signature)
    out[present] <- 1
    return(out)
  }
  second <- solve_dampened_wls(signature[, present, drop = FALSE], spot_expr, j, eps, max_iter, tol)
  out <- rep(0, ncol(signature))
  names(out) <- colnames(signature)
  out[names(second)] <- second
  out
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
  dampening_j <- as.numeric(ifelse(is.null(args[["dampening-j"]]), "2", args[["dampening-j"]]))
  max_iter <- as.integer(ifelse(is.null(args[["max-iter"]]), "100", args[["max-iter"]]))
  tol <- as.numeric(ifelse(is.null(args[["tol"]]), "0.01", args[["tol"]]))
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
    stop("scRNA label table is missing cell id or cell type columns")
  }
  labels[[cell_id_col]] <- as.character(labels[[cell_id_col]])
  labels[[cell_type_col]] <- as.character(labels[[cell_type_col]])
  labels <- labels[labels[[cell_type_col]] %in% cell_types, , drop = FALSE]

  sc_count_all <- align_expression(scrna_expression, cell_id_col, genes)
  shared_cells <- intersect(rownames(sc_count_all), labels[[cell_id_col]])
  if (length(shared_cells) == 0) {
    stop("No scRNA cells overlap labels")
  }
  labels <- labels[match(shared_cells, labels[[cell_id_col]]), , drop = FALSE]
  sc_count <- sc_count_all[shared_cells, , drop = FALSE]

  marker_info <- select_marker_genes(sc_count, labels, cell_id_col, cell_type_col, cell_types, top_markers, pseudo_count)
  signature_genes <- intersect(marker_info$markers, colnames(spatial_count))
  signature <- marker_info$means[signature_genes, cell_types, drop = FALSE]
  signature <- as.matrix(signature)
  signature[signature < eps] <- eps
  spatial_signature <- spatial_run[, signature_genes, drop = FALSE]

  raw_results <- matrix(0, nrow = nrow(spatial_signature), ncol = length(cell_types))
  rownames(raw_results) <- rownames(spatial_signature)
  colnames(raw_results) <- cell_types
  for (i in seq_len(nrow(spatial_signature))) {
    raw_results[i, ] <- deconvolve_spot(
      signature = signature,
      spot_expr = as.numeric(spatial_signature[i, ]),
      j = dampening_j,
      eps = eps,
      max_iter = max_iter,
      tol = tol,
      n_cell = n_cell
    )
  }

  weights <- normalize_and_align(raw_results, spot_ids_all, cell_types)
  proportions_path <- file.path(output_dir, "spatialdwls_proportions.csv")
  out <- data.frame(spot_id = rownames(weights), weights, check.names = FALSE)
  write.csv(out, proportions_path, row.names = FALSE)

  signature_path <- file.path(output_dir, "spatialdwls_signature_matrix.csv")
  write.csv(data.frame(gene = rownames(signature), signature, check.names = FALSE), signature_path, row.names = FALSE)
  marker_path <- file.path(output_dir, "spatialdwls_marker_genes.csv")
  write.csv(marker_info$marker_table, marker_path, row.names = FALSE)

  manifest <- list(
    method = "SpatialDWLS",
    implementation = "standalone_quadprog_runner",
    provenance = list(
      giotto_runDWLSDeconv = "https://giottosuite.com/reference/runDWLSDeconv.html",
      dwls_source = "https://github.com/dtsoucas/DWLS",
      spatialdwls_paper = "https://doi.org/10.1186/s13059-021-02362-7"
    ),
    package_versions = list(
      Matrix = as.character(utils::packageVersion("Matrix")),
      quadprog = as.character(utils::packageVersion("quadprog"))
    ),
    spatial_expression = spatial_expression,
    scrna_expression = scrna_expression,
    scrna_labels = scrna_labels,
    genes = genes_path,
    cell_types = cell_types_path,
    top_markers = top_markers,
    n_cell = n_cell,
    dampening_j = dampening_j,
    max_iter = max_iter,
    tol = tol,
    pseudo_count = pseudo_count,
    eps = eps,
    max_spots = max_spots,
    num_spots_input = nrow(spatial_run),
    num_spots_output = nrow(weights),
    num_genes = length(genes),
    num_signature_genes = length(signature_genes),
    num_reference_cells = length(shared_cells),
    num_cell_types = length(cell_types),
    proportions_path = proportions_path,
    signature_path = signature_path,
    marker_path = marker_path
  )
  jsonlite::write_json(manifest, file.path(output_dir, "spatialdwls_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
}

main()
