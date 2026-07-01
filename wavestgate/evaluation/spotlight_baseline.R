suppressPackageStartupMessages(library(SPOTlight))

parse_args <- function(argv) {
  args <- list()
  idx <- 1
  while (idx <= length(argv)) {
    key <- argv[[idx]]
    if (!startsWith(key, "--")) {
      stop("Unexpected argument: ", key)
    }
    name <- substring(key, 3)
    if (idx == length(argv) || startsWith(argv[[idx + 1]], "--")) {
      args[[name]] <- TRUE
      idx <- idx + 1
    } else {
      args[[name]] <- argv[[idx + 1]]
      idx <- idx + 2
    }
  }
  args
}

need <- function(args, key) {
  value <- args[[key]]
  if (is.null(value) || isTRUE(value) || value == "") {
    stop("Missing required argument --", key)
  }
  value
}

read_list <- function(path) {
  values <- readLines(path, warn = FALSE)
  values[nzchar(values)]
}

read_expression <- function(path, id_col, genes) {
  frame <- read.csv(path, check.names = FALSE)
  if (!(id_col %in% colnames(frame))) {
    stop(path, " is missing required id column ", id_col)
  }
  missing <- setdiff(genes, colnames(frame))
  if (length(missing) > 0) {
    stop(path, " is missing benchmark genes: ", paste(head(missing, 10), collapse = ", "))
  }
  ids <- as.character(frame[[id_col]])
  mat <- as.matrix(frame[, genes, drop = FALSE])
  mat[is.na(mat)] <- 0
  mat[mat < 0] <- 0
  rownames(mat) <- ids
  mat
}

subset_reference <- function(expr, labels, cell_id_col, cell_type_col, cell_types, max_cells_per_type, seed) {
  if (!(cell_id_col %in% colnames(labels)) || !(cell_type_col %in% colnames(labels))) {
    stop("scRNA labels are missing required columns")
  }
  labels <- labels[!duplicated(labels[[cell_id_col]]), c(cell_id_col, cell_type_col), drop = FALSE]
  labels[[cell_id_col]] <- as.character(labels[[cell_id_col]])
  labels[[cell_type_col]] <- as.character(labels[[cell_type_col]])
  common <- intersect(rownames(expr), labels[[cell_id_col]])
  labels <- labels[match(common, labels[[cell_id_col]]), , drop = FALSE]
  expr <- expr[common, , drop = FALSE]
  keep <- labels[[cell_type_col]] %in% cell_types
  labels <- labels[keep, , drop = FALSE]
  expr <- expr[keep, , drop = FALSE]
  if (nrow(expr) == 0) {
    stop("No labelled scRNA cells overlap benchmark cell types")
  }
  missing_types <- setdiff(cell_types, unique(labels[[cell_type_col]]))
  if (length(missing_types) > 0) {
    stop("scRNA reference is missing benchmark cell types: ", paste(missing_types, collapse = ", "))
  }
  if (!is.na(max_cells_per_type) && max_cells_per_type > 0) {
    set.seed(seed)
    keep_idx <- unlist(lapply(cell_types, function(ct) {
      idx <- which(labels[[cell_type_col]] == ct)
      if (length(idx) > max_cells_per_type) {
        sample(idx, max_cells_per_type)
      } else {
        idx
      }
    }), use.names = FALSE)
    keep_idx <- sort(unique(keep_idx))
    labels <- labels[keep_idx, , drop = FALSE]
    expr <- expr[keep_idx, , drop = FALSE]
  }
  list(expr = expr, labels = labels)
}

build_markers <- function(expr, labels, genes, cell_type_col, top_markers) {
  groups <- as.character(labels[[cell_type_col]])
  cell_types <- unique(groups)
  global_mean <- colMeans(expr, na.rm = TRUE)
  rows <- list()
  for (ct in cell_types) {
    in_ct <- groups == ct
    ct_mean <- colMeans(expr[in_ct, , drop = FALSE], na.rm = TRUE)
    other_mean <- if (sum(!in_ct) > 0) colMeans(expr[!in_ct, , drop = FALSE], na.rm = TRUE) else global_mean
    score <- log1p(ct_mean) - log1p(other_mean)
    score[is.na(score)] <- 0
    score <- score + 1e-6 * log1p(ct_mean)
    order_idx <- order(score, decreasing = TRUE)
    order_idx <- order_idx[score[order_idx] > 0]
    if (length(order_idx) == 0) {
      order_idx <- order(log1p(ct_mean), decreasing = TRUE)
    }
    selected <- genes[order_idx[seq_len(min(top_markers, length(order_idx)))]]
    selected_score <- score[selected]
    selected_score[selected_score <= 0 | is.na(selected_score)] <- min(selected_score[selected_score > 0], 1e-6)
    rows[[ct]] <- data.frame(
      gene = selected,
      cluster = ct,
      weight = as.numeric(selected_score),
      stringsAsFactors = FALSE
    )
  }
  unique(do.call(rbind, rows))
}

normalize_and_align <- function(mat, spot_ids, cell_types) {
  frame <- as.data.frame(mat, check.names = FALSE)
  frame <- frame[spot_ids, , drop = FALSE]
  for (ct in cell_types) {
    if (!(ct %in% colnames(frame))) {
      frame[[ct]] <- 0
    }
  }
  frame <- frame[, cell_types, drop = FALSE]
  values <- as.matrix(frame)
  values[is.na(values)] <- 0
  values[values < 0] <- 0
  sums <- rowSums(values)
  empty <- sums <= 1e-12
  if (any(empty)) {
    values[empty, ] <- 1 / length(cell_types)
    sums <- rowSums(values)
  }
  values <- values / pmax(sums, 1e-12)
  rownames(values) <- spot_ids
  colnames(values) <- cell_types
  values
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  spatial_expression <- need(args, "spatial-expression")
  scrna_expression <- need(args, "scrna-expression")
  scrna_labels <- need(args, "scrna-labels")
  genes_path <- need(args, "genes")
  cell_types_path <- need(args, "cell-types")
  output_dir <- need(args, "output-dir")
  spot_id_col <- ifelse(is.null(args[["spot-id-col"]]), "spot_id", args[["spot-id-col"]])
  cell_id_col <- ifelse(is.null(args[["cell-id-col"]]), "cell_id", args[["cell-id-col"]])
  cell_type_col <- ifelse(is.null(args[["cell-type-col"]]), "cell_type", args[["cell-type-col"]])
  top_markers <- as.integer(ifelse(is.null(args[["top-markers"]]), "25", args[["top-markers"]]))
  max_cells_per_type <- as.integer(ifelse(is.null(args[["max-cells-per-type"]]), "-1", args[["max-cells-per-type"]]))
  seed <- as.integer(ifelse(is.null(args[["seed"]]), "7", args[["seed"]]))
  nrun <- as.integer(ifelse(is.null(args[["nrun"]]), "1", args[["nrun"]]))
  max_iter <- as.integer(ifelse(is.null(args[["max-iter"]]), "200", args[["max-iter"]]))
  min_prop <- as.numeric(ifelse(is.null(args[["min-prop"]]), "0.0", args[["min-prop"]]))

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  set.seed(seed)
  genes <- read_list(genes_path)
  cell_types <- read_list(cell_types_path)
  spatial <- read_expression(spatial_expression, spot_id_col, genes)
  scrna <- read_expression(scrna_expression, cell_id_col, genes)
  labels <- read.csv(scrna_labels, check.names = FALSE)
  reference <- subset_reference(scrna, labels, cell_id_col, cell_type_col, cell_types, max_cells_per_type, seed)
  markers <- build_markers(reference$expr, reference$labels, genes, cell_type_col, top_markers)
  write.csv(markers, file.path(output_dir, "spotlight_marker_genes.csv"), row.names = FALSE)

  x <- t(reference$expr)
  y <- t(spatial)
  groups <- as.character(reference$labels[[cell_type_col]])
  colnames(x) <- as.character(reference$labels[[cell_id_col]])
  rownames(x) <- genes
  colnames(y) <- rownames(spatial)
  rownames(y) <- genes

  result <- SPOTlight(
    x = x,
    y = y,
    groups = groups,
    mgs = markers,
    n_top = top_markers,
    gene_id = "gene",
    group_id = "cluster",
    weight_id = "weight",
    model = "ns",
    min_prop = min_prop,
    verbose = TRUE,
    nrun = nrun,
    maxIter = max_iter
  )
  weights <- normalize_and_align(result$mat, rownames(spatial), cell_types)
  write.csv(weights, file.path(output_dir, "spotlight_proportions.csv"), quote = TRUE)
  write.csv(as.data.frame(result$res_ss), file.path(output_dir, "spotlight_residual_ss.csv"), quote = TRUE)

  counts <- table(factor(reference$labels[[cell_type_col]], levels = cell_types))
  manifest <- list(
    method = "SPOTlight",
    package = "SPOTlight",
    package_version = as.character(utils::packageVersion("SPOTlight")),
    spatial_expression_path = spatial_expression,
    scrna_expression_path = scrna_expression,
    scrna_labels_path = scrna_labels,
    genes_path = genes_path,
    cell_types_path = cell_types_path,
    output_dir = output_dir,
    num_spots = nrow(spatial),
    num_genes = length(genes),
    num_cells_used = nrow(reference$expr),
    cell_type_counts = as.list(as.integer(counts)),
    cell_types = cell_types,
    top_markers = top_markers,
    max_cells_per_type = ifelse(max_cells_per_type > 0, max_cells_per_type, NA),
    nrun = nrun,
    max_iter = max_iter,
    min_prop = min_prop,
    seed = seed,
    proportions_path = file.path(output_dir, "spotlight_proportions.csv"),
    markers_path = file.path(output_dir, "spotlight_marker_genes.csv")
  )
  names(manifest$cell_type_counts) <- cell_types
  jsonlite::write_json(manifest, file.path(output_dir, "spotlight_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
}

main()
