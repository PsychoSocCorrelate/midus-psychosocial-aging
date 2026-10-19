#!/usr/bin/env Rscript
# Measurement-invariance CFA for 9 latent composites, plus sleepAvg
#
# Tests configural and metric invariance across 4 cohorts
# (MIDUS2, MIDUSR, MIDUS3, MR2).
# Extracts factor scores from metric model via lavPredict(method="EBM").
#
# Run from project root:  Rscript preprocess/composites/cfa_composites.R
# Input:  data/processed/wave1/all_M2MR_py.csv
#         data/processed/wave2/all_M3MR2_py.csv
# Output: data/processed/shared/composite_cfa_scores.csv
#         data/processed/shared/composite_cfa_fit.csv
#         data/processed/shared/composite_align_fit.csv

library(lavaan)
library(dplyr)
library(purrr)

cat("CFA composite construction\n\n")

# 1. Load and stack

w1 <- read.csv("data/processed/wave1/all_M2MR_py.csv")
w2 <- read.csv("data/processed/wave2/all_M3MR2_py.csv")

w1$group <- as.character(w1$cohort)       # "MIDUS2" / "MIDUSR"
w2$group <- as.character(w2$cohort)      # "MIDUS3" / "MR2"
w1$wave  <- "W1"
w2$wave  <- "W2"

# Harmonise sampleId type (W1 numeric SPSS codes, W2 character)
w1$sampleId <- as.character(w1$sampleId)
w2$sampleId <- as.character(w2$sampleId)

pooled <- bind_rows(w1, w2)
pooled$.row_id <- seq_len(nrow(pooled))

cat("Pooled: n =", nrow(pooled),
    "(W1:", nrow(w1), " W2:", nrow(w2), ")\n")
cat("Groups:", paste(sort(unique(pooled$group)), collapse = ", "), "\n\n")

# 2. Reverse-code negatively keyed items

pooled <- pooled %>%
  mutate(
    percConstraints_r = -percConstraints,
    spouseStrain_r    = -spouseStrain,
    spouseDisagree_r  = -spouseDisagree,
    maritalRisk_r     = -maritalRisk,
    famStrain_r       = -famStrain,
    friendStrain_r    = -friendStrain
  )

# 3. Helpers

get_fit <- function(fit) {
  all_fm <- fitMeasures(fit)
  # Prefer robust versions if available, fall back to standard
  cfi   <- if ("cfi.robust" %in% names(all_fm)) all_fm["cfi.robust"] else all_fm["cfi"]
  rmsea <- if ("rmsea.robust" %in% names(all_fm)) all_fm["rmsea.robust"] else all_fm["rmsea"]
  srmr  <- all_fm["srmr"]
  chisq <- if ("chisq.scaled" %in% names(all_fm)) all_fm["chisq.scaled"] else all_fm["chisq"]
  df    <- all_fm["df"]
  pval  <- if ("pvalue.scaled" %in% names(all_fm)) all_fm["pvalue.scaled"] else all_fm["pvalue"]
  data.frame(cfi = unname(cfi), rmsea = unname(rmsea), srmr = unname(srmr),
             chisq = unname(chisq), df = unname(df), pvalue = unname(pval))
}

extract_scores <- function(fit, data) {
  # Returns vector of z-scored factor scores aligned with data rows
  scores_raw <- lavPredict(fit, type = "lv", method = "EBM")
  group_labels <- lavInspect(fit, "group.label")

  out <- rep(NA_real_, nrow(data))
  for (g in seq_along(group_labels)) {
    g_rows <- which(data$group == group_labels[g])
    sv <- scores_raw[[g]][, 1]
    if (length(g_rows) == length(sv)) {
      out[g_rows] <- sv
    } else {
      warning(sprintf("Row mismatch group %s: data=%d scores=%d",
                      group_labels[g], length(g_rows), length(sv)))
    }
  }
  as.numeric(scale(out))
}

run_mi <- function(model_str, data, name) {
  # Returns list(fit_config, fit_metric, fit_scalar, delta_cfi, scores)
  cat(sprintf("\n%s\n", name))

  fit_config <- tryCatch(
    cfa(model_str, data = data, group = "group",
        missing = "fiml", estimator = "MLR"),
    error = function(e) { cat("  ERROR configural:", e$message, "\n"); NULL }
  )
  if (is.null(fit_config) || !lavInspect(fit_config, "converged")) {
    cat("  Configural did not converge.\n"); return(NULL)
  }

  fit_metric <- tryCatch(
    cfa(model_str, data = data, group = "group",
        group.equal = "loadings", missing = "fiml", estimator = "MLR"),
    error = function(e) { cat("  ERROR metric:", e$message, "\n"); NULL }
  )
  if (is.null(fit_metric) || !lavInspect(fit_metric, "converged")) {
    cat("  Metric did not converge.\n"); return(NULL)
  }

  fc <- get_fit(fit_config)
  fm <- get_fit(fit_metric)
  delta_cfi <- fm$cfi - fc$cfi
  cat(sprintf("  Configural: CFI=%.4f  RMSEA=%.4f  SRMR=%.4f\n",
              fc$cfi, fc$rmsea, fc$srmr))
  cat(sprintf("  Metric:     CFI=%.4f  RMSEA=%.4f  SRMR=%.4f\n",
              fm$cfi, fm$rmsea, fm$srmr))
  cat(sprintf("  dCFI = %.4f %s\n", delta_cfi,
              ifelse(abs(delta_cfi) <= 0.01, "metric OK", "metric fails")))

  # Scalar (informational only)
  fit_scalar <- tryCatch(
    cfa(model_str, data = data, group = "group",
        group.equal = c("loadings", "intercepts"),
        missing = "fiml", estimator = "MLR"),
    error = function(e) NULL
  )
  if (!is.null(fit_scalar) && lavInspect(fit_scalar, "converged")) {
    fs <- get_fit(fit_scalar)
    dcs <- fs$cfi - fm$cfi
    cat(sprintf("  Scalar:     CFI=%.4f  RMSEA=%.4f  dCFI=%.4f %s\n",
                fs$cfi, fs$rmsea, dcs,
                ifelse(abs(dcs) <= 0.01, "ok", "fails")))
  }

  # Extract scores from metric model
  scores <- NULL
  if (abs(delta_cfi) <= 0.01) {
    scores <- extract_scores(fit_metric, data)
    cat(sprintf("  Scores extracted: %d non-NA\n", sum(!is.na(scores))))
  } else {
    cat("  Metric invariance fails; using configural scores\n")
    # Fall back to configural scores
    scores <- extract_scores(fit_config, data)
    cat(sprintf("  Fallback configural scores: %d non-NA (not measurement-invariant)\n",
                sum(!is.na(scores))))
  }

  # Collect fit table rows
  fits <- list(
    cbind(data.frame(composite = name, level = "configural"), fc),
    cbind(data.frame(composite = name, level = "metric"), fm)
  )
  if (!is.null(fit_scalar) && lavInspect(fit_scalar, "converged"))
    fits[[3]] <- cbind(data.frame(composite = name, level = "scalar"),
                       get_fit(fit_scalar))

  list(scores = scores, fit_table = bind_rows(fits), delta_cfi = delta_cfi)
}

# 4. Define models

models <- list(

  # 1. Psychological wellbeing: 1f vs 2f
  psychWB_1f = list(
    name = "psychWellbeing_1f",
    model = '
      pwb =~ autonomy + envMastery + persGrowth + posRelation +
             lifePurpose + selfAccept + persMastery + percConstraints_r + selfEsteem
      selfAccept ~~ selfEsteem
      envMastery ~~ percConstraints_r
    ',
    score_name = "psychWellbeing"
  ),

  psychWB_2f = list(
    name = "psychWellbeing_2f",
    model = '
      mastery =~ envMastery + selfAccept + selfEsteem + persMastery + percConstraints_r
      growth  =~ autonomy + persGrowth + posRelation + lifePurpose
      selfAccept ~~ selfEsteem
    ',
    score_names = c("psychWB_mastery", "psychWB_growth")
  ),

  # 2-4. Spouse, family, friend quality: alignment (section 5c)
  # CFA not feasible: structural missingness (~30% non-partnered),
  # near-singular covariance (r up to 0.92), 3-item just-identified.

  # 5. Give Emotional Support
  giveSupport = list(
    name = "giveSupport",
    model = '
      giveS =~ givePartnerEmoSupp + giveParentEmoSupp + giveInLawEmoSupp +
               giveChildEmoSupp + giveFriendEmoSupp
    ',
    score_name = "giveSupport"
  ),

  # 6. Receive emotional support: alignment (section 5c)
  # MI failed: dCFI=-0.015; loadings vary 0.31-0.77 across groups.

  # 7. Religiosity (5 core items)
  religiosity = list(
    name = "religiosity",
    model = '
      relig =~ spirituality + religiousId + attendReligious +
               religiousCopingA + religiousCopingB
      religiousCopingA ~~ religiousCopingB
    ',
    score_name = "religiosity"
  ),

  # 8. Active Coping
  activeCoping = list(
    name = "activeCoping",
    model = '
      actCope =~ copingGrowth + copingActive + planning +
                 primaryControl + posReappraisal
      copingActive ~~ planning
    ',
    score_name = "activeCoping"
  ),

  # 9. Life satisfaction: alignment (section 5c)
  # CFA: poor fit (RMSEA>0.30); rateCurrentFin loads differently.
  dummy_end = NULL
)

# 5. Run MI tests

all_fit <- list()
score_cols <- list()

# 5a. Psychological wellbeing: 1f vs 2f
cat("Psychological well-being: 1 vs 2 factors\n")
res_1f <- run_mi(models$psychWB_1f$model, pooled, "psychWellbeing_1f")
res_2f <- run_mi(models$psychWB_2f$model, pooled, "psychWellbeing_2f")

if (!is.null(res_1f) && !is.null(res_2f)) {
  cfi_1f <- res_1f$fit_table %>% filter(level == "configural") %>% pull(cfi)
  cfi_2f <- res_2f$fit_table %>% filter(level == "configural") %>% pull(cfi)
  cat(sprintf("\n  1-factor configural CFI = %.4f\n", cfi_1f))
  cat(sprintf("  2-factor configural CFI = %.4f\n", cfi_2f))

  if (cfi_1f >= 0.90 || (cfi_1f - cfi_2f) > -0.02) {
    cat("  Using the 1-factor model (adequate fit, or 2 factors improve it only marginally)\n\n")
    score_cols[["psychWellbeing"]] <- res_1f$scores
    all_fit <- c(all_fit, list(res_1f$fit_table))
  } else {
    cat("  Using the 2-factor model (substantially better fit)\n\n")
    # Extract 2 scores from 2f metric model
    fit_2f_metric <- cfa(models$psychWB_2f$model, data = pooled,
                         group = "group", group.equal = "loadings",
                         missing = "fiml", estimator = "MLR")
    scores_2f <- lavPredict(fit_2f_metric, type = "lv", method = "EBM")
    group_labels <- lavInspect(fit_2f_metric, "group.label")

    mastery_vec <- growth_vec <- rep(NA_real_, nrow(pooled))
    for (g in seq_along(group_labels)) {
      g_rows <- which(pooled$group == group_labels[g])
      mastery_vec[g_rows] <- scores_2f[[g]][, 1]
      growth_vec[g_rows]  <- scores_2f[[g]][, 2]
    }
    score_cols[["psychWB_mastery"]] <- as.numeric(scale(mastery_vec))
    score_cols[["psychWB_growth"]]  <- as.numeric(scale(growth_vec))
    all_fit <- c(all_fit, list(res_2f$fit_table))
  }
} else if (!is.null(res_1f)) {
  score_cols[["psychWellbeing"]] <- res_1f$scores
  all_fit <- c(all_fit, list(res_1f$fit_table))
}

# 5b. Remaining CFA composites
cfa_composites <- c("giveSupport", "religiosity", "activeCoping")

for (comp_name in cfa_composites) {
  comp <- models[[comp_name]]
  res <- run_mi(comp$model, pooled, comp$name)
  if (!is.null(res)) {
    score_cols[[comp$score_name]] <- res$scores
    all_fit <- c(all_fit, list(res$fit_table))
  }
  cat("\n")
}

# 5c. Alignment method for the remaining composites
# Standard multi-group CFA failed for these composites (convergence,
# near-singularity, or poor fit). The alignment method (Asparouhov &
# Muthén, 2014) estimates group-specific parameters freely (configural),
# then optimally aligns them to identify the minimal set of non-invariant
# parameters. R2(lambda) and R2(nu) quantify how well invariance holds;
# >= 0.95 indicates adequate approximate invariance.
#
# Fallback: mean-of-z if alignment also fails (e.g., convergence issues
# from structural missingness).

library(sirt)

cat("Alignment-method composites\n\n")

align_defs <- list(
  recSupport      = c("recPartnerEmoSupp", "recParentEmoSupp", "recInLawEmoSupp",
                       "recChildEmoSupp", "recFriendEmoSupp"),
  spouseQuality   = c("spouseSupport", "spouseStrain_r", "spouseSolidarity",
                       "spouseDisagree_r", "rateCurrentRelation", "maritalRisk_r"),
  familyQuality   = c("famSupport", "famStrain_r", "famSolidarity", "supportForFam"),
  friendQuality   = c("friendSupport", "friendStrain_r", "friendSolidarity"),
  lifeSatisfaction = c("lifeSat", "lifeOverall", "rateToday", "rateCurrentFin")
)

# Helper: compute aligned factor scores from configural CFA parameters
# Uses the aligned lambda/nu to score each person:
#   score_i = mean( (x_ij - nu_j^aligned) / lambda_j^aligned )
# Then z-standardise across entire pooled sample.
compute_aligned_scores <- function(data, items, group_col,
                                   lambda_aligned, nu_aligned) {
  group_labels <- rownames(lambda_aligned)
  out <- rep(NA_real_, nrow(data))
  for (g in group_labels) {
    g_rows <- which(data[[group_col]] == g)
    if (length(g_rows) == 0) next
    lam_g <- lambda_aligned[g, ]
    nu_g  <- nu_aligned[g, ]
    # Standardised residual for each item, then mean across items
    item_mat <- as.matrix(data[g_rows, items])
    std_mat  <- sweep(item_mat, 2, nu_g, "-")
    std_mat  <- sweep(std_mat, 2, lam_g, "/")
    n_valid  <- rowSums(!is.na(std_mat))
    scores   <- ifelse(n_valid >= ceiling(length(items) / 2),
                       rowMeans(std_mat, na.rm = TRUE), NA_real_)
    out[g_rows] <- scores
  }
  as.numeric(scale(out))
}

align_fit_rows <- list()

for (comp_name in names(align_defs)) {
  items <- align_defs[[comp_name]]
  cat(sprintf("\n%s (%d items)\n", comp_name, length(items)))

  # Prepare sub-data (complete cases per group for configural estimation)
  sub <- pooled[, c(items, "group"), drop = FALSE]

  # Step 1: group-specific configural CFA
  pars <- tryCatch(
    invariance_alignment_cfa_config(sub[, items, drop = FALSE],
                                     group = sub$group, verbose = FALSE),
    error = function(e) { cat("  CONFIG ERROR:", e$message, "\n"); NULL }
  )

  if (is.null(pars)) {
    # Fallback to mean-of-z
    cat("  Falling back to mean of z-scores\n")
    z_mat <- sapply(items, function(v) as.numeric(scale(pooled[[v]])))
    n_valid <- rowSums(!is.na(z_mat))
    score_cols[[comp_name]] <- ifelse(n_valid >= ceiling(length(items) / 2),
                                     rowMeans(z_mat, na.rm = TRUE), NA_real_)
    score_cols[[comp_name]] <- as.numeric(scale(score_cols[[comp_name]]))
    cat(sprintf("  non-NA=%d (%.1f%%)\n",
                sum(!is.na(score_cols[[comp_name]])),
                100 * sum(!is.na(score_cols[[comp_name]])) / nrow(pooled)))
    align_fit_rows[[comp_name]] <- data.frame(
      composite = comp_name, method = "mean-of-z",
      R2_lambda = NA_real_, R2_nu = NA_real_,
      n_noninvariant_lambda = NA_integer_, n_noninvariant_nu = NA_integer_
    )
    cat("\n")
    next
  }

  # Step 2: alignment
  mod <- tryCatch(
    invariance.alignment(lambda = pars$lambda, nu = pars$nu, eps = 0.01),
    error = function(e) { cat("  ALIGN ERROR:", e$message, "\n"); NULL }
  )

  if (is.null(mod)) {
    cat("  Falling back to mean of z-scores\n")
    z_mat <- sapply(items, function(v) as.numeric(scale(pooled[[v]])))
    n_valid <- rowSums(!is.na(z_mat))
    score_cols[[comp_name]] <- ifelse(n_valid >= ceiling(length(items) / 2),
                                     rowMeans(z_mat, na.rm = TRUE), NA_real_)
    score_cols[[comp_name]] <- as.numeric(scale(score_cols[[comp_name]]))
    cat(sprintf("  non-NA=%d (%.1f%%)\n",
                sum(!is.na(score_cols[[comp_name]])),
                100 * sum(!is.na(score_cols[[comp_name]])) / nrow(pooled)))
    align_fit_rows[[comp_name]] <- data.frame(
      composite = comp_name, method = "mean-of-z",
      R2_lambda = NA_real_, R2_nu = NA_real_,
      n_noninvariant_lambda = NA_integer_, n_noninvariant_nu = NA_integer_
    )
    cat("\n")
    next
  }

  # R2 and non-invariant counts
  r2 <- mod$es.invariance
  r2_lam <- r2["R2", "loadings"]
  r2_nu  <- r2["R2", "intercepts"]
  cat(sprintf("  R2(lambda) = %.4f   R2(nu) = %.4f\n", r2_lam, r2_nu))
  cat(sprintf("  Approximate MI: %s\n",
              ifelse(r2_lam >= 0.95 & r2_nu >= 0.95, "ADEQUATE",
                     ifelse(r2_lam >= 0.90 & r2_nu >= 0.90, "MARGINAL", "POOR"))))

  # Aligned pars
  lam_al <- mod$lambda.aligned
  nu_al  <- mod$nu.aligned
  colnames(lam_al) <- items
  colnames(nu_al)  <- items
  cat("  Aligned loadings:\n")
  print(round(lam_al, 3))
  cat("  Aligned intercepts:\n")
  print(round(nu_al, 3))

  # Compute scores using aligned parameters
  score_cols[[comp_name]] <- compute_aligned_scores(
    pooled, items, "group", lam_al, nu_al
  )
  cat(sprintf("  Scores: non-NA=%d (%.1f%%)\n",
              sum(!is.na(score_cols[[comp_name]])),
              100 * sum(!is.na(score_cols[[comp_name]])) / nrow(pooled)))

  align_fit_rows[[comp_name]] <- data.frame(
    composite = comp_name, method = "alignment",
    R2_lambda = r2_lam, R2_nu = r2_nu,
    n_noninvariant_lambda = sum(r2_lam < 0.95),
    n_noninvariant_nu = sum(r2_nu < 0.95)
  )
  cat("\n")
}

# 5d. Simple computed composites
cat("\nSimple composites\n\n")
score_cols[["sleepAvg"]] <- (5 * pooled$sleepWorkday + 2 * pooled$sleepWknd) / 7
cat(sprintf("  sleepAvg             non-NA=%d (%.1f%%)\n",
            sum(!is.na(score_cols[["sleepAvg"]])),
            100 * sum(!is.na(score_cols[["sleepAvg"]])) / nrow(pooled)))

# 6. Build output

out <- pooled %>% select(M2MRID, wave, group)
for (nm in names(score_cols)) {
  out[[nm]] <- score_cols[[nm]]
}

# Existing scores are kept unless replacement is requested. Re-running under
# renv.lock changes four composites by at most 5e-06.
#
#   Rscript preprocess/composites/cfa_composites.R --force-overwrite
#   MIDUS_CFA_FORCE=1 Rscript preprocess/composites/cfa_composites.R
dir.create("data/processed/shared", showWarnings = FALSE, recursive = TRUE)
.scores_path <- "data/processed/shared/composite_cfa_scores.csv"
.force <- ("--force-overwrite" %in% commandArgs(trailingOnly = TRUE)) ||
          nzchar(Sys.getenv("MIDUS_CFA_FORCE"))
if (file.exists(.scores_path) && !.force) {
  cat("\nComposite scores already exist and were not overwritten.\n")
  cat("  Pass --force-overwrite to replace them.\n")
  cat("\nDone (no files written).\n")
  quit(save = "no", status = 0)
}

write.csv(out, .scores_path, row.names = FALSE)
cat(sprintf("\nScores saved: data/processed/shared/composite_cfa_scores.csv\n"))
cat(sprintf("  Rows: %d | Composite columns: %d\n",
            nrow(out), length(score_cols)))
for (nm in names(score_cols)) {
  cat(sprintf("  %-25s  non-NA: %d (%.1f%%)\n",
              nm, sum(!is.na(score_cols[[nm]])),
              100 * sum(!is.na(score_cols[[nm]])) / nrow(out)))
}

# Save fit table (CFA MI)
fit_table <- bind_rows(all_fit)
write.csv(fit_table, "data/processed/shared/composite_cfa_fit.csv", row.names = FALSE)
cat(sprintf("\nCFA fit table saved: data/processed/shared/composite_cfa_fit.csv\n"))

# Save alignment fit table
align_fit_table <- bind_rows(align_fit_rows)
write.csv(align_fit_table, "data/processed/shared/composite_align_fit.csv", row.names = FALSE)
cat("Alignment fit table saved: data/processed/shared/composite_align_fit.csv\n")
print(align_fit_table)

cat("\nDone.\n")
