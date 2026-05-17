/**
 * Static Phase 7.3 validation snapshot — the headline numbers from the
 * manuscript's §3 (Results). Hand-transcribed from
 *   docs/research/phase7_3_validation.md  (canonical source)
 *   /root/repos/exports/2026-05-15_phase7_3_*.json  (export artefacts)
 *
 * This module is the single source of truth for the ValidationLab section.
 * Numbers are frozen — they represent the snapshot deposited at the
 * v0.7.3-paper-submission tag, not live experiment output.
 */

export const HUFNER = {
  // Constants enter the Hüfner equation; none is tuned.
  hb_g_per_dL: 14.5,
  hufner_mL_per_g: 1.34,
  henry_mL_per_dL_per_mmHg: 0.003,
  paco2_mmHg: 40,
  rq: 0.8,
  altitude_m: 0,
} as const;

export const TRAIN_MAE = {
  cgem_s: 0.000426,
  pulse_s: 0.000001,
} as const;

export const SOBOL = {
  // First-order Sobol indices over the full unfiltered design.
  // Source: corrected_time_sobol.json, headline_s1_*
  ranked: [
    { name: "gz_peak", S1: 0.6076, ST: 0.6189, label: "Peak +Gz" },
    { name: "fio2_inspired", S1: 0.2491, ST: 0.2499, label: "FiO₂ (Hüfner)" },
    { name: "gz_onset_rate", S1: 0.1316, ST: 0.1419, label: "Gz onset rate" },
    { name: "rest", S1: 0.0117, ST: 0.0, label: "Residual (8 axes)" },
  ],
  s2_gz_fio2: 0.0026,
  s2_gz_fio2_conf: 0.0305,
  stability: 0.8139, // bootstrap-CI convergence diagnostic
} as const;

export const ABLATION = {
  coupled_S1_fio2: 0.2491,
  ablated_S1_fio2: -2.8953e-10,
  ablated_S1_fio2_conf: 5.299e-9,
  delta: 0.2491,
} as const;

export const CONFORMAL = {
  alpha: 0.1,
  target: 0.9,
  marginal_coverage: 0.9067,
  q_hat_s: 0.0044,
  interval_width_s: 0.0087,
  n_cal: 461,
  n_test: 461,
  mc_propagation: 0.9805,
  strata: [
    { name: "normo", label: "Normoxia",         n: 431, coverage: 0.9026 },
    { name: "hypo3000", label: "Hypobaric ≈ 3 000 m", n: 28,  coverage: 0.9643 },
    { name: "hypo4000", label: "Hypobaric ≈ 4 000 m", n: 2,   coverage: 1.0000 },
  ],
} as const;

export const WHINNERY = {
  // The 8 in-scope rapid-onset bins with non-zero Phase 7.1b design coverage.
  // Source: whinnery_bins_anchor.json, per_bin where predicted_corrected_time_s is not null.
  closed_form_mae_s: 0.58,
  closed_form_n: 47,
  bin_median_abs_z: 0.224,
  bin_fraction_in_envelope: 1.0,
  ror_spearman: 0.119,
  gor_spearman_out_of_scope: -0.8,
  bins: [
    { table: 1, label: "8 ≤ Gz < 9",  n_paper:  97, paper_mean: 9.00,  predicted: 8.9148, z: -0.0396 },
    { table: 1, label: "7 ≤ Gz < 8",  n_paper:  70, paper_mean: 9.64,  predicted: 8.9724, z: -0.2029 },
    { table: 1, label: "6 ≤ Gz < 7",  n_paper:  49, paper_mean: 10.61, predicted: 9.4847, z: -0.2452 },
    { table: 1, label: "5 ≤ Gz < 6",  n_paper:   8, paper_mean: 11.50, predicted: 9.9587, z: -0.3661 },
    { table: 2, label: "≥ 4 G/s",     n_paper:   8, paper_mean: 9.13,  predicted: 9.1587, z:  0.0091 },
    { table: 2, label: "3 ≤ ṙ < 4",   n_paper: 120, paper_mean: 8.83,  predicted: 9.5363, z:  0.2803 },
    { table: 2, label: "2 ≤ ṙ < 3",   n_paper: 129, paper_mean: 8.68,  predicted: 9.5899, z:  0.4174 },
    { table: 2, label: "1 ≤ ṙ < 2",   n_paper: 134, paper_mean: 9.75,  predicted: 9.4798, z: -0.1150 },
  ],
  omitted: { label: "9+ Gz", n_paper: 164, reason: "zero design coverage" },
  corpus_total: 888,
  corpus_ror_table1: 406,
  corpus_ror_table2: 432,
  corpus_gor_table3: 293,
} as const;

export const DIRECTIONAL = {
  median_low_fio2_s: 8.9044,
  median_high_fio2_s: 10.0326,
  n_low: 2304,
  n_high: 161916,
  threshold_low: 0.16,
  threshold_high: 0.30,
  fio2_range: [0.15, 1.0],
  mann_whitney_p: 1e-6, // reported as < 10⁻⁶
} as const;

export const PHASE_TAG = {
  release: "v0.7.3-paper-submission",
  date: "2026-05-16",
  commit_short: "62deb6a",
  parquet_rows: 6144,
  cgem_rows: 3072,
  pulse_rows: 3072,
  n_base_paired: 128,
  n_base_refit: 8192,
} as const;
