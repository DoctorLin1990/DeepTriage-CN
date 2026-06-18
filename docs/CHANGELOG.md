# Changelog

## v1.1.0 — Revision (2025)

### Bug fixes
- **B4 (preprocess, critical)**: `_apply_degradation()` in robustness simulation
  called `StructuredEncoder.fit_transform()` on validation data, causing data leakage.
  Fixed to use `encoder.transform()` exclusively on held-out data.
- **B5 (preprocess)**: Sparse narrative identification used `.str.len()` (character
  count) instead of `.str.split().str.len()` (word count). Fixed to match paper
  definition of ≤3 words (Section 4.7).

### New analyses (reviewer response)
- Added bootstrapped age × model interaction contrasts (Section 4.4;
  DeepTriage-CN vs NEWS2: *p* = 0.001; vs TabNet: *p* = 0.13).
- Added Table 3: calibration metrics stratified by age subgroup
  (Brier score, calibration slope/intercept, Hosmer–Lemeshow test).
- Added Table 4: systematic alert burden analysis at four classification
  thresholds (0.15, 0.28, 0.40, 0.50).
- Added continuous NRI 95% confidence intervals.

### Clarifications
- Disclosed PPV = 0.50 at Youden-optimal threshold (0.28) in Abstract.
- Added caution note on MNAR simulation limitations in Section 4.5.
- Added discussion of Chinese medical-domain pretrained models (Section 2.3).

## v1.0.0 — Initial submission (2024)
