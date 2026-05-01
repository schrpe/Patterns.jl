# Patterns

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://schrpe.github.io/Patterns.jl/dev)

Patterns.jl provides pattern matching algorithms for machine vision tasks
in pure Julia.

The package bundles three complementary classes of matching algorithms,
each with single-, per-model multi-, and pixel-wise consolidated variants
where applicable, plus pyramid-accelerated and keypoint-based pipelines:

- **Template matching** — SAD, NCC, and NCC-with-contrast-reversal.
- **Gradient matching** — cosine similarity of Sobel gradients, with
  optional global or local contrast reversal.
- **Shape-based matching** — gradient matching with explicit 2×2
  transformation, enabling rotation- and scale-invariant search.
- **Pyramid search** — multi-resolution acceleration via 2×2 mean-filter
  pyramids.
- **Keypoint matching** — FAST/ORB detection and Hamming-distance matching
  with Lowe's ratio test, plus a similarity-transform RANSAC for pose
  estimation (single best and multi-instance).
- **Subpixel localisation** — 2D quadratic peak fit and Levenberg-Marquardt
  pose refinement using edge tangents.
