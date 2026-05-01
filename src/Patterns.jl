"""
    Patterns

A Julia package for pattern matching in machine vision tasks. The
package provides three complementary classes of matching algorithms,
each with single-, per-model multi-, and pixel-wise consolidated
variants where applicable, plus pyramid-accelerated and keypoint-based
pipelines.

# Templatebasiertes Matching ([`sad_match`](@ref), [`ncc_match`](@ref), [`nccr_match`](@ref))

Pixel-wise comparison of source and template using one of three
similarity metrics:

  - [`sad_match`](@ref)  — Sum of Absolute Differences (lower = better, supports early abort)
  - [`ncc_match`](@ref)  — Normalized Cross-Correlation (range `[-1, 1]`)
  - [`nccr_match`](@ref) — NCC with global contrast reversal (range `[0, 1]`)

Pixel-wise multi-template variants:
  - [`sad_match_pixelwise`](@ref), [`ncc_match_pixelwise`](@ref),
    [`nccr_match_pixelwise`](@ref)

# Gradientenbasiertes Matching ([`gdp_match`](@ref), [`gdpr_match`](@ref), [`gdpr_local_match`](@ref))

Cosine similarity of Sobel gradients (range `[0, 1]`):

  - [`gdp_match`](@ref)        — no contrast reversal
  - [`gdpr_match`](@ref)       — global contrast reversal
  - [`gdpr_local_match`](@ref) — local contrast reversal

Pixel-wise multi-model variants are also exported.

# Shape-Based Matching

Gradient matching with an explicit 2×2 transformation `A` and
translation `t`, enabling rotation- and scale-invariant search:

  - [`build_shape_model`](@ref) — extract a `(point, gradient_vector)` list from a reference image
  - [`transform_shape_model`](@ref) — pre-rotate/scale a model
  - [`shape_score_gdp`](@ref), [`shape_score_gdpr`](@ref), [`shape_score_gdpr_local`](@ref)
  - [`create_similarity_template`](@ref) — build a `RotatedScaledSearchModel`
    with pre-computed `(θ, scale)` variants
  - [`nearest_variant`](@ref)

# Pyramidensuche

Multi-resolution acceleration via 2×2 mean-filter pyramids:

  - [`ImagePyramid`](@ref), [`MaskPyramid`](@ref)
  - [`pyramid_search`](@ref) — translation-only search with `:ncc`, `:nccr`,
    `:gdp`, `:gdpr`, `:gdpr_local`
  - [`shape_search`](@ref) — single-resolution shape-based search across all
    (θ, scale) variants of a `RotatedScaledSearchModel`

# Mehrmodell-Pipelines

Search multiple distinct models in one source:

  - [`search_templates`](@ref) — per-model pyramid search
  - [`search_templates_pixelwise`](@ref) — pixel-wise consolidated, all metrics
  - [`search_shape_models`](@ref) — per-model shape-based search

# Keypoint-Matching

Wrappers around `ImageFeatures` plus a similarity-transform RANSAC:

  - [`PatternKeypoint`](@ref) — position + binary descriptor
  - [`detect_fast`](@ref), [`detect_fast_scale_invariant`](@ref), [`detect_orb`](@ref)
  - [`match_keypoints`](@ref) — Hamming-distance matching with Lowe's ratio test
  - [`match_all_keypoints`](@ref) — unfiltered matching
  - [`compute_similarity_pose`](@ref) — closed-form similarity from 2 match pairs
  - [`estimate_pose`](@ref) — RANSAC for the single best similarity pose
  - [`estimate_all_poses`](@ref) — multi-instance RANSAC, with multi-model dispatch

# Subpixel-Lokalisierung

  - [`subpixel_peak`](@ref) — 2D quadratic LSQ fit on a 3×3 score peak
  - [`refine_pose_similarity`](@ref) — Levenberg-Marquardt pose refinement
    using edge tangents

# Core data types

  - [`Pose2D`](@ref), [`Match`](@ref), [`ShapeModel`](@ref),
    [`KeypointMatch`](@ref), [`TransformResult`](@ref)
  - [`ShapeVariant`](@ref), [`RotatedScaledSearchModel`](@ref) `<: AbstractSearchModel`
  - [`ImagePyramid`](@ref), [`MaskPyramid`](@ref)
  - [`apply_pose`](@ref), [`identity_pose`](@ref), [`similarity_pose`](@ref),
    [`image_gradients`](@ref)
"""
module Patterns

using LinearAlgebra

include("types.jl")
include("subpixel.jl")
include("template_matching.jl")
include("gradient_matching.jl")
include("shape_based.jl")
include("similarity_search.jl")
include("pyramid.jl")
include("pyramid_search.jl")
include("multi_model.jl")
include("keypoint.jl")
include("keypoint_match.jl")
include("ransac.jl")

end # module
