#= ------------------------------------------------------------------------

    Pyramidenbasierte Suche

    Two entry points:

    `pyramid_search`         — translation-only multi-resolution search
                                across NCC, NCCR, GDP, GDPR and GDPR-local.
                                Builds source/template/mask pyramids,
                                searches exhaustively on the coarsest level
                                and refines candidates downwards with
                                level-adjusted thresholds and a small
                                spatial padding around each candidate.

    `shape_search`           — shape-based search that iterates over all
                                (θ, scale) variants of a
                                `RotatedScaledSearchModel`, evaluating the
                                shape score at each candidate position.
                                Pyramid-based (θ, s)-granularity refinement
                                is deferred to a later step; this version
                                searches at a single resolution.

    Both functions return a `Vector{Match}` after non-maximum suppression
    in (row, col) space.

------------------------------------------------------------------------ =#

export pyramid_search, shape_search


# ------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------

# Dispatch a similarity-style metric to its match function.
function _translation_match(metric::Symbol,
                             src::AbstractMatrix, tmpl::AbstractMatrix,
                             mask, gradient_threshold::Real)
    if metric === :ncc
        return ncc_match(src, tmpl; mask = mask)
    elseif metric === :nccr
        return nccr_match(src, tmpl; mask = mask)
    elseif metric === :gdp
        return gdp_match(src, tmpl; mask = mask,
                         gradient_threshold = gradient_threshold)
    elseif metric === :gdpr
        return gdpr_match(src, tmpl; mask = mask,
                          gradient_threshold = gradient_threshold)
    elseif metric === :gdpr_local
        return gdpr_local_match(src, tmpl; mask = mask,
                                 gradient_threshold = gradient_threshold)
    end
    throw(ArgumentError("unsupported metric: $metric " *
                        "(use :ncc, :nccr, :gdp, :gdpr, :gdpr_local)"))
end

# Greedy non-maximum suppression on a list of (r, c, score, ...) candidates.
# Within a Chebyshev radius `suppression_radius` of any higher-scoring
# already-selected candidate, lower-scoring ones are dropped.
function _nms_translation(candidates, suppression_radius::Int)
    isempty(candidates) && return candidates
    sorted = sort(candidates, by = x -> -x.score)
    selected = similar(candidates, 0)
    for cand in sorted
        suppressed = false
        for sel in selected
            if abs(sel.r - cand.r) <= suppression_radius &&
               abs(sel.c - cand.c) <= suppression_radius
                suppressed = true
                break
            end
        end
        suppressed || push!(selected, cand)
    end
    return selected
end

function _candidates_above_threshold(score_image::AbstractMatrix{Float64},
                                      threshold::Real)
    h, w = size(score_image)
    candidates = NamedTuple{(:r, :c, :score), Tuple{Int,Int,Float64}}[]
    @inbounds for j in 1:w, i in 1:h
        if score_image[i, j] >= threshold
            push!(candidates,
                  (r = i, c = j, score = Float64(score_image[i, j])))
        end
    end
    return candidates
end


# ------------------------------------------------------------------------
# Translation pyramid search
# ------------------------------------------------------------------------

"""
    pyramid_search(source::AbstractMatrix, template::AbstractMatrix;
                   metric::Symbol = :ncc,
                   mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                   min_score::Real = 0.5,
                   min_score_adjust::Real = 0.9,
                   min_size::Int = 8,
                   stop_at_level::Int = 1,
                   padding::Int = 2,
                   gradient_threshold::Real = 0.0)
        -> Vector{Match{Tuple{Int,Int}}}

Multi-resolution translation search. Builds 2×2 mean-filter pyramids of
both source and template (and the mask if provided), runs an exhaustive
search at the coarsest level, then refines candidates downwards. At each
finer level only a `padding`-sized window around each candidate is
re-scored.

Per-level threshold: `min_score · min_score_adjust^(level − 1)` — coarser
levels have a lower threshold (PDF-Folie 450).

`metric` is one of `:ncc`, `:nccr`, `:gdp`, `:gdpr`, `:gdpr_local`. SAD is
not supported here because its score is not bounded in `[0, 1]`.

Each returned match's `pose` is `(row, col)` of the template's top-left
corner in the original source.
"""
function pyramid_search(source::AbstractMatrix, template::AbstractMatrix;
                         metric::Symbol = :ncc,
                         mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                         min_score::Real = 0.5,
                         min_score_adjust::Real = 0.9,
                         min_size::Int = 8,
                         stop_at_level::Int = 1,
                         padding::Int = 2,
                         gradient_threshold::Real = 0.0)
    h_src, w_src = size(source)
    h_tmpl, w_tmpl = size(template)
    (h_src >= h_tmpl && w_src >= w_tmpl) ||
        throw(ArgumentError("source ($h_src×$w_src) must be ≥ template ($h_tmpl×$w_tmpl)"))
    min_score_adjust > 0 ||
        throw(ArgumentError("min_score_adjust must be positive"))
    stop_at_level >= 1 ||
        throw(ArgumentError("stop_at_level must be ≥ 1"))
    padding >= 0 ||
        throw(ArgumentError("padding must be ≥ 0"))
    min_size >= 2 ||
        throw(ArgumentError("min_size must be ≥ 2"))

    # Template pyramid — drives the level count
    tmpl_levels = Matrix{Float64}[Float64.(template)]
    while min(size(tmpl_levels[end])...) ÷ 2 >= min_size
        push!(tmpl_levels, _mean_downsample(tmpl_levels[end]))
    end
    n_levels = length(tmpl_levels)
    stop_at_level <= n_levels ||
        throw(ArgumentError("stop_at_level ($stop_at_level) > pyramid height ($n_levels)"))

    # Source pyramid — same height
    src_levels = Matrix{Float64}[Float64.(source)]
    for _ in 2:n_levels
        push!(src_levels, _mean_downsample(src_levels[end]))
    end

    # Optional mask pyramid
    msk_levels = if mask === nothing
        nothing
    else
        size(mask) == (h_tmpl, w_tmpl) ||
            throw(ArgumentError("mask size must equal template size"))
        levels = BitMatrix[BitMatrix(mask)]
        for _ in 2:n_levels
            push!(levels, _erode_downsample(levels[end]))
        end
        levels
    end

    threshold(k) = min_score * min_score_adjust^(k - 1)

    # Coarsest level: exhaustive search
    msk_top = msk_levels === nothing ? nothing : msk_levels[n_levels]
    score_top = _translation_match(metric,
                                    src_levels[n_levels], tmpl_levels[n_levels],
                                    msk_top, gradient_threshold)
    sup_top = max(1, min(size(tmpl_levels[n_levels])...) ÷ 2)
    candidates = _nms_translation(
        _candidates_above_threshold(score_top, threshold(n_levels)),
        sup_top,
    )

    # Refine through finer levels
    for k in (n_levels - 1):-1:stop_at_level
        h_src_k, w_src_k = size(src_levels[k])
        h_tmpl_k, w_tmpl_k = size(tmpl_levels[k])
        n_pos_r = h_src_k - h_tmpl_k + 1
        n_pos_c = w_src_k - w_tmpl_k + 1
        msk_k = msk_levels === nothing ? nothing : msk_levels[k]
        level_threshold = threshold(k)

        new_candidates = NamedTuple{(:r, :c, :score), Tuple{Int,Int,Float64}}[]
        for cand in candidates
            r_center = 2 * cand.r - 1
            c_center = 2 * cand.c - 1

            r_min = max(1, r_center - padding)
            r_max = min(n_pos_r, r_center + padding + 1)
            c_min = max(1, c_center - padding)
            c_max = min(n_pos_c, c_center + padding + 1)
            (r_min <= r_max && c_min <= c_max) || continue

            r_end = min(h_src_k, r_max + h_tmpl_k - 1)
            c_end = min(w_src_k, c_max + w_tmpl_k - 1)

            sub_src = view(src_levels[k], r_min:r_end, c_min:c_end)
            sub_score = _translation_match(metric,
                                            sub_src, tmpl_levels[k],
                                            msk_k, gradient_threshold)

            @inbounds for j in 1:size(sub_score, 2), i in 1:size(sub_score, 1)
                if sub_score[i, j] >= level_threshold
                    push!(new_candidates,
                          (r = r_min + i - 1, c = c_min + j - 1,
                           score = Float64(sub_score[i, j])))
                end
            end
        end

        sup_radius = max(1, min(h_tmpl_k, w_tmpl_k) ÷ 2)
        candidates = _nms_translation(new_candidates, sup_radius)
    end

    return [Match(c.score, (c.r, c.c)) for c in candidates]
end


# ------------------------------------------------------------------------
# Shape-based search (single-resolution; pyramid acceleration on the
# (θ, s) granularity is deferred to a later step)
# ------------------------------------------------------------------------

# Score the entire source for one shape variant. Returns a Vector of
# (r, c, score) tuples whose score >= min_score.
function _shape_score_image(score_fn, src_gx, src_gy, model::ShapeModel,
                             h_src::Int, w_src::Int,
                             min_score::Float64,
                             min_source_gradient::Float64)
    candidates = NamedTuple{(:r, :c, :score), Tuple{Int,Int,Float64}}[]
    @inbounds for c in 1:w_src, r in 1:h_src
        score = score_fn(src_gx, src_gy, model, [Float64(r), Float64(c)];
                         min_source_gradient = min_source_gradient)
        if score >= min_score
            push!(candidates, (r = r, c = c, score = score))
        end
    end
    return candidates
end


"""
    shape_search(source::AbstractMatrix,
                 search_model::RotatedScaledSearchModel;
                 variant::Symbol = :gdp,
                 min_score::Real = 0.5,
                 min_source_gradient::Real = 0.0,
                 suppression_radius::Int = -1)
        -> Vector{Match{NamedTuple{(:r, :c, :θ, :scale), …}}}

Iterate over every `(θ, scale)` variant in `search_model`, evaluate the
shape-based score at every position in `source`, and keep candidates
with `score ≥ min_score`. After collecting candidates from all variants,
non-maximum suppression in `(row, col)` keeps only the strongest match
within a Chebyshev radius of `suppression_radius`. If
`suppression_radius < 0` it defaults to half of the typical model
extent (estimated from the base model's bounding box).

`variant` selects the metric: `:gdp`, `:gdpr` (global contrast reversal)
or `:gdpr_local` (local contrast reversal).

Each returned match's `pose` is a `NamedTuple` with the four fields
`(:r, :c, :θ, :scale)`.

Note: the (θ, scale) granularity is **not** halved per pyramid level in
this version — every variant in `search_model` is evaluated at full
source resolution. Multi-resolution refinement of the (θ, scale) grid is
left to a future step.
"""
function shape_search(source::AbstractMatrix,
                      search_model::RotatedScaledSearchModel;
                      variant::Symbol = :gdp,
                      min_score::Real = 0.5,
                      min_source_gradient::Real = 0.0,
                      suppression_radius::Int = -1)
    score_fn = if variant === :gdp
        shape_score_gdp
    elseif variant === :gdpr
        shape_score_gdpr
    elseif variant === :gdpr_local
        shape_score_gdpr_local
    else
        throw(ArgumentError("unsupported variant: $variant " *
                            "(use :gdp, :gdpr, :gdpr_local)"))
    end

    src_gx, src_gy = image_gradients(source)
    h_src, w_src = size(source)
    min_score_f = Float64(min_score)
    min_grad_f = Float64(min_source_gradient)

    # Default suppression radius: half the model extent in (row, col)
    if suppression_radius < 0
        max_extent = 0.0
        for p in search_model.base_model.points
            ext = max(abs(p[1]), abs(p[2]))
            ext > max_extent && (max_extent = ext)
        end
        suppression_radius = max(1, round(Int, max_extent))
    end

    all_candidates = NamedTuple{(:r, :c, :θ, :scale, :score),
                                Tuple{Int,Int,Float64,Float64,Float64}}[]
    for v in search_model
        per_variant = _shape_score_image(score_fn, src_gx, src_gy, v.model,
                                          h_src, w_src,
                                          min_score_f, min_grad_f)
        for cand in per_variant
            push!(all_candidates,
                  (r = cand.r, c = cand.c, θ = v.θ, scale = v.scale,
                   score = cand.score))
        end
    end

    # NMS in (r, c) regardless of (θ, scale)
    isempty(all_candidates) && return Match[]
    sorted = sort(all_candidates, by = x -> -x.score)
    selected = similar(sorted, 0)
    for cand in sorted
        suppressed = false
        for sel in selected
            if abs(sel.r - cand.r) <= suppression_radius &&
               abs(sel.c - cand.c) <= suppression_radius
                suppressed = true
                break
            end
        end
        suppressed || push!(selected, cand)
    end

    return [Match(c.score, (r = c.r, c = c.c, θ = c.θ, scale = c.scale))
            for c in selected]
end
