#= ------------------------------------------------------------------------

    Templatebasiertes Matching

    Three similarity metrics, each with single-model, multi-model and
    pixel-wise consolidated variants:

    - SAD  (Sum of Absolute Differences)        — lower is better
    - NCC  (Normalized Cross Correlation)       — range [-1, 1]
    - NCCR (NCC with contrast reversal — |NCC|) — range [ 0,  1]

    SAD supports early termination via `max_score` (PDF-Folie 445):
    positions whose partial sum exceeds `max_score · n` are aborted and
    reported as `Inf`.

------------------------------------------------------------------------ =#

export sad_match, ncc_match, nccr_match
export sad_match_pixelwise, ncc_match_pixelwise, nccr_match_pixelwise


# ------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------

function _ensure_mask(mask, h_mdl, w_mdl)
    if mask === nothing
        return trues(h_mdl, w_mdl)
    end
    size(mask) == (h_mdl, w_mdl) ||
        throw(ArgumentError("mask size $(size(mask)) does not match model size ($h_mdl, $w_mdl)"))
    return BitMatrix(mask)
end

function _check_dimensions(source, model)
    h_src, w_src = size(source)
    h_mdl, w_mdl = size(model)
    h_out = h_src - h_mdl + 1
    w_out = w_src - w_mdl + 1
    (h_out >= 1 && w_out >= 1) ||
        throw(ArgumentError("source ($h_src×$w_src) must be ≥ model ($h_mdl×$w_mdl) in both dimensions"))
    return (h_out, w_out, h_mdl, w_mdl)
end

# Common multi-model setup: same size required, optional masks aligned.
function _check_multi_models(source, models, masks)
    n_models = length(models)
    n_models > 0 || throw(ArgumentError("models must be non-empty"))
    h_mdl, w_mdl = size(models[1])
    for k in 2:n_models
        size(models[k]) == (h_mdl, w_mdl) ||
            throw(ArgumentError("all models must have the same size; models[$k] differs"))
    end
    h_src, w_src = size(source)
    h_out = h_src - h_mdl + 1
    w_out = w_src - w_mdl + 1
    (h_out >= 1 && w_out >= 1) ||
        throw(ArgumentError("source ($h_src×$w_src) must be ≥ model ($h_mdl×$w_mdl) in both dimensions"))
    if masks !== nothing
        length(masks) == n_models ||
            throw(ArgumentError("masks must have the same length as models"))
    end
    return (n_models, h_out, w_out, h_mdl, w_mdl)
end

# Model count, mean and Σ(t-m)² over the masked region — all with one pass each.
function _model_moments(model::AbstractMatrix{Float64}, mask::AbstractMatrix{Bool})
    n = 0
    s = 0.0
    h, w = size(model)
    @inbounds for j in 1:w, i in 1:h
        if mask[i, j]
            n += 1
            s += model[i, j]
        end
    end
    n > 0 || throw(ArgumentError("mask must contain at least one true element"))
    m = s / n
    s2 = 0.0
    @inbounds for j in 1:w, i in 1:h
        if mask[i, j]
            d = model[i, j] - m
            s2 += d * d
        end
    end
    return (n, m, s2)
end


# ------------------------------------------------------------------------
# SAD
# ------------------------------------------------------------------------

# Per-position SAD with early abort. Returns Inf if aborted, else the
# unnormalized sum (caller divides by n).
@inline function _sad_sum_at(src::Matrix{Float64}, mdl::Matrix{Float64},
                              msk::BitMatrix, r::Int, c::Int,
                              h_mdl::Int, w_mdl::Int, threshold_sum::Float64)
    s = 0.0
    @inbounds for cm in 1:w_mdl
        for rm in 1:h_mdl
            if msk[rm, cm]
                s += abs(mdl[rm, cm] - src[r + rm - 1, c + cm - 1])
                if s > threshold_sum
                    return Inf
                end
            end
        end
    end
    return s
end


"""
    sad_match(source::AbstractMatrix, model::AbstractMatrix;
              mask::Union{Nothing,AbstractMatrix{Bool}}=nothing,
              max_score::Real=Inf) -> Matrix{Float64}

Sum-of-Absolute-Differences score image. Lower is better; a perfect match
has score `0`. Output size is `size(source) .- size(model) .+ 1`.

If `mask` is given (same size as `model`), only `true` pixels contribute.

If `max_score < Inf`, positions whose partial absolute-difference sum exceeds
`max_score · n` are aborted early and reported as `Inf` (PDF-Folie 445).

```jldoctest
julia> using Patterns

julia> source = zeros(10, 10); template = [1.0 2.0; 3.0 4.0];

julia> source[4:5, 6:7] .= template;

julia> scores = sad_match(source, template);

julia> scores[4, 6] ≈ 0.0
true
```
"""
function sad_match(source::AbstractMatrix, model::AbstractMatrix;
                   mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                   max_score::Real = Inf)
    h_out, w_out, h_mdl, w_mdl = _check_dimensions(source, model)
    src = Float64.(source)
    mdl = Float64.(model)
    msk = _ensure_mask(mask, h_mdl, w_mdl)
    n = count(msk)
    n > 0 || throw(ArgumentError("mask must contain at least one true element"))

    threshold_sum = isfinite(max_score) ? Float64(max_score) * n : Inf
    scores = fill(Inf, h_out, w_out)

    @inbounds for c in 1:w_out, r in 1:h_out
        s = _sad_sum_at(src, mdl, msk, r, c, h_mdl, w_mdl, threshold_sum)
        if isfinite(s)
            scores[r, c] = s / n
        end
    end
    return scores
end


"""
    sad_match(source::AbstractMatrix,
              models::AbstractVector{<:AbstractMatrix};
              masks=nothing, max_score::Real=Inf) -> Vector{Matrix{Float64}}

Multi-model SAD: returns one score image per model. All models must share
the same size. `masks`, if given, must be a vector of the same length as
`models`; each entry may be `nothing` or an `AbstractMatrix{Bool}` of the
model's size.
"""
function sad_match(source::AbstractMatrix,
                   models::AbstractVector{<:AbstractMatrix};
                   masks = nothing, max_score::Real = Inf)
    n_models, h_out, w_out, h_mdl, w_mdl = _check_multi_models(source, models, masks)
    src = Float64.(source)
    score_imgs = Vector{Matrix{Float64}}(undef, n_models)

    for k in 1:n_models
        mdl = Float64.(models[k])
        msk = _ensure_mask(masks === nothing ? nothing : masks[k], h_mdl, w_mdl)
        n = count(msk)
        n > 0 || throw(ArgumentError("mask $k must contain at least one true element"))
        threshold_sum = isfinite(max_score) ? Float64(max_score) * n : Inf

        scores = fill(Inf, h_out, w_out)
        @inbounds for c in 1:w_out, r in 1:h_out
            s = _sad_sum_at(src, mdl, msk, r, c, h_mdl, w_mdl, threshold_sum)
            if isfinite(s)
                scores[r, c] = s / n
            end
        end
        score_imgs[k] = scores
    end
    return score_imgs
end


"""
    sad_match_pixelwise(source, models;
                        masks=nothing, max_score::Real=Inf)
        -> (min_scores::Matrix{Float64}, best_index::Matrix{Int})

Pixel-wise consolidated SAD across multiple models: returns the minimum
SAD score per pixel and the 1-based index of the model that achieved it.
Index `0` means no model produced a finite score (all aborted).
"""
function sad_match_pixelwise(source::AbstractMatrix,
                              models::AbstractVector{<:AbstractMatrix};
                              masks = nothing, max_score::Real = Inf)
    n_models, h_out, w_out, h_mdl, w_mdl = _check_multi_models(source, models, masks)
    src = Float64.(source)
    min_scores = fill(Inf, h_out, w_out)
    best_idx = zeros(Int, h_out, w_out)

    for k in 1:n_models
        mdl = Float64.(models[k])
        msk = _ensure_mask(masks === nothing ? nothing : masks[k], h_mdl, w_mdl)
        n = count(msk)
        n > 0 || throw(ArgumentError("mask $k must contain at least one true element"))
        threshold_sum = isfinite(max_score) ? Float64(max_score) * n : Inf

        @inbounds for c in 1:w_out, r in 1:h_out
            s = _sad_sum_at(src, mdl, msk, r, c, h_mdl, w_mdl, threshold_sum)
            if isfinite(s)
                normalized = s / n
                if normalized < min_scores[r, c]
                    min_scores[r, c] = normalized
                    best_idx[r, c] = k
                end
            end
        end
    end
    return (min_scores, best_idx)
end


# ------------------------------------------------------------------------
# NCC / NCCR
# ------------------------------------------------------------------------

# Per-position NCC. Single pass through the masked region; uses the
# Σf, Σf², Σtf identities to derive cor and σ_f.
@inline function _ncc_at(src::Matrix{Float64}, mdl::Matrix{Float64},
                          msk::BitMatrix, r::Int, c::Int,
                          h_mdl::Int, w_mdl::Int,
                          n::Int, m_t::Float64, s2_t::Float64)
    sf = 0.0
    sf2 = 0.0
    s_tf = 0.0
    @inbounds for cm in 1:w_mdl, rm in 1:h_mdl
        if msk[rm, cm]
            f = src[r + rm - 1, c + cm - 1]
            t = mdl[rm, cm]
            sf  += f
            sf2 += f * f
            s_tf += t * f
        end
    end
    m_f = sf / n
    s2_f = sf2 - n * m_f * m_f
    cor = s_tf - n * m_t * m_f
    denom = sqrt(s2_t * s2_f)
    if !(denom > 1e-12)
        return 0.0
    end
    return clamp(cor / denom, -1.0, 1.0)
end


"""
    ncc_match(source::AbstractMatrix, model::AbstractMatrix;
              mask::Union{Nothing,AbstractMatrix{Bool}}=nothing) -> Matrix{Float64}

Normalized Cross-Correlation score image, range `[-1, 1]`. A perfect match
has score `1`; a perfect inverse match has `-1`. Returns `0` where either
the model or the source region under the mask has zero variance.

```jldoctest
julia> using Patterns

julia> source = fill(2.0, 10, 10); template = [1.0 2.0; 3.0 4.0];

julia> source[4:5, 6:7] .= 3.0 .* template .+ 1.0;  # contrast change preserves NCC

julia> scores = ncc_match(source, template);

julia> isapprox(scores[4, 6], 1.0; atol=1e-10)
true
```
"""
function ncc_match(source::AbstractMatrix, model::AbstractMatrix;
                   mask::Union{Nothing,AbstractMatrix{Bool}} = nothing)
    h_out, w_out, h_mdl, w_mdl = _check_dimensions(source, model)
    src = Float64.(source)
    mdl = Float64.(model)
    msk = _ensure_mask(mask, h_mdl, w_mdl)
    n, m_t, s2_t = _model_moments(mdl, msk)

    scores = zeros(Float64, h_out, w_out)
    @inbounds for c in 1:w_out, r in 1:h_out
        scores[r, c] = _ncc_at(src, mdl, msk, r, c, h_mdl, w_mdl, n, m_t, s2_t)
    end
    return scores
end


"""
    ncc_match(source::AbstractMatrix,
              models::AbstractVector{<:AbstractMatrix};
              masks=nothing) -> Vector{Matrix{Float64}}

Multi-model NCC: returns one score image per model. All models must share
the same size.
"""
function ncc_match(source::AbstractMatrix,
                   models::AbstractVector{<:AbstractMatrix};
                   masks = nothing)
    n_models, h_out, w_out, h_mdl, w_mdl = _check_multi_models(source, models, masks)
    src = Float64.(source)
    score_imgs = Vector{Matrix{Float64}}(undef, n_models)

    for k in 1:n_models
        mdl = Float64.(models[k])
        msk = _ensure_mask(masks === nothing ? nothing : masks[k], h_mdl, w_mdl)
        n, m_t, s2_t = _model_moments(mdl, msk)

        scores = zeros(Float64, h_out, w_out)
        @inbounds for c in 1:w_out, r in 1:h_out
            scores[r, c] = _ncc_at(src, mdl, msk, r, c, h_mdl, w_mdl, n, m_t, s2_t)
        end
        score_imgs[k] = scores
    end
    return score_imgs
end


"""
    ncc_match_pixelwise(source, models; masks=nothing)
        -> (max_scores::Matrix{Float64}, best_index::Matrix{Int})

Pixel-wise consolidated NCC across multiple models: returns the maximum
NCC score per pixel and the 1-based index of the model that achieved it.
"""
function ncc_match_pixelwise(source::AbstractMatrix,
                              models::AbstractVector{<:AbstractMatrix};
                              masks = nothing)
    n_models, h_out, w_out, h_mdl, w_mdl = _check_multi_models(source, models, masks)
    src = Float64.(source)
    max_scores = fill(-Inf, h_out, w_out)
    best_idx = zeros(Int, h_out, w_out)

    for k in 1:n_models
        mdl = Float64.(models[k])
        msk = _ensure_mask(masks === nothing ? nothing : masks[k], h_mdl, w_mdl)
        n, m_t, s2_t = _model_moments(mdl, msk)

        @inbounds for c in 1:w_out, r in 1:h_out
            score = _ncc_at(src, mdl, msk, r, c, h_mdl, w_mdl, n, m_t, s2_t)
            if score > max_scores[r, c]
                max_scores[r, c] = score
                best_idx[r, c] = k
            end
        end
    end
    return (max_scores, best_idx)
end


"""
    nccr_match(source::AbstractMatrix, model::AbstractMatrix;
               mask::Union{Nothing,AbstractMatrix{Bool}}=nothing) -> Matrix{Float64}

NCC with contrast reversal: returns `|NCC|`, range `[0, 1]`. Both the
direct match and the contrast-inverted match yield score `1`.
"""
nccr_match(source::AbstractMatrix, model::AbstractMatrix;
           mask::Union{Nothing,AbstractMatrix{Bool}} = nothing) =
    abs.(ncc_match(source, model; mask = mask))


"""
    nccr_match(source, models; masks=nothing) -> Vector{Matrix{Float64}}

Multi-model NCCR.
"""
function nccr_match(source::AbstractMatrix,
                    models::AbstractVector{<:AbstractMatrix};
                    masks = nothing)
    return [abs.(s) for s in ncc_match(source, models; masks = masks)]
end


"""
    nccr_match_pixelwise(source, models; masks=nothing)
        -> (max_scores::Matrix{Float64}, best_index::Matrix{Int})

Pixel-wise consolidated NCCR.
"""
function nccr_match_pixelwise(source::AbstractMatrix,
                               models::AbstractVector{<:AbstractMatrix};
                               masks = nothing)
    n_models, h_out, w_out, h_mdl, w_mdl = _check_multi_models(source, models, masks)
    src = Float64.(source)
    max_scores = fill(-Inf, h_out, w_out)
    best_idx = zeros(Int, h_out, w_out)

    for k in 1:n_models
        mdl = Float64.(models[k])
        msk = _ensure_mask(masks === nothing ? nothing : masks[k], h_mdl, w_mdl)
        n, m_t, s2_t = _model_moments(mdl, msk)

        @inbounds for c in 1:w_out, r in 1:h_out
            score = abs(_ncc_at(src, mdl, msk, r, c, h_mdl, w_mdl, n, m_t, s2_t))
            if score > max_scores[r, c]
                max_scores[r, c] = score
                best_idx[r, c] = k
            end
        end
    end
    return (max_scores, best_idx)
end
