#= ------------------------------------------------------------------------

    Gradientenbasiertes Matching

    Three metrics, each computed as the average normalized dot product of
    Sobel gradients between source and model:

    - GDP          (Gradient Dot Products)              — max(0, Σ rᵢ / n)
    - GDPR (global)  (|Σ rᵢ| / n)                       — globale Kontrastumkehr
    - GDPR (lokal)   (Σ |rᵢ| / n)                       — lokale Kontrastumkehr

    where each rᵢ = (gᵢ_model · gᵢ_source) / (|gᵢ_model| · |gᵢ_source|)
    is the cosine of the angle between the model and source gradient vectors
    at model point i.

    Model points with `|gradient| ≤ gradient_threshold` are excluded.
    Source points with zero gradient contribute 0 to the sum.

    GDP and GDPR-local support early termination via `min_score` (PDF-Folie 464).
    GDPR-global has no useful early-abort criterion.

------------------------------------------------------------------------ =#

export gdp_match, gdpr_match, gdpr_local_match
export gdp_match_pixelwise, gdpr_match_pixelwise, gdpr_local_match_pixelwise


# ------------------------------------------------------------------------
# Sobel gradient
# ------------------------------------------------------------------------

"""
    _sobel(img) -> (gx, gy)

3×3 Sobel gradient. Boundary pixels (first/last row and column) are 0.

Sobel kernels:
    Sx = [-1 0 1; -2 0 2; -1 0 1]   (x-derivative, columns)
    Sy = [-1 -2 -1; 0 0 0; 1 2 1]   (y-derivative, rows)
"""
function _sobel(img::AbstractMatrix{Float64})
    h, w = size(img)
    gx = zeros(h, w)
    gy = zeros(h, w)
    @inbounds for j in 2:w-1, i in 2:h-1
        gx[i, j] = (img[i-1, j+1] + 2 * img[i, j+1] + img[i+1, j+1]) -
                   (img[i-1, j-1] + 2 * img[i, j-1] + img[i+1, j-1])
        gy[i, j] = (img[i+1, j-1] + 2 * img[i+1, j] + img[i+1, j+1]) -
                   (img[i-1, j-1] + 2 * img[i-1, j] + img[i-1, j+1])
    end
    return (gx, gy)
end


# ------------------------------------------------------------------------
# Model gradient points
# ------------------------------------------------------------------------

# Pre-compute the model gradient vectors at all masked positions whose
# gradient magnitude exceeds the threshold. Result is a NamedTuple of
# parallel vectors for cache-friendly iteration.
function _model_gradient_points(mdl::Matrix{Float64}, msk::BitMatrix,
                                 gradient_threshold::Float64)
    h, w = size(mdl)
    gx, gy = _sobel(mdl)
    is = Int[]
    js = Int[]
    gxs = Float64[]
    gys = Float64[]
    mags = Float64[]
    @inbounds for j in 1:w, i in 1:h
        if msk[i, j]
            mag = sqrt(gx[i, j]^2 + gy[i, j]^2)
            if mag > gradient_threshold
                push!(is, i)
                push!(js, j)
                push!(gxs, gx[i, j])
                push!(gys, gy[i, j])
                push!(mags, mag)
            end
        end
    end
    return (i = is, j = js, gx = gxs, gy = gys, mag = mags)
end


# Common setup for any single-model gradient match:
# returns source gradients, model points, and output dimensions.
function _gradient_setup(source::AbstractMatrix, model::AbstractMatrix,
                          mask, gradient_threshold::Real)
    h_out, w_out, h_mdl, w_mdl = _check_dimensions(source, model)
    src = Float64.(source)
    mdl = Float64.(model)
    msk = _ensure_mask(mask, h_mdl, w_mdl)
    src_gx, src_gy = _sobel(src)
    mdl_pts = _model_gradient_points(mdl, msk, Float64(gradient_threshold))
    return (src_gx, src_gy, mdl_pts, h_out, w_out)
end


# ------------------------------------------------------------------------
# Inner score loops
# ------------------------------------------------------------------------

# GDP / GDPR-global accumulator: signed sum of cosines.
@inline function _gdp_sum_at(src_gx::Matrix{Float64}, src_gy::Matrix{Float64},
                              mdl_pts, r0::Int, c0::Int, n::Int,
                              abort_threshold::Float64)
    s = 0.0
    @inbounds for k in 1:n
        i = mdl_pts.i[k]
        j = mdl_pts.j[k]
        sgx = src_gx[r0 + i - 1, c0 + j - 1]
        sgy = src_gy[r0 + i - 1, c0 + j - 1]
        smag = sqrt(sgx * sgx + sgy * sgy)
        mmag = mdl_pts.mag[k]
        denom = smag * mmag
        if denom > 0
            s += (sgx * mdl_pts.gx[k] + sgy * mdl_pts.gy[k]) / denom
        end
        # Early abort: if even maximum future contributions cannot reach
        # the running threshold (k + abort_threshold) we can stop.
        if isfinite(abort_threshold) && s < k + abort_threshold
            return -Inf
        end
    end
    return s
end

# GDPR-local accumulator: sum of absolute cosines (each in [0, 1]).
@inline function _gdpr_local_sum_at(src_gx::Matrix{Float64}, src_gy::Matrix{Float64},
                                     mdl_pts, r0::Int, c0::Int, n::Int,
                                     abort_threshold::Float64)
    s = 0.0
    @inbounds for k in 1:n
        i = mdl_pts.i[k]
        j = mdl_pts.j[k]
        sgx = src_gx[r0 + i - 1, c0 + j - 1]
        sgy = src_gy[r0 + i - 1, c0 + j - 1]
        smag = sqrt(sgx * sgx + sgy * sgy)
        mmag = mdl_pts.mag[k]
        denom = smag * mmag
        if denom > 0
            s += abs((sgx * mdl_pts.gx[k] + sgy * mdl_pts.gy[k]) / denom)
        end
        if isfinite(abort_threshold) && s < k + abort_threshold
            return -Inf
        end
    end
    return s
end


# ------------------------------------------------------------------------
# GDP
# ------------------------------------------------------------------------

"""
    gdp_match(source::AbstractMatrix, model::AbstractMatrix;
              mask=nothing, gradient_threshold::Real=0.0,
              min_score::Real=0.0) -> Matrix{Float64}

Gradient Dot Products score image, range `[0, 1]`. A perfect match has
score `1`. Negative cosine sums are clamped to `0` (no contrast reversal —
gradients pointing in opposite directions do not contribute).

Model points with `|gradient| ≤ gradient_threshold` are excluded.

If `min_score > 0`, the inner loop aborts early at positions where the
partial sum can no longer reach `min_score · n` (PDF-Folie 464). Aborted
positions are reported as `0`.
"""
function gdp_match(source::AbstractMatrix, model::AbstractMatrix;
                   mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                   gradient_threshold::Real = 0.0,
                   min_score::Real = 0.0)
    src_gx, src_gy, mdl_pts, h_out, w_out =
        _gradient_setup(source, model, mask, gradient_threshold)
    n = length(mdl_pts.i)
    scores = zeros(h_out, w_out)
    n == 0 && return scores
    abort_threshold = min_score > 0 ? -n * (1 - Float64(min_score)) : -Inf

    @inbounds for c in 1:w_out, r in 1:h_out
        s = _gdp_sum_at(src_gx, src_gy, mdl_pts, r, c, n, abort_threshold)
        scores[r, c] = isfinite(s) ? max(0.0, s / n) : 0.0
    end
    return scores
end

function gdp_match(source::AbstractMatrix,
                   models::AbstractVector{<:AbstractMatrix};
                   masks = nothing,
                   gradient_threshold::Real = 0.0,
                   min_score::Real = 0.0)
    n_models, h_out, w_out, h_mdl, w_mdl = _check_multi_models(source, models, masks)
    src_gx, src_gy = _sobel(Float64.(source))
    score_imgs = Vector{Matrix{Float64}}(undef, n_models)

    for k in 1:n_models
        mdl = Float64.(models[k])
        msk = _ensure_mask(masks === nothing ? nothing : masks[k], h_mdl, w_mdl)
        mdl_pts = _model_gradient_points(mdl, msk, Float64(gradient_threshold))
        n = length(mdl_pts.i)
        scores = zeros(h_out, w_out)
        if n > 0
            abort_threshold = min_score > 0 ? -n * (1 - Float64(min_score)) : -Inf
            @inbounds for c in 1:w_out, r in 1:h_out
                s = _gdp_sum_at(src_gx, src_gy, mdl_pts, r, c, n, abort_threshold)
                scores[r, c] = isfinite(s) ? max(0.0, s / n) : 0.0
            end
        end
        score_imgs[k] = scores
    end
    return score_imgs
end

"""
    gdp_match_pixelwise(source, models; ...) -> (max_scores, best_index)

Pixel-wise consolidated GDP across multiple models.
"""
function gdp_match_pixelwise(source::AbstractMatrix,
                              models::AbstractVector{<:AbstractMatrix};
                              masks = nothing,
                              gradient_threshold::Real = 0.0,
                              min_score::Real = 0.0)
    n_models, h_out, w_out, h_mdl, w_mdl = _check_multi_models(source, models, masks)
    src_gx, src_gy = _sobel(Float64.(source))
    max_scores = fill(-Inf, h_out, w_out)
    best_idx = zeros(Int, h_out, w_out)

    for k in 1:n_models
        mdl = Float64.(models[k])
        msk = _ensure_mask(masks === nothing ? nothing : masks[k], h_mdl, w_mdl)
        mdl_pts = _model_gradient_points(mdl, msk, Float64(gradient_threshold))
        n = length(mdl_pts.i)
        n == 0 && continue
        abort_threshold = min_score > 0 ? -n * (1 - Float64(min_score)) : -Inf

        @inbounds for c in 1:w_out, r in 1:h_out
            s = _gdp_sum_at(src_gx, src_gy, mdl_pts, r, c, n, abort_threshold)
            score = isfinite(s) ? max(0.0, s / n) : 0.0
            if score > max_scores[r, c]
                max_scores[r, c] = score
                best_idx[r, c] = k
            end
        end
    end
    # Replace any remaining -Inf (no model contributed) with 0.0
    @inbounds for i in eachindex(max_scores)
        if max_scores[i] == -Inf
            max_scores[i] = 0.0
        end
    end
    return (max_scores, best_idx)
end


# ------------------------------------------------------------------------
# GDPR (global contrast reversal)
# ------------------------------------------------------------------------

"""
    gdpr_match(source::AbstractMatrix, model::AbstractMatrix;
               mask=nothing, gradient_threshold::Real=0.0) -> Matrix{Float64}

GDP with global contrast reversal: `|Σ rᵢ| / n`, range `[0, 1]`. Both
matching and globally inverted-contrast matching yield score `1`.
No early-abort criterion is available for this metric.
"""
function gdpr_match(source::AbstractMatrix, model::AbstractMatrix;
                    mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                    gradient_threshold::Real = 0.0)
    src_gx, src_gy, mdl_pts, h_out, w_out =
        _gradient_setup(source, model, mask, gradient_threshold)
    n = length(mdl_pts.i)
    scores = zeros(h_out, w_out)
    n == 0 && return scores

    @inbounds for c in 1:w_out, r in 1:h_out
        s = _gdp_sum_at(src_gx, src_gy, mdl_pts, r, c, n, -Inf)
        scores[r, c] = abs(s) / n
    end
    return scores
end

function gdpr_match(source::AbstractMatrix,
                    models::AbstractVector{<:AbstractMatrix};
                    masks = nothing,
                    gradient_threshold::Real = 0.0)
    n_models, h_out, w_out, h_mdl, w_mdl = _check_multi_models(source, models, masks)
    src_gx, src_gy = _sobel(Float64.(source))
    score_imgs = Vector{Matrix{Float64}}(undef, n_models)

    for k in 1:n_models
        mdl = Float64.(models[k])
        msk = _ensure_mask(masks === nothing ? nothing : masks[k], h_mdl, w_mdl)
        mdl_pts = _model_gradient_points(mdl, msk, Float64(gradient_threshold))
        n = length(mdl_pts.i)
        scores = zeros(h_out, w_out)
        if n > 0
            @inbounds for c in 1:w_out, r in 1:h_out
                s = _gdp_sum_at(src_gx, src_gy, mdl_pts, r, c, n, -Inf)
                scores[r, c] = abs(s) / n
            end
        end
        score_imgs[k] = scores
    end
    return score_imgs
end

"""
    gdpr_match_pixelwise(source, models; ...) -> (max_scores, best_index)

Pixel-wise consolidated GDPR (global) across multiple models.
"""
function gdpr_match_pixelwise(source::AbstractMatrix,
                               models::AbstractVector{<:AbstractMatrix};
                               masks = nothing,
                               gradient_threshold::Real = 0.0)
    n_models, h_out, w_out, h_mdl, w_mdl = _check_multi_models(source, models, masks)
    src_gx, src_gy = _sobel(Float64.(source))
    max_scores = fill(-Inf, h_out, w_out)
    best_idx = zeros(Int, h_out, w_out)

    for k in 1:n_models
        mdl = Float64.(models[k])
        msk = _ensure_mask(masks === nothing ? nothing : masks[k], h_mdl, w_mdl)
        mdl_pts = _model_gradient_points(mdl, msk, Float64(gradient_threshold))
        n = length(mdl_pts.i)
        n == 0 && continue
        @inbounds for c in 1:w_out, r in 1:h_out
            s = _gdp_sum_at(src_gx, src_gy, mdl_pts, r, c, n, -Inf)
            score = abs(s) / n
            if score > max_scores[r, c]
                max_scores[r, c] = score
                best_idx[r, c] = k
            end
        end
    end
    @inbounds for i in eachindex(max_scores)
        if max_scores[i] == -Inf
            max_scores[i] = 0.0
        end
    end
    return (max_scores, best_idx)
end


# ------------------------------------------------------------------------
# GDPR (local contrast reversal)
# ------------------------------------------------------------------------

"""
    gdpr_local_match(source::AbstractMatrix, model::AbstractMatrix;
                     mask=nothing, gradient_threshold::Real=0.0,
                     min_score::Real=0.0) -> Matrix{Float64}

GDP with local contrast reversal: `Σ |rᵢ| / n`, range `[0, 1]`. Each
gradient pair is allowed to be inverted independently — the score
remains `1` even with locally inverted contrast (PDF-Folie 462).

Supports early termination via `min_score`.
"""
function gdpr_local_match(source::AbstractMatrix, model::AbstractMatrix;
                          mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                          gradient_threshold::Real = 0.0,
                          min_score::Real = 0.0)
    src_gx, src_gy, mdl_pts, h_out, w_out =
        _gradient_setup(source, model, mask, gradient_threshold)
    n = length(mdl_pts.i)
    scores = zeros(h_out, w_out)
    n == 0 && return scores
    abort_threshold = min_score > 0 ? -n * (1 - Float64(min_score)) : -Inf

    @inbounds for c in 1:w_out, r in 1:h_out
        s = _gdpr_local_sum_at(src_gx, src_gy, mdl_pts, r, c, n, abort_threshold)
        scores[r, c] = isfinite(s) ? s / n : 0.0
    end
    return scores
end

function gdpr_local_match(source::AbstractMatrix,
                          models::AbstractVector{<:AbstractMatrix};
                          masks = nothing,
                          gradient_threshold::Real = 0.0,
                          min_score::Real = 0.0)
    n_models, h_out, w_out, h_mdl, w_mdl = _check_multi_models(source, models, masks)
    src_gx, src_gy = _sobel(Float64.(source))
    score_imgs = Vector{Matrix{Float64}}(undef, n_models)

    for k in 1:n_models
        mdl = Float64.(models[k])
        msk = _ensure_mask(masks === nothing ? nothing : masks[k], h_mdl, w_mdl)
        mdl_pts = _model_gradient_points(mdl, msk, Float64(gradient_threshold))
        n = length(mdl_pts.i)
        scores = zeros(h_out, w_out)
        if n > 0
            abort_threshold = min_score > 0 ? -n * (1 - Float64(min_score)) : -Inf
            @inbounds for c in 1:w_out, r in 1:h_out
                s = _gdpr_local_sum_at(src_gx, src_gy, mdl_pts, r, c, n, abort_threshold)
                scores[r, c] = isfinite(s) ? s / n : 0.0
            end
        end
        score_imgs[k] = scores
    end
    return score_imgs
end

"""
    gdpr_local_match_pixelwise(source, models; ...) -> (max_scores, best_index)

Pixel-wise consolidated GDPR-local across multiple models.
"""
function gdpr_local_match_pixelwise(source::AbstractMatrix,
                                     models::AbstractVector{<:AbstractMatrix};
                                     masks = nothing,
                                     gradient_threshold::Real = 0.0,
                                     min_score::Real = 0.0)
    n_models, h_out, w_out, h_mdl, w_mdl = _check_multi_models(source, models, masks)
    src_gx, src_gy = _sobel(Float64.(source))
    max_scores = fill(-Inf, h_out, w_out)
    best_idx = zeros(Int, h_out, w_out)

    for k in 1:n_models
        mdl = Float64.(models[k])
        msk = _ensure_mask(masks === nothing ? nothing : masks[k], h_mdl, w_mdl)
        mdl_pts = _model_gradient_points(mdl, msk, Float64(gradient_threshold))
        n = length(mdl_pts.i)
        n == 0 && continue
        abort_threshold = min_score > 0 ? -n * (1 - Float64(min_score)) : -Inf
        @inbounds for c in 1:w_out, r in 1:h_out
            s = _gdpr_local_sum_at(src_gx, src_gy, mdl_pts, r, c, n, abort_threshold)
            score = isfinite(s) ? s / n : 0.0
            if score > max_scores[r, c]
                max_scores[r, c] = score
                best_idx[r, c] = k
            end
        end
    end
    @inbounds for i in eachindex(max_scores)
        if max_scores[i] == -Inf
            max_scores[i] = 0.0
        end
    end
    return (max_scores, best_idx)
end
