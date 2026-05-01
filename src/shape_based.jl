#= ------------------------------------------------------------------------

    Shape-Based Matching

    A `ShapeModel` is a list of (point, gradient_vector) pairs extracted
    from a reference image's gradient. The model is matched against the
    Sobel gradient of the source image under a 2×2 transformation A and a
    translation t (combined as a `Pose2D`):

      pᵢ' = A · pᵢ           (transformed model point)
      dᵢ' = (A⁻¹)ᵀ · dᵢ      (covariant gradient transformation)

    The match score at search position q = t evaluates the cosine
    similarity between each transformed model gradient dᵢ' and the source
    gradient e at position q + pᵢ'.

    Three variants (PDF-Folien 459–462):

      GDP            : max(0, Σ rᵢ / n)              — no contrast reversal
      GDPR (global)  : |Σ rᵢ| / n                    — global contrast reversal
      GDPR (lokal)   : Σ |rᵢ| / n                    — local contrast reversal

    Bias correction (PDF-Folie 463): points where the source gradient
    magnitude is below `min_source_gradient` are skipped — their
    contribution would otherwise bias the score by 2/π ≈ 0.637 under random
    orientation noise.

    GDP and GDPR-local support early termination via `min_score`
    (PDF-Folie 464).

------------------------------------------------------------------------ =#

export image_gradients, build_shape_model, transform_shape_model
export shape_score_gdp, shape_score_gdpr, shape_score_gdpr_local


# ------------------------------------------------------------------------
# Public Sobel wrapper
# ------------------------------------------------------------------------

"""
    image_gradients(image::AbstractMatrix) -> (gx::Matrix{Float64}, gy::Matrix{Float64})

Sobel gradient pair of `image`. Boundary pixels are returned as `0`.

`gy[i, j]` is the row-direction derivative (∂img/∂row), and `gx[i, j]` is
the column-direction derivative (∂img/∂col). The gradient vector at
pixel `(i, j)` in `(row, col)` order is therefore `(gy[i, j], gx[i, j])`.
"""
function image_gradients(image::AbstractMatrix)
    return _sobel(Float64.(image))
end


# ------------------------------------------------------------------------
# Building / transforming a ShapeModel
# ------------------------------------------------------------------------

"""
    build_shape_model(image::AbstractMatrix;
                      gradient_threshold::Real = 0.0,
                      mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                      origin::Union{Nothing,Tuple{Real,Real}} = nothing)
        -> ShapeModel

Build a `ShapeModel` from a reference `image` by selecting all pixels with
Sobel gradient magnitude above `gradient_threshold`. Each model point's
position is stored relative to `origin` in `(row, col)` order; if `origin`
is `nothing`, the image centre `((h+1)/2, (w+1)/2)` is used.

The gradient stored at each point is `(gy, gx)` — i.e. the gradient vector
in `(row, col)` order, consistent with how positions are stored.

```jldoctest
julia> using Patterns

julia> img = Float64[i + j for i in 1:5, j in 1:5];

julia> m = build_shape_model(img; gradient_threshold = 1.0);

julia> length(m) > 0
true
```
"""
function build_shape_model(image::AbstractMatrix;
                            gradient_threshold::Real = 0.0,
                            mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                            origin::Union{Nothing,Tuple{Real,Real}} = nothing)
    h, w = size(image)
    img = Float64.(image)
    gx, gy = _sobel(img)
    msk = mask === nothing ? trues(h, w) : BitMatrix(mask)
    size(msk) == (h, w) || throw(ArgumentError("mask size $(size(msk)) ≠ image size ($h, $w)"))

    org_r, org_c = origin === nothing ? ((h + 1) / 2, (w + 1) / 2) :
                   (Float64(origin[1]), Float64(origin[2]))

    points = Vector{Vector{Float64}}()
    gradients = Vector{Vector{Float64}}()
    @inbounds for j in 1:w, i in 1:h
        if msk[i, j]
            mag = sqrt(gx[i, j]^2 + gy[i, j]^2)
            if mag > gradient_threshold
                push!(points, [i - org_r, j - org_c])
                push!(gradients, [gy[i, j], gx[i, j]])  # (row-, col-component)
            end
        end
    end
    return ShapeModel(points, gradients)
end


"""
    transform_shape_model(model::ShapeModel, A::AbstractMatrix) -> ShapeModel

Return a new `ShapeModel` whose points are transformed by `A` and whose
gradients are transformed by `(A⁻¹)ᵀ` (covariant rule):

    p' = A · p,    d' = (A⁻¹)ᵀ · d

Useful for pre-computing rotated and/or scaled model variants once,
allowing the inner search loop to reduce to translation only.

```jldoctest
julia> using Patterns

julia> m = ShapeModel([[1.0, 0.0]], [[1.0, 0.0]]);

julia> m_rot = transform_shape_model(m, [0.0 -1.0; 1.0 0.0]);

julia> m_rot.points[1] ≈ [0.0, 1.0]
true

julia> m_rot.gradients[1] ≈ [0.0, 1.0]
true
```
"""
function transform_shape_model(model::ShapeModel, A::AbstractMatrix)
    size(A) == (2, 2) || throw(ArgumentError("A must be 2×2"))
    A_mat = Matrix{Float64}(A)
    Ainv_T = transpose(inv(A_mat))
    new_points = [A_mat * p for p in model.points]
    new_gradients = [Ainv_T * d for d in model.gradients]
    return ShapeModel(new_points, new_gradients)
end


# ------------------------------------------------------------------------
# Bilinear gradient lookup
# ------------------------------------------------------------------------

# Bilinearly interpolated source gradient at fractional position (r, c).
# Returns (egx, egy); returns (0.0, 0.0) if (r, c) is outside [1, h]×[1, w].
@inline function _bilinear_gradient(src_gx::Matrix{Float64}, src_gy::Matrix{Float64},
                                     r::Float64, c::Float64)
    h, w = size(src_gx)
    if r < 1 || r > h || c < 1 || c > w
        return (0.0, 0.0)
    end
    i0 = floor(Int, r)
    j0 = floor(Int, c)
    i1 = min(i0 + 1, h)
    j1 = min(j0 + 1, w)
    dr = r - i0
    dc = c - j0
    w00 = (1 - dr) * (1 - dc)
    w01 = (1 - dr) * dc
    w10 = dr * (1 - dc)
    w11 = dr * dc
    @inbounds egx = w00 * src_gx[i0, j0] + w01 * src_gx[i0, j1] +
                    w10 * src_gx[i1, j0] + w11 * src_gx[i1, j1]
    @inbounds egy = w00 * src_gy[i0, j0] + w01 * src_gy[i0, j1] +
                    w10 * src_gy[i1, j0] + w11 * src_gy[i1, j1]
    return (egx, egy)
end


# ------------------------------------------------------------------------
# Inner accumulators
# ------------------------------------------------------------------------

# Signed cosine sum for GDP / GDPR-global, with optional early abort.
@inline function _shape_sum(src_gx::Matrix{Float64}, src_gy::Matrix{Float64},
                             pts::Vector{Vector{Float64}},
                             grads::Vector{Vector{Float64}},
                             A::Matrix{Float64}, Ainv_T::Matrix{Float64},
                             tr::Float64, tc::Float64,
                             n::Int, min_source_gradient::Float64,
                             abort_threshold::Float64)
    s = 0.0
    @inbounds for k in 1:n
        p = pts[k]
        d = grads[k]
        # Transformed model point
        pr = A[1, 1] * p[1] + A[1, 2] * p[2]
        pc = A[2, 1] * p[1] + A[2, 2] * p[2]
        egx, egy = _bilinear_gradient(src_gx, src_gy, tr + pr, tc + pc)
        emag = sqrt(egx * egx + egy * egy)
        if emag > min_source_gradient
            # Transformed model gradient
            drow = Ainv_T[1, 1] * d[1] + Ainv_T[1, 2] * d[2]
            dcol = Ainv_T[2, 1] * d[1] + Ainv_T[2, 2] * d[2]
            dmag = sqrt(drow * drow + dcol * dcol)
            if dmag > 0
                s += (drow * egy + dcol * egx) / (dmag * emag)
            end
        end
        if isfinite(abort_threshold) && s < k + abort_threshold
            return -Inf
        end
    end
    return s
end

# Absolute-cosine sum for GDPR-local; same early-abort formula.
@inline function _shape_local_sum(src_gx::Matrix{Float64}, src_gy::Matrix{Float64},
                                   pts::Vector{Vector{Float64}},
                                   grads::Vector{Vector{Float64}},
                                   A::Matrix{Float64}, Ainv_T::Matrix{Float64},
                                   tr::Float64, tc::Float64,
                                   n::Int, min_source_gradient::Float64,
                                   abort_threshold::Float64)
    s = 0.0
    @inbounds for k in 1:n
        p = pts[k]
        d = grads[k]
        pr = A[1, 1] * p[1] + A[1, 2] * p[2]
        pc = A[2, 1] * p[1] + A[2, 2] * p[2]
        egx, egy = _bilinear_gradient(src_gx, src_gy, tr + pr, tc + pc)
        emag = sqrt(egx * egx + egy * egy)
        if emag > min_source_gradient
            drow = Ainv_T[1, 1] * d[1] + Ainv_T[1, 2] * d[2]
            dcol = Ainv_T[2, 1] * d[1] + Ainv_T[2, 2] * d[2]
            dmag = sqrt(drow * drow + dcol * dcol)
            if dmag > 0
                s += abs((drow * egy + dcol * egx) / (dmag * emag))
            end
        end
        if isfinite(abort_threshold) && s < k + abort_threshold
            return -Inf
        end
    end
    return s
end


# ------------------------------------------------------------------------
# Score functions
# ------------------------------------------------------------------------

"""
    shape_score_gdp(src_gx, src_gy, model::ShapeModel, pose::Pose2D;
                    min_source_gradient::Real = 0.0,
                    min_score::Real = 0.0) -> Float64

Shape-Based GDP score at the pose `pose`. Range `[0, 1]`. Negative
sums are clamped to `0` (no contrast reversal).

`min_source_gradient` skips model points whose corresponding source
gradient magnitude is below the threshold (PDF-Folie 463 bias correction).

`min_score > 0` enables early-abort in the inner loop (PDF-Folie 464);
aborted positions return `0`.

The translation `pose.t` is interpreted in `(row, col)` order — `pose.t[1]`
is the row coordinate of the model origin, `pose.t[2]` the column.
"""
function shape_score_gdp(src_gx::AbstractMatrix, src_gy::AbstractMatrix,
                          model::ShapeModel, pose::Pose2D;
                          min_source_gradient::Real = 0.0,
                          min_score::Real = 0.0)
    n = length(model)
    n > 0 || return 0.0
    A = pose.A
    Ainv_T = transpose(inv(A))
    abort_threshold = min_score > 0 ? -n * (1 - Float64(min_score)) : -Inf
    s = _shape_sum(Matrix{Float64}(src_gx), Matrix{Float64}(src_gy),
                   model.points, model.gradients,
                   Matrix{Float64}(A), Matrix{Float64}(Ainv_T),
                   Float64(pose.t[1]), Float64(pose.t[2]),
                   n, Float64(min_source_gradient), abort_threshold)
    return isfinite(s) ? max(0.0, s / n) : 0.0
end

"""
    shape_score_gdp(src_gx, src_gy, model::ShapeModel, t::AbstractVector;
                    kwargs...) -> Float64

Translation-only GDP evaluation. Equivalent to using `Pose2D(I, t)` —
intended for repeated calls when the model has been pre-transformed via
`transform_shape_model`.
"""
shape_score_gdp(src_gx, src_gy, model::ShapeModel, t::AbstractVector; kwargs...) =
    shape_score_gdp(src_gx, src_gy, model,
                    Pose2D([1.0 0.0; 0.0 1.0], collect(t));
                    kwargs...)


"""
    shape_score_gdpr(src_gx, src_gy, model::ShapeModel, pose::Pose2D;
                     min_source_gradient::Real = 0.0) -> Float64

Shape-Based GDPR (global contrast reversal): `|Σ rᵢ| / n`, range `[0, 1]`.
No early-abort criterion is applicable for this metric.
"""
function shape_score_gdpr(src_gx::AbstractMatrix, src_gy::AbstractMatrix,
                           model::ShapeModel, pose::Pose2D;
                           min_source_gradient::Real = 0.0)
    n = length(model)
    n > 0 || return 0.0
    A = pose.A
    Ainv_T = transpose(inv(A))
    s = _shape_sum(Matrix{Float64}(src_gx), Matrix{Float64}(src_gy),
                   model.points, model.gradients,
                   Matrix{Float64}(A), Matrix{Float64}(Ainv_T),
                   Float64(pose.t[1]), Float64(pose.t[2]),
                   n, Float64(min_source_gradient), -Inf)
    return abs(s) / n
end

shape_score_gdpr(src_gx, src_gy, model::ShapeModel, t::AbstractVector; kwargs...) =
    shape_score_gdpr(src_gx, src_gy, model,
                     Pose2D([1.0 0.0; 0.0 1.0], collect(t));
                     kwargs...)


"""
    shape_score_gdpr_local(src_gx, src_gy, model::ShapeModel, pose::Pose2D;
                           min_source_gradient::Real = 0.0,
                           min_score::Real = 0.0) -> Float64

Shape-Based GDPR (local contrast reversal): `Σ |rᵢ| / n`, range `[0, 1]`.
Each gradient pair may be inverted independently — the score remains `1`
under locally inverted contrast (PDF-Folie 462).

`min_score > 0` enables early-abort.
"""
function shape_score_gdpr_local(src_gx::AbstractMatrix, src_gy::AbstractMatrix,
                                 model::ShapeModel, pose::Pose2D;
                                 min_source_gradient::Real = 0.0,
                                 min_score::Real = 0.0)
    n = length(model)
    n > 0 || return 0.0
    A = pose.A
    Ainv_T = transpose(inv(A))
    abort_threshold = min_score > 0 ? -n * (1 - Float64(min_score)) : -Inf
    s = _shape_local_sum(Matrix{Float64}(src_gx), Matrix{Float64}(src_gy),
                         model.points, model.gradients,
                         Matrix{Float64}(A), Matrix{Float64}(Ainv_T),
                         Float64(pose.t[1]), Float64(pose.t[2]),
                         n, Float64(min_source_gradient), abort_threshold)
    return isfinite(s) ? s / n : 0.0
end

shape_score_gdpr_local(src_gx, src_gy, model::ShapeModel, t::AbstractVector; kwargs...) =
    shape_score_gdpr_local(src_gx, src_gy, model,
                            Pose2D([1.0 0.0; 0.0 1.0], collect(t));
                            kwargs...)
