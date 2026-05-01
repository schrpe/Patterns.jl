#= ------------------------------------------------------------------------

    Subpixel-Lokalisierung

    Two complementary functions:

    - subpixel_peak: 3×3 quadratic least-squares fit on a score matrix,
      to refine the position of a discrete maximum to subpixel accuracy.
      Closed-form solution.

    - refine_pose_similarity: iterative Gauss-Newton refinement of a
      similarity pose (rotation + uniform scale + translation) using the
      perpendicular-to-tangent distance between matched image edge points
      and model edges (PDF-Folie 465).

------------------------------------------------------------------------ =#

export subpixel_peak, refine_pose_similarity


"""
    subpixel_peak(scores::AbstractMatrix, peak::CartesianIndex{2})
        -> (dx::Float64, dy::Float64, refined_score::Float64)

Refine the position of a discrete score maximum at `peak` to subpixel
accuracy via a 2D-quadratic least-squares fit on the 3×3 neighborhood
around `peak`.

Returns the fractional offsets `(dx, dy)` along the rows / columns axes
(typically in `[-0.5, 0.5]`) and the interpolated score at the refined
position.

The fit uses the model
`f(x, y) = c₀ + c₁·x + c₂·y + c₃·x² + c₄·y² + c₅·xy`
solved in closed form over the 9 samples at `(i, j) ∈ {-1, 0, 1}²`. The
extremum location is found analytically.

Falls back to `(0.0, 0.0, scores[peak])` when:

- `peak` is on the image border (no full 3×3 neighborhood)
- the Hessian is degenerate or non-negative-definite (peak is not a maximum)
- the refined offset would lie outside `[-1, 1]` (the discrete sample is
  not the closest to the true maximum)

```jldoctest
julia> using Patterns

julia> scores = [1.0 - ((i - 3.3)^2 + (j - 2.8)^2) for i in 1:5, j in 1:5];

julia> dx, dy, _ = subpixel_peak(scores, CartesianIndex(3, 3));

julia> isapprox(dx, 0.3; atol=1e-10) && isapprox(dy, -0.2; atol=1e-10)
true
```
"""
function subpixel_peak(scores::AbstractMatrix, peak::CartesianIndex{2})
    i, j = Tuple(peak)
    h, w = size(scores)
    if i < 2 || i > h - 1 || j < 2 || j > w - 1
        return (0.0, 0.0, Float64(scores[peak]))
    end

    S0 = 0.0      # Σ f
    Si = 0.0      # Σ i·f
    Sj = 0.0      # Σ j·f
    Sii = 0.0     # Σ i²·f
    Sjj = 0.0     # Σ j²·f
    Sij = 0.0     # Σ i·j·f

    @inbounds for di in -1:1, dj in -1:1
        f = Float64(scores[i + di, j + dj])
        S0  += f
        Si  += di * f
        Sj  += dj * f
        Sii += di * di * f
        Sjj += dj * dj * f
        Sij += di * dj * f
    end

    # Closed-form least-squares coefficients (derivation: see plan / inline notes).
    c0 = (5 * S0 - 3 * Sii - 3 * Sjj) / 9
    c1 = Si / 6
    c2 = Sj / 6
    c3 = (3 * Sii - 2 * S0) / 6
    c4 = (3 * Sjj - 2 * S0) / 6
    c5 = Sij / 4

    # Hessian of the quadratic: H = [2c3 c5; c5 2c4]; need H negative-definite for a maximum.
    det = 4 * c3 * c4 - c5 * c5
    if abs(det) < 1e-12 || c3 >= 0 || c4 >= 0
        return (0.0, 0.0, Float64(scores[peak]))
    end

    dx = (-2 * c1 * c4 + c2 * c5) / det
    dy = (-2 * c2 * c3 + c1 * c5) / det

    if abs(dx) > 1.0 || abs(dy) > 1.0
        return (0.0, 0.0, Float64(scores[peak]))
    end

    refined_score = c0 + c1 * dx + c2 * dy + c3 * dx * dx + c4 * dy * dy + c5 * dx * dy
    return (dx, dy, refined_score)
end


"""
    refine_pose_similarity(pose0::Pose2D, correspondences;
                           max_iters::Int=20, tol::Real=1e-8)
        -> (refined::Pose2D, residual_norm::Float64, iterations::Int)

Refine an initial similarity pose `pose0` using iterative Gauss-Newton on
the perpendicular-to-tangent distance from matched image edge points to
the model edge.

Each entry of `correspondences` is a `NamedTuple` with fields:

- `model_point` — 2-vector, model-frame position
- `model_gradient` — 2-vector, gradient direction at that model point
- `image_edge` — 2-vector, matched image edge position (image frame)

The pose is parameterized as a 4-DOF similarity:
`A = [a -b; b a]`, `t = [tx, ty]`. If `pose0.A` is not exactly a
similarity, the closest similarity (`a = (A₁₁ + A₂₂)/2`,
`b = (A₂₁ - A₁₂)/2`) is used as starting point.

The residual for correspondence `i` is the model-frame projection of the
back-transformed edge point onto the model gradient:

    rᵢ(θ) = gᵢ · (A⁻¹ (qᵢ - t) - pᵢ)

Levenberg-Marquardt iterations solve `(JᵀJ + λ·diag(JᵀJ)) Δθ = -Jᵀr`,
update `θ ← θ + Δθ` only if the cost decreases, and adjust the damping
`λ` accordingly. The Jacobian is computed by central finite differences
with a per-parameter step. Iterations stop when either `‖Δθ‖₂ < tol` or
`max_iters` is reached.

Returns the refined `Pose2D`, the final residual L2-norm, and the
number of iterations used.

# Errors
Throws `ArgumentError` if fewer than 2 correspondences are given.
"""
function refine_pose_similarity(pose0::Pose2D, correspondences;
                                max_iters::Int = 20, tol::Real = 1e-8)
    n = length(correspondences)
    n >= 2 || throw(ArgumentError("at least 2 correspondences required (got $n)"))

    # Project pose0.A to the closest similarity
    a = (pose0.A[1, 1] + pose0.A[2, 2]) / 2
    b = (pose0.A[2, 1] - pose0.A[1, 2]) / 2
    tx = pose0.t[1]
    ty = pose0.t[2]
    θ = [a, b, tx, ty]

    r = _residuals_similarity(θ, correspondences)
    cost = sum(abs2, r)
    last_norm = sqrt(cost)
    λ = 1e-3
    iters = 0

    for iter in 1:max_iters
        iters = iter

        # Per-parameter central-difference step
        hs = (max(1e-7, 1e-7 * abs(θ[1])),
              max(1e-7, 1e-7 * abs(θ[2])),
              max(1e-7, 1e-7 * abs(θ[3])),
              max(1e-7, 1e-7 * abs(θ[4])))
        J = Matrix{Float64}(undef, n, 4)
        for k in 1:4
            h = hs[k]
            θp = copy(θ); θp[k] += h
            θm = copy(θ); θm[k] -= h
            rp = _residuals_similarity(θp, correspondences)
            rm = _residuals_similarity(θm, correspondences)
            @inbounds for i in 1:n
                J[i, k] = (rp[i] - rm[i]) / (2 * h)
            end
        end

        H = J' * J
        g = J' * r

        # Inner LM loop: try a step, increase λ on rejection, accept on cost decrease.
        accepted = false
        Δθ = zeros(4)
        for _ in 1:20
            H_lm = copy(H)
            for k in 1:4
                H_lm[k, k] += λ * max(H[k, k], 1e-12) + 1e-12
            end
            Δθ = -(H_lm \ g)

            θ_new = θ .+ Δθ
            # Reject steps that drive the rotation/scale to degeneracy
            if θ_new[1]^2 + θ_new[2]^2 < 1e-12
                λ *= 10
                continue
            end

            r_new = _residuals_similarity(θ_new, correspondences)
            cost_new = sum(abs2, r_new)
            if cost_new < cost
                θ = θ_new
                r = r_new
                cost = cost_new
                last_norm = sqrt(cost)
                λ = max(λ / 10, 1e-12)
                accepted = true
                break
            else
                λ *= 10
                if λ > 1e10
                    break
                end
            end
        end

        if !accepted
            break
        end
        if sqrt(sum(abs2, Δθ)) < tol
            break
        end
    end

    a, b, tx, ty = θ
    return (Pose2D([a -b; b a], [tx, ty]), last_norm, iters)
end


# Helper: residual vector for the similarity parameterization (a, b, tx, ty).
function _residuals_similarity(θ, correspondences)
    n = length(correspondences)
    a, b, tx, ty = θ
    s2 = a * a + b * b
    r = Vector{Float64}(undef, n)
    if s2 < 1e-20
        fill!(r, Inf)
        return r
    end
    @inbounds for i in 1:n
        c = correspondences[i]
        px, py = c.model_point[1], c.model_point[2]
        gx, gy = c.model_gradient[1], c.model_gradient[2]
        qx, qy = c.image_edge[1], c.image_edge[2]
        dx = qx - tx
        dy = qy - ty
        # A⁻¹ (q - t) = (1/s²) · [a b; -b a] · (q - t)
        ux = ( a * dx + b * dy) / s2
        vy = (-b * dx + a * dy) / s2
        r[i] = gx * (ux - px) + gy * (vy - py)
    end
    return r
end
