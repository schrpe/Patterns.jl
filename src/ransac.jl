#= ------------------------------------------------------------------------

    RANSAC-basierte Similarity-Posenschätzung

    Closed-form similarity fitting from two keypoint matches, a single-pose
    RANSAC search, and a multi-instance variant that recovers several
    occurrences of the same model in the image.

    Conventions
    -----------
    All positions in `KeypointMatch` are `(row, col)` tuples. The
    similarity transform is parameterized as
        q = A · p + t,    A = [a -b; b a]
    where `a = scale·cos(θ)` and `b = scale·sin(θ)` (in the (row, col)
    frame). `Pose2D(A, t)` carries the result.

    Algorithm
    ---------
    Each RANSAC iteration:
      1. samples two distinct keypoint matches,
      2. fits the unique similarity that maps the two model points to the
         two image points,
      3. rejects the candidate if its scale is outside `[min_scale, max_scale]`,
      4. classifies all matches as inliers / outliers using a Euclidean
         distance threshold `error`,
      5. keeps the best candidate (most inliers; ties broken by lower
         mean inlier distance).

    The final score and certainty: certainty grows linearly from 0 at
    2 inliers to 1 at 7+ inliers, and score is
    `1 - (mean_inlier_distance / (1.8·error))²`.

    Multi-instance variant (`estimate_all_poses`) iterates RANSAC on the
    remaining matches, stripping outliers that fall inside the
    bounding-box of each accepted instance — so distinct image
    occurrences of the same model can all be recovered.

    Default iteration count of 40 corresponds to an outlier ratio of
    0.5 and a target success probability of 0.99999.

------------------------------------------------------------------------ =#

export compute_similarity_pose, estimate_pose, estimate_all_poses


# ------------------------------------------------------------------------
# Closed-form similarity from two matches
# ------------------------------------------------------------------------

"""
    compute_similarity_pose(match1::KeypointMatch, match2::KeypointMatch) -> Pose2D

Closed-form similarity pose `q = A · p + t` (with `A = [a -b; b a]`)
that maps the two model points (`match*.model_pos`) to the two image
points (`match*.image_pos`). Throws `ArgumentError` if the two model
points coincide.
"""
function compute_similarity_pose(match1::KeypointMatch, match2::KeypointMatch)
    p1r, p1c = match1.model_pos
    p2r, p2c = match2.model_pos
    q1r, q1c = match1.image_pos
    q2r, q2c = match2.image_pos

    Δp_r = p2r - p1r
    Δp_c = p2c - p1c
    Δq_r = q2r - q1r
    Δq_c = q2c - q1c

    l2 = Δp_r * Δp_r + Δp_c * Δp_c
    l2 > 1e-20 ||
        throw(ArgumentError("the two model points coincide; cannot estimate similarity"))

    # Solve [Δp_r -Δp_c; Δp_c Δp_r] · [a; b] = [Δq_r; Δq_c]
    a = (Δp_r * Δq_r + Δp_c * Δq_c) / l2
    b = (Δp_r * Δq_c - Δp_c * Δq_r) / l2

    # Translation: t = q_mid - A · p_mid (averaging both pairs for stability)
    p_mid_r = 0.5 * (p1r + p2r)
    p_mid_c = 0.5 * (p1c + p2c)
    q_mid_r = 0.5 * (q1r + q2r)
    q_mid_c = 0.5 * (q1c + q2c)
    t_r = q_mid_r - (a * p_mid_r - b * p_mid_c)
    t_c = q_mid_c - (b * p_mid_r + a * p_mid_c)

    return Pose2D([a -b; b a], [t_r, t_c])
end


# ------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------

# Apply a Pose2D to a (row, col) tuple.
@inline function _apply(pose::Pose2D, p::Tuple{Float64,Float64})
    A = pose.A
    t = pose.t
    return (A[1, 1] * p[1] + A[1, 2] * p[2] + t[1],
            A[2, 1] * p[1] + A[2, 2] * p[2] + t[2])
end

# Recover the uniform scale of a Pose2D whose `A` is a similarity matrix.
@inline _pose_scale(pose::Pose2D) = sqrt(pose.A[1, 1]^2 + pose.A[2, 1]^2)

# One RANSAC step: classify matches, return (inliers, outliers, mean_distance).
function _classify(pose::Pose2D, matches, error::Float64)
    inliers = KeypointMatch[]
    outliers = KeypointMatch[]
    distance_sum = 0.0
    for m in matches
        tp = _apply(pose, m.model_pos)
        d = sqrt((tp[1] - m.image_pos[1])^2 + (tp[2] - m.image_pos[2])^2)
        if d < error
            push!(inliers, m)
            distance_sum += d
        else
            push!(outliers, m)
        end
    end
    mean_distance = isempty(inliers) ? Inf : distance_sum / length(inliers)
    return (inliers, outliers, mean_distance)
end


# ------------------------------------------------------------------------
# RANSAC: single-best similarity pose
# ------------------------------------------------------------------------

"""
    estimate_pose(matches::AbstractVector{KeypointMatch};
                  error::Real = 5.0,
                  min_scale::Real = 0.8,
                  max_scale::Real = 1.2,
                  max_iterations::Int = 40)
        -> TransformResult

RANSAC estimate of the best similarity pose explaining the largest
subset of `matches`. Returns a `TransformResult` with the estimated
pose, inlier and outlier match lists, an overall `score ∈ [0, 1]` and a
`certainty ∈ [0, 1]`.

If fewer than 2 matches are supplied, throws `ArgumentError`. If no
candidate satisfies `min_scale ≤ scale ≤ max_scale` and yields any
inliers, returns a `TransformResult` with the identity pose, all
matches as outliers, `score = 0`, `certainty = 0`.
"""
function estimate_pose(matches::AbstractVector{KeypointMatch};
                       error::Real = 5.0,
                       min_scale::Real = 0.8,
                       max_scale::Real = 1.2,
                       max_iterations::Int = 40)
    n = length(matches)
    n >= 2 || throw(ArgumentError("at least 2 matches required, got $n"))
    error >= 0 || throw(ArgumentError("error must be ≥ 0"))
    min_scale > 0 || throw(ArgumentError("min_scale must be > 0"))
    max_scale >= min_scale ||
        throw(ArgumentError("max_scale must be ≥ min_scale"))
    max_iterations >= 1 ||
        throw(ArgumentError("max_iterations must be ≥ 1"))

    err_f = Float64(error)
    minS = Float64(min_scale)
    maxS = Float64(max_scale)

    best_pose = identity_pose()
    best_inliers = KeypointMatch[]
    best_outliers = collect(matches)
    best_inlier_count = 0
    best_mean_distance = Inf

    for _ in 1:max_iterations
        i1 = rand(1:n)
        i2 = rand(1:n)
        i1 == i2 && continue

        pose = try
            compute_similarity_pose(matches[i1], matches[i2])
        catch e
            e isa ArgumentError ? continue : rethrow(e)
        end

        scale = _pose_scale(pose)
        (minS <= scale <= maxS) || continue

        inliers, outliers, mean_d = _classify(pose, matches, err_f)
        ic = length(inliers)
        ic == 0 && continue

        if mean_d < err_f && (ic > best_inlier_count ||
                              (ic == best_inlier_count && mean_d < best_mean_distance))
            best_pose = pose
            best_inliers = inliers
            best_outliers = outliers
            best_inlier_count = ic
            best_mean_distance = mean_d
        end
    end

    # Score and certainty
    certainty = clamp((best_inlier_count - 2) / 5.0, 0.0, 1.0)
    score = 0.0
    if best_inlier_count > 0 && isfinite(best_mean_distance)
        max_err = 1.8 * err_f
        if max_err > 0
            score = clamp(1.0 - (best_mean_distance / max_err)^2, 0.0, 1.0)
        end
    end
    if best_inlier_count == 0
        score = 0.0
    end

    return TransformResult(best_pose, best_inliers, best_outliers, score, certainty)
end


# ------------------------------------------------------------------------
# Multi-instance: find all distinct similarity poses
# ------------------------------------------------------------------------

"""
    estimate_all_poses(matches::AbstractVector{KeypointMatch};
                       error::Real = 5.0,
                       min_scale::Real = 0.8,
                       max_scale::Real = 1.2,
                       max_iterations::Int = 40,
                       min_inliers::Int = 4) -> Vector{TransformResult}

Iteratively estimate similarity poses, peeling off inliers (and any
outliers whose image position falls inside the inlier bounding box)
after each successful fit. Stops when either:

- the remaining match count drops below 4, or
- a fit produces fewer than `min_inliers` inliers.

Recovers distinct image-frame instances of the same model.
"""
function estimate_all_poses(matches::AbstractVector{KeypointMatch};
                            error::Real = 5.0,
                            min_scale::Real = 0.8,
                            max_scale::Real = 1.2,
                            max_iterations::Int = 40,
                            min_inliers::Int = 4)
    min_inliers >= 2 || throw(ArgumentError("min_inliers must be ≥ 2"))

    results = TransformResult[]
    remaining = collect(matches)

    while length(remaining) > 3
        result = estimate_pose(remaining;
                                error = error,
                                min_scale = min_scale,
                                max_scale = max_scale,
                                max_iterations = max_iterations)
        if length(result.inliers) >= min_inliers
            push!(results, result)

            # Bounding box of inliers in image-frame coordinates
            min_r = minimum(m.image_pos[1] for m in result.inliers)
            max_r = maximum(m.image_pos[1] for m in result.inliers)
            min_c = minimum(m.image_pos[2] for m in result.inliers)
            max_c = maximum(m.image_pos[2] for m in result.inliers)

            # Carry forward only outliers outside that box
            new_remaining = KeypointMatch[]
            for m in result.outliers
                r, c = m.image_pos
                if r < min_r || r > max_r || c < min_c || c > max_c
                    push!(new_remaining, m)
                end
            end
            remaining = new_remaining
        else
            break
        end
    end

    return results
end


"""
    estimate_all_poses(match_lists::AbstractVector{<:AbstractVector{KeypointMatch}};
                       kwargs...) -> Vector{Vector{TransformResult}}

Multi-model dispatch. Runs `estimate_all_poses` on each per-model match
list and returns one result list per model, in the same order.
"""
function estimate_all_poses(match_lists::AbstractVector{<:AbstractVector{KeypointMatch}};
                             kwargs...)
    isempty(match_lists) &&
        throw(ArgumentError("match_lists must be non-empty"))
    return [estimate_all_poses(matches; kwargs...) for matches in match_lists]
end
