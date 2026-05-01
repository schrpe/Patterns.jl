#= ------------------------------------------------------------------------

    Keypoint-Matching mit Hamming-Distanz und Lowe's Ratio Test

    `match_keypoints(template_kps, image_kps; ...)`: for each template
    keypoint, find the best and second-best image keypoint by Hamming
    distance and accept the match only if the best distance is strictly
    below `ratio · second_best` (Lowe's ratio test, not provided by
    `ImageFeatures.match_keypoints`).

    `match_all_keypoints` is the simpler unfiltered variant — every pair
    with `distance ≤ max_distance` is reported (no ratio test).

    Multi-template dispatch: pass a `Vector{Vector{PatternKeypoint}}` as
    the first argument and the source keypoints (computed once) as the
    second; receive `Vector{Vector{KeypointMatch}}` — one list per
    template, in the same order.

------------------------------------------------------------------------ =#

export match_keypoints, match_all_keypoints


# Hamming distance between two binary descriptors.
@inline function _hamming(a::BitVector, b::BitVector)
    length(a) == length(b) ||
        throw(DimensionMismatch("descriptor lengths differ: $(length(a)) ≠ $(length(b))"))
    return count(a .⊻ b)
end


"""
    match_keypoints(template_keypoints::AbstractVector{PatternKeypoint},
                    image_keypoints::AbstractVector{PatternKeypoint};
                    max_distance::Real = 80,
                    ratio::Real = 0.75) -> Vector{KeypointMatch}

For each template keypoint, find the closest and second-closest image
keypoint by Hamming distance. Accept the match if both:

1. `best_distance < max_distance` (initial bound on the running best)
2. `best_distance < ratio · second_best_distance` (Lowe's ratio test)

If only one image keypoint lies within `max_distance`, the second-best
distance defaults to `max_distance`, so the test becomes
`best < max_distance · ratio`.

Each accepted `KeypointMatch` carries:

- `model_pos` — the template keypoint's position
- `image_pos` — the matched image keypoint's position
- `distance` — the Hamming distance between their descriptors

Note: a template keypoint can only contribute a single match. Multiple
template keypoints may map to the same image keypoint (no uniqueness
constraint is enforced — downstream RANSAC handles such ambiguity).
"""
function match_keypoints(template_keypoints::AbstractVector{PatternKeypoint},
                          image_keypoints::AbstractVector{PatternKeypoint};
                          max_distance::Real = 80,
                          ratio::Real = 0.75)
    0 < ratio <= 1 || throw(ArgumentError("ratio must be in (0, 1]"))
    max_distance >= 0 || throw(ArgumentError("max_distance must be ≥ 0"))

    matches = KeypointMatch[]
    md = Int(round(max_distance))

    for tkp in template_keypoints
        best_distance = md
        second_best_distance = md
        best_image_kp = nothing

        for ikp in image_keypoints
            d = _hamming(tkp.descriptor, ikp.descriptor)
            if d < best_distance
                second_best_distance = best_distance
                best_distance = d
                best_image_kp = ikp
            elseif d < second_best_distance
                second_best_distance = d
            end
        end

        # Lowe's ratio test: best < second_best · ratio
        if best_image_kp !== nothing &&
           best_distance < second_best_distance * ratio
            push!(matches,
                  KeypointMatch(tkp.position,
                                best_image_kp.position,
                                best_distance))
        end
    end

    return matches
end


"""
    match_keypoints(template_keypoint_sets::AbstractVector{<:AbstractVector{PatternKeypoint}},
                    image_keypoints::AbstractVector{PatternKeypoint};
                    max_distance::Real = 80,
                    ratio::Real = 0.75) -> Vector{Vector{KeypointMatch}}

Multi-template dispatch. The image keypoints (typically detected from a
single search image) are matched once against each template keypoint
set, and a vector of match lists — one per template, in the same order
— is returned.
"""
function match_keypoints(template_keypoint_sets::AbstractVector{<:AbstractVector{PatternKeypoint}},
                          image_keypoints::AbstractVector{PatternKeypoint};
                          max_distance::Real = 80,
                          ratio::Real = 0.75)
    isempty(template_keypoint_sets) &&
        throw(ArgumentError("template_keypoint_sets must be non-empty"))
    return [match_keypoints(t, image_keypoints;
                             max_distance = max_distance, ratio = ratio)
            for t in template_keypoint_sets]
end


"""
    match_all_keypoints(template_keypoints::AbstractVector{PatternKeypoint},
                        image_keypoints::AbstractVector{PatternKeypoint};
                        max_distance::Real = 80) -> Vector{KeypointMatch}

Unfiltered keypoint matching: report every (template, image) pair whose
Hamming distance is strictly below `max_distance`. No Lowe's ratio test
is applied. Useful when downstream consensus (e.g. RANSAC) is meant to
absorb ambiguous correspondences.
"""
function match_all_keypoints(template_keypoints::AbstractVector{PatternKeypoint},
                              image_keypoints::AbstractVector{PatternKeypoint};
                              max_distance::Real = 80)
    max_distance >= 0 || throw(ArgumentError("max_distance must be ≥ 0"))
    md = Int(round(max_distance))

    matches = KeypointMatch[]
    for tkp in template_keypoints
        for ikp in image_keypoints
            d = _hamming(tkp.descriptor, ikp.descriptor)
            if d < md
                push!(matches,
                      KeypointMatch(tkp.position, ikp.position, d))
            end
        end
    end
    return matches
end
