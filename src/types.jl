#= ------------------------------------------------------------------------

    Patterns.jl — Type definitions

    This file defines the core data types used throughout the package:

    - Pose2D: 2D pose with 2×2 linear part + translation
    - Match{P}: a single match (score + pose-parameter)
    - ShapeModel: list of (point, gradient_vector) — basis for shape-based matching
    - KeypointMatch: a pair of corresponding keypoint positions + Hamming distance
    - TransformResult: result of pose estimation (RANSAC) — pose + inliers/outliers + score + certainty

    Search-model types (SearchModel, RotatedSearchModel, RotatedScaledSearchModel)
    are declared abstractly here and given concrete implementations in their
    respective module files (pyramid.jl, similarity_search.jl) once the
    pyramid types are available.

------------------------------------------------------------------------ =#

import Base: ==, length
export Pose2D, Match, ShapeModel, KeypointMatch, TransformResult
export AbstractSearchModel
export apply_pose, identity_pose, similarity_pose


"""
    AbstractSearchModel

Abstract supertype for all search-model types. Concrete subtypes
(`SearchModel`, `RotatedSearchModel`, `RotatedScaledSearchModel`) are defined
in their respective implementation files.
"""
abstract type AbstractSearchModel end


"""
    Pose2D(A::AbstractMatrix, t::AbstractVector)

A 2D pose described by a 2×2 linear matrix `A` and a 2-element translation
vector `t`. The transformation maps a model point `p` to `A * p + t`.

`A` may encode rotation (orthogonal with `det == 1`), uniform scaling,
similarity (rotation + uniform scale), or any 2×2 linear map. No
orthogonality is enforced.

```jldoctest
julia> using Patterns

julia> p = Pose2D([1.0 0.0; 0.0 1.0], [3.0, 4.0]);

julia> apply_pose(p, [1.0, 1.0])
2-element Vector{Float64}:
 4.0
 5.0
```
"""
struct Pose2D
    A::Matrix{Float64}
    t::Vector{Float64}

    function Pose2D(A::AbstractMatrix, t::AbstractVector)
        size(A) == (2, 2) || throw(ArgumentError("A must be 2×2 (got size $(size(A)))"))
        length(t) == 2 || throw(ArgumentError("t must have length 2 (got $(length(t)))"))
        new(Matrix{Float64}(A), Vector{Float64}(t))
    end
end

==(a::Pose2D, b::Pose2D) = a.A == b.A && a.t == b.t


"""
    identity_pose() -> Pose2D

The identity pose: `A = I`, `t = (0, 0)`.

```jldoctest
julia> using Patterns

julia> identity_pose()
Pose2D([1.0 0.0; 0.0 1.0], [0.0, 0.0])
```
"""
identity_pose() = Pose2D([1.0 0.0; 0.0 1.0], [0.0, 0.0])


"""
    similarity_pose(θ::Real, scale::Real, tx::Real, ty::Real) -> Pose2D
    similarity_pose(θ::Real, scale::Real, t::AbstractVector) -> Pose2D

Construct a similarity pose: rotation by `θ` (radians), uniform `scale`, and
translation `(tx, ty)` (or `t` as a 2-vector).

```jldoctest
julia> using Patterns

julia> p = similarity_pose(0.0, 2.0, 5.0, 0.0);

julia> apply_pose(p, [1.0, 0.0])
2-element Vector{Float64}:
 7.0
 0.0
```
"""
function similarity_pose(θ::Real, scale::Real, tx::Real, ty::Real)
    c, s = cos(θ), sin(θ)
    Pose2D([scale*c -scale*s; scale*s scale*c], [Float64(tx), Float64(ty)])
end
similarity_pose(θ::Real, scale::Real, t::AbstractVector) =
    similarity_pose(θ, scale, t[1], t[2])


"""
    apply_pose(p::Pose2D, point::AbstractVector) -> Vector{Float64}

Apply the pose to a point: returns `A * point + t`.

```jldoctest
julia> using Patterns

julia> apply_pose(identity_pose(), [3.0, 4.0])
2-element Vector{Float64}:
 3.0
 4.0
```
"""
function apply_pose(p::Pose2D, point::AbstractVector)
    length(point) == 2 || throw(ArgumentError("point must have length 2"))
    return p.A * Vector{Float64}(point) + p.t
end


"""
    Match{P}(score::Real, pose::P)

A single match: a similarity score (typically in `[0, 1]`) plus a pose
parameter `pose` of arbitrary type `P`. Common parameter types:

- `Tuple{Int,Int}` — pure pixel translation `(x, y)`
- `NamedTuple{(:x,:y,:θ)}` — translation + rotation
- `NamedTuple{(:x,:y,:θ,:s)}` — translation + rotation + scale
- `Pose2D` — full 2×2 + translation

```jldoctest
julia> using Patterns

julia> Match(0.95, (10, 20))
Match{Tuple{Int64, Int64}}(0.95, (10, 20))
```
"""
struct Match{P}
    score::Float64
    pose::P

    Match(score::Real, pose::P) where {P} = new{P}(Float64(score), pose)
end

==(a::Match, b::Match) = a.score == b.score && a.pose == b.pose


"""
    ShapeModel(points, gradients)

A shape-based model: a list of `(point, gradient_vector)` pairs.

- `points` — model-relative 2D positions (each entry must have length 2)
- `gradients` — local image gradient at each point in the reference image
  (each entry must have length 2)

Both lists must have the same length.

```jldoctest
julia> using Patterns

julia> m = ShapeModel([[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]);

julia> length(m)
2
```
"""
struct ShapeModel
    points::Vector{Vector{Float64}}
    gradients::Vector{Vector{Float64}}

    function ShapeModel(points::AbstractVector, gradients::AbstractVector)
        length(points) == length(gradients) ||
            throw(ArgumentError("points and gradients must have the same length"))
        for (i, p) in enumerate(points)
            length(p) == 2 ||
                throw(ArgumentError("points[$i] must have length 2 (got $(length(p)))"))
        end
        for (i, g) in enumerate(gradients)
            length(g) == 2 ||
                throw(ArgumentError("gradients[$i] must have length 2 (got $(length(g)))"))
        end
        new(
            [Vector{Float64}(p) for p in points],
            [Vector{Float64}(g) for g in gradients],
        )
    end
end

length(m::ShapeModel) = length(m.points)

==(a::ShapeModel, b::ShapeModel) =
    a.points == b.points && a.gradients == b.gradients


"""
    KeypointMatch(model_pos, image_pos, distance)

A correspondence between a model keypoint and an image keypoint. Stores both
2D positions and the Hamming distance between the binary descriptors.

```jldoctest
julia> using Patterns

julia> KeypointMatch((1.0, 2.0), (10.0, 20.0), 42)
KeypointMatch((1.0, 2.0), (10.0, 20.0), 42)
```
"""
struct KeypointMatch
    model_pos::Tuple{Float64,Float64}
    image_pos::Tuple{Float64,Float64}
    distance::Int

    function KeypointMatch(model_pos, image_pos, distance::Integer)
        new(
            (Float64(model_pos[1]), Float64(model_pos[2])),
            (Float64(image_pos[1]), Float64(image_pos[2])),
            Int(distance),
        )
    end
end

==(a::KeypointMatch, b::KeypointMatch) =
    a.model_pos == b.model_pos && a.image_pos == b.image_pos && a.distance == b.distance


"""
    TransformResult(pose, inliers, outliers, score, certainty)

Result of a pose-estimation step (typically RANSAC). Holds:

- `pose` — estimated `Pose2D`
- `inliers` — `KeypointMatch` correspondences consistent with `pose`
- `outliers` — `KeypointMatch` correspondences inconsistent with `pose`
- `score` — overall match quality, in `[0, 1]`
- `certainty` — confidence in the estimate, in `[0, 1]`

```jldoctest
julia> using Patterns

julia> r = TransformResult(identity_pose(), KeypointMatch[], KeypointMatch[], 0.0, 0.0);

julia> r.score
0.0
```
"""
struct TransformResult
    pose::Pose2D
    inliers::Vector{KeypointMatch}
    outliers::Vector{KeypointMatch}
    score::Float64
    certainty::Float64

    function TransformResult(pose::Pose2D, inliers::AbstractVector,
                             outliers::AbstractVector, score::Real, certainty::Real)
        new(
            pose,
            Vector{KeypointMatch}(inliers),
            Vector{KeypointMatch}(outliers),
            Float64(score),
            Float64(certainty),
        )
    end
end

==(a::TransformResult, b::TransformResult) =
    a.pose == b.pose && a.inliers == b.inliers && a.outliers == b.outliers &&
    a.score == b.score && a.certainty == b.certainty
