#= ------------------------------------------------------------------------

    Vorberechnete Rotation × Skalen-Modellvarianten

    `RotatedScaledSearchModel` bundles a base `ShapeModel` together with a
    grid of pre-transformed variants, one per (θ, scale) pair. Each
    variant's `model` field is `transform_shape_model(base_model, A)` for
    `A = scale · R(θ)` — i.e. the transformation that combines uniform
    scaling and rotation in the (row, col) frame.

    The grid is produced by `create_similarity_template` from a reference
    image, with user-controlled rotation/scale ranges and granularities.

    A subsequent pyramidensuche (step 7+) will use this type to enumerate
    candidates on the coarsest pyramid level and refine them on finer
    levels with halved angular/scale granularity per level.

------------------------------------------------------------------------ =#

export ShapeVariant, RotatedScaledSearchModel
export create_similarity_template, nearest_variant


"""
    ShapeVariant(θ::Float64, scale::Float64, model::ShapeModel)

A single rotation-and-scale variant: stores the rotation angle `θ`
(radians), the uniform `scale` factor, and the resulting transformed
`ShapeModel`.
"""
struct ShapeVariant
    θ::Float64
    scale::Float64
    model::ShapeModel
end

==(a::ShapeVariant, b::ShapeVariant) =
    a.θ == b.θ && a.scale == b.scale && a.model == b.model


"""
    RotatedScaledSearchModel <: AbstractSearchModel

A search model holding the original `base_model` together with all
pre-computed `(θ, scale)` variants of it. The grid is regular: angles step
by `angular_granularity` from `rotation_range[1]` to `rotation_range[2]`,
and scales step by `scale_granularity` from `scale_range[1]` to
`scale_range[2]`. Variants are stored in row-major order (outer loop on
angles, inner on scales).

Use `create_similarity_template` to construct, `length` to query the
number of variants, indexing `m[k]` to fetch the k-th, iteration to walk
all of them, and `nearest_variant(m, θ, scale)` to look up the variant
nearest to a target pose on the (θ, scale) grid.
"""
struct RotatedScaledSearchModel <: AbstractSearchModel
    base_model::ShapeModel
    variants::Vector{ShapeVariant}
    rotation_range::Tuple{Float64,Float64}
    angular_granularity::Float64
    n_rotations::Int
    scale_range::Tuple{Float64,Float64}
    scale_granularity::Float64
    n_scales::Int
end

length(m::RotatedScaledSearchModel) = length(m.variants)
Base.getindex(m::RotatedScaledSearchModel, k::Integer) = m.variants[k]
Base.firstindex(m::RotatedScaledSearchModel) = 1
Base.lastindex(m::RotatedScaledSearchModel) = length(m.variants)
Base.iterate(m::RotatedScaledSearchModel, state::Int = 1) =
    state > length(m) ? nothing : (m[state], state + 1)


"""
    create_similarity_template(image::AbstractMatrix;
                               rotation_range = (0.0, 0.0),
                               angular_granularity::Real = π/180,
                               scale_range = (1.0, 1.0),
                               scale_granularity::Real = 0.1,
                               gradient_threshold::Real = 0.0,
                               mask = nothing,
                               origin = nothing) -> RotatedScaledSearchModel

Build a `RotatedScaledSearchModel` by:

1. Constructing the base `ShapeModel` from `image` via `build_shape_model`
   (with the supplied `gradient_threshold`, `mask`, `origin`).
2. Enumerating all `(θ, scale)` pairs on the regular grid defined by the
   rotation/scale ranges and granularities.
3. Pre-computing each `(θ, scale)` variant by `transform_shape_model`
   with `A = scale · R(θ)`.

The defaults `rotation_range = (0.0, 0.0)` and `scale_range = (1.0, 1.0)`
produce a single identity variant (equivalent to the base shape model).

```jldoctest
julia> using Patterns

julia> img = Float64[(i-3)^2 + (j-3)^2 for i in 1:5, j in 1:5];

julia> m = create_similarity_template(img;
                                      rotation_range = (0.0, π/2),
                                      angular_granularity = π/4,
                                      gradient_threshold = 1.0);

julia> length(m)
3
```
"""
function create_similarity_template(image::AbstractMatrix;
                                     rotation_range = (0.0, 0.0),
                                     angular_granularity::Real = π / 180,
                                     scale_range = (1.0, 1.0),
                                     scale_granularity::Real = 0.1,
                                     gradient_threshold::Real = 0.0,
                                     mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                                     origin::Union{Nothing,Tuple{Real,Real}} = nothing)
    angular_granularity > 0 ||
        throw(ArgumentError("angular_granularity must be positive"))
    scale_granularity > 0 ||
        throw(ArgumentError("scale_granularity must be positive"))

    θ_min = Float64(rotation_range[1])
    θ_max = Float64(rotation_range[2])
    θ_max >= θ_min ||
        throw(ArgumentError("rotation_range[2] must be ≥ rotation_range[1]"))

    s_min = Float64(scale_range[1])
    s_max = Float64(scale_range[2])
    s_max >= s_min ||
        throw(ArgumentError("scale_range[2] must be ≥ scale_range[1]"))
    s_min > 0 ||
        throw(ArgumentError("scale must be strictly positive"))

    base_model = build_shape_model(image;
                                    gradient_threshold = gradient_threshold,
                                    mask = mask,
                                    origin = origin)

    θ_values = collect(θ_min:Float64(angular_granularity):θ_max)
    s_values = collect(s_min:Float64(scale_granularity):s_max)
    n_θ = length(θ_values)
    n_s = length(s_values)
    n_total = n_θ * n_s

    variants = Vector{ShapeVariant}(undef, n_total)
    idx = 0
    for θ in θ_values
        cosθ, sinθ = cos(θ), sin(θ)
        for s in s_values
            idx += 1
            A = [s*cosθ -s*sinθ; s*sinθ s*cosθ]
            variants[idx] = ShapeVariant(θ, s,
                                          transform_shape_model(base_model, A))
        end
    end

    return RotatedScaledSearchModel(
        base_model, variants,
        (θ_min, θ_max), Float64(angular_granularity), n_θ,
        (s_min, s_max), Float64(scale_granularity), n_s,
    )
end


"""
    nearest_variant(m::RotatedScaledSearchModel, θ::Real, scale::Real) -> ShapeVariant

Return the pre-computed variant whose `(θ, scale)` is closest to the
target on the (θ, scale) grid. Targets outside the grid are clamped to
the nearest grid point.
"""
function nearest_variant(m::RotatedScaledSearchModel, θ::Real, scale::Real)
    θ_idx = m.n_rotations <= 1 ? 1 :
            round(Int,
                  (Float64(θ) - m.rotation_range[1]) / m.angular_granularity) + 1
    s_idx = m.n_scales <= 1 ? 1 :
            round(Int,
                  (Float64(scale) - m.scale_range[1]) / m.scale_granularity) + 1

    θ_idx = clamp(θ_idx, 1, m.n_rotations)
    s_idx = clamp(s_idx, 1, m.n_scales)

    flat_idx = (θ_idx - 1) * m.n_scales + s_idx
    return m.variants[flat_idx]
end
