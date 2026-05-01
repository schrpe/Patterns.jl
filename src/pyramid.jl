#= ------------------------------------------------------------------------

    Bild-, Modell- und Maskenpyramiden

    Two pyramid types — one for image data (`ImagePyramid`) and one for
    binary masks (`MaskPyramid`) — share the same level convention:

      level 1 = the original input (finest resolution)
      level k > 1 = level k-1 downsampled by 2 in each dimension
      level length(p) = the coarsest level

    Image pyramids use a strict 2×2 mean filter (Mittelwert-Pyramide,
    PDF-Folie 448), which preserves pixel-aligned coordinates without the
    0.25-pixel offset of a Gauß-pyramide.

    Mask pyramids use conservative 2×2 erosion + downsampling: a pixel
    at level L is `true` iff all four source pixels at level L-1 are true.

    Pyramid height is determined by `min_size` — the pyramid keeps adding
    levels while the next downsampled level would still have its smallest
    dimension ≥ `min_size`. An explicit `n_levels` constructor is also
    provided. Odd-sized rows/columns at the boundary are truncated.

------------------------------------------------------------------------ =#

export ImagePyramid, MaskPyramid


# 2×2 mean downsampling. Drops odd-numbered last rows / columns.
function _mean_downsample(img::AbstractMatrix{Float64})
    h, w = size(img)
    new_h = h ÷ 2
    new_w = w ÷ 2
    out = Matrix{Float64}(undef, new_h, new_w)
    @inbounds for j in 1:new_w, i in 1:new_h
        i0 = 2 * i - 1
        j0 = 2 * j - 1
        out[i, j] = (img[i0,     j0] + img[i0 + 1, j0] +
                     img[i0, j0 + 1] + img[i0 + 1, j0 + 1]) / 4
    end
    return out
end

# Conservative 2×2 erosion + downsampling for binary masks: every output
# pixel is true iff all four source pixels are true.
function _erode_downsample(mask::BitMatrix)
    h, w = size(mask)
    new_h = h ÷ 2
    new_w = w ÷ 2
    out = falses(new_h, new_w)
    @inbounds for j in 1:new_w, i in 1:new_h
        i0 = 2 * i - 1
        j0 = 2 * j - 1
        out[i, j] = mask[i0,     j0] && mask[i0 + 1, j0] &&
                    mask[i0, j0 + 1] && mask[i0 + 1, j0 + 1]
    end
    return out
end


"""
    ImagePyramid(image::AbstractMatrix; min_size::Int = 8)
    ImagePyramid(image::AbstractMatrix, n_levels::Integer)

A 2×2 mean-filter pyramid. Level 1 is the original image (converted to
`Matrix{Float64}`); each subsequent level is the previous one halved in
both dimensions.

The keyword form keeps adding levels while the next downsampled level
would still have its minimum dimension ≥ `min_size`. The positional
form builds exactly `n_levels` levels.

Supports `length`, `getindex`, and iteration.

```jldoctest
julia> using Patterns

julia> img = Float64[i + j for i in 1:8, j in 1:8];

julia> p = ImagePyramid(img, 3);

julia> size.((p[1], p[2], p[3]))
((8, 8), (4, 4), (2, 2))

julia> p[2][1, 1]    # mean of img[1:2, 1:2] = (2+3+3+4)/4
3.0
```
"""
struct ImagePyramid
    levels::Vector{Matrix{Float64}}
end

function ImagePyramid(image::AbstractMatrix; min_size::Int = 8)
    min_size > 0 || throw(ArgumentError("min_size must be positive"))
    levels = Matrix{Float64}[Float64.(image)]
    while min(size(levels[end])...) ÷ 2 >= min_size
        push!(levels, _mean_downsample(levels[end]))
    end
    return ImagePyramid(levels)
end

function ImagePyramid(image::AbstractMatrix, n_levels::Integer)
    n_levels >= 1 || throw(ArgumentError("n_levels must be ≥ 1"))
    levels = Matrix{Float64}[Float64.(image)]
    for _ in 2:n_levels
        prev = levels[end]
        min(size(prev)...) >= 2 ||
            throw(ArgumentError(
                "cannot build $n_levels levels from $(size(image)) input"))
        push!(levels, _mean_downsample(prev))
    end
    return ImagePyramid(levels)
end

length(p::ImagePyramid) = length(p.levels)
Base.getindex(p::ImagePyramid, k::Integer) = p.levels[k]
Base.firstindex(p::ImagePyramid) = 1
Base.lastindex(p::ImagePyramid) = length(p.levels)
Base.iterate(p::ImagePyramid, state::Int = 1) =
    state > length(p) ? nothing : (p[state], state + 1)


"""
    MaskPyramid(mask::AbstractMatrix{Bool}; min_size::Int = 8)
    MaskPyramid(mask::AbstractMatrix{Bool}, n_levels::Integer)

A binary mask pyramid using conservative 2×2 erosion + downsampling: a
pixel at level L is `true` only if all four source pixels at level L-1
are true.

Same level convention and interface as `ImagePyramid`.

```jldoctest
julia> using Patterns

julia> mask = trues(8, 8); mask[1:2, 1:2] .= false;

julia> p = MaskPyramid(mask, 2);

julia> p[2][1, 1]    # all four of mask[1:2, 1:2] are false
false

julia> p[2][2, 2]    # mask[3:4, 3:4] are all true
true
```
"""
struct MaskPyramid
    levels::Vector{BitMatrix}
end

function MaskPyramid(mask::AbstractMatrix{Bool}; min_size::Int = 8)
    min_size > 0 || throw(ArgumentError("min_size must be positive"))
    levels = BitMatrix[BitMatrix(mask)]
    while min(size(levels[end])...) ÷ 2 >= min_size
        push!(levels, _erode_downsample(levels[end]))
    end
    return MaskPyramid(levels)
end

function MaskPyramid(mask::AbstractMatrix{Bool}, n_levels::Integer)
    n_levels >= 1 || throw(ArgumentError("n_levels must be ≥ 1"))
    levels = BitMatrix[BitMatrix(mask)]
    for _ in 2:n_levels
        prev = levels[end]
        min(size(prev)...) >= 2 ||
            throw(ArgumentError(
                "cannot build $n_levels levels from $(size(mask)) input"))
        push!(levels, _erode_downsample(prev))
    end
    return MaskPyramid(levels)
end

length(p::MaskPyramid) = length(p.levels)
Base.getindex(p::MaskPyramid, k::Integer) = p.levels[k]
Base.firstindex(p::MaskPyramid) = 1
Base.lastindex(p::MaskPyramid) = length(p.levels)
Base.iterate(p::MaskPyramid, state::Int = 1) =
    state > length(p) ? nothing : (p[state], state + 1)
