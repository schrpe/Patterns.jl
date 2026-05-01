#= ------------------------------------------------------------------------

    Keypoint-Detektion und -Beschreibung

    Thin wrappers around `ImageFeatures.fastcorners` and `ImageFeatures.ORB`
    plus a scale-invariant FAST detection routine that runs detection
    across down- and upscaled copies of the image.

    The `PatternKeypoint` type bundles a 2D position (in `(row, col)`)
    with a binary descriptor. Pure FAST detection (no descriptors) returns
    plain `(row, col)` tuples.

    For Gray conversion at the ImageFeatures boundary we depend on
    `ColorTypes.Gray`. The detection input may be any `AbstractMatrix`
    that broadcasts to `Float64` (Gray, Float, Int).

------------------------------------------------------------------------ =#

import ImageFeatures
import ImageTransformations
import ColorTypes

export PatternKeypoint
export detect_fast, detect_fast_scale_invariant, detect_orb


# Convert any image to a Gray{Float64} matrix (the form ImageFeatures expects).
function _to_gray(image::AbstractMatrix)
    return ColorTypes.Gray.(Float64.(image))
end


"""
    PatternKeypoint(position::Tuple{Float64,Float64}, descriptor::BitVector)

A keypoint with a `(row, col)` position and a binary descriptor.
"""
struct PatternKeypoint
    position::Tuple{Float64,Float64}
    descriptor::BitVector
end

==(a::PatternKeypoint, b::PatternKeypoint) =
    a.position == b.position && a.descriptor == b.descriptor


"""
    detect_fast(image::AbstractMatrix; threshold::Real = 0.15, n::Int = 12)
        -> Vector{Tuple{Float64,Float64}}

Detect FAST corners. A pixel is a corner if at least `n` of the 16 pixels
on a circle around it are uniformly brighter or darker than the centre by
more than `threshold`. Returns positions as `(row, col)` tuples.
"""
function detect_fast(image::AbstractMatrix; threshold::Real = 0.15, n::Int = 12)
    img = _to_gray(image)
    corners = ImageFeatures.fastcorners(img, n, Float64(threshold))
    return [(Float64(idx[1]), Float64(idx[2]))
            for idx in ImageFeatures.Keypoints(corners)]
end


"""
    detect_fast_scale_invariant(image::AbstractMatrix;
                                threshold::Real = 0.15,
                                n::Int = 12,
                                downscale_factor::Real = 0.8,
                                upscale_factor::Real = 1.25,
                                upscale_iters::Int = 3,
                                min_dim::Int = 16)
        -> Vector{Tuple{Float64,Float64}}

Detect FAST corners across multiple scales (down- and upscaling loops).
Positions of corners detected on rescaled images are back-transformed to
the original coordinate frame.

The downscaling loop multiplies the scale factor by `downscale_factor`
each iteration and stops when the smallest image dimension would fall
below `min_dim`. The upscaling loop multiplies by `upscale_factor` for
`upscale_iters` iterations.

Duplicate positions across scales are kept (no deduplication); downstream
matching / RANSAC filters redundant detections.
"""
function detect_fast_scale_invariant(image::AbstractMatrix;
                                       threshold::Real = 0.15,
                                       n::Int = 12,
                                       downscale_factor::Real = 0.8,
                                       upscale_factor::Real = 1.25,
                                       upscale_iters::Int = 3,
                                       min_dim::Int = 16)
    0 < downscale_factor < 1 ||
        throw(ArgumentError("downscale_factor must be in (0, 1)"))
    upscale_factor > 1 ||
        throw(ArgumentError("upscale_factor must be > 1"))
    upscale_iters >= 0 ||
        throw(ArgumentError("upscale_iters must be ≥ 0"))
    min_dim >= 7 ||
        throw(ArgumentError("min_dim must be ≥ 7 (FAST needs a 7×7 window)"))

    img = _to_gray(image)
    h, w = size(img)
    h_f = Float64(h)
    w_f = Float64(w)

    keypoints = detect_fast(img; threshold = threshold, n = n)

    # Helper: keep only back-transformed positions that lie inside the original image.
    function _add_scaled!(keypoints, scaled_kps, scale)
        for (r, c) in scaled_kps
            r_back = r / scale
            c_back = c / scale
            if 1.0 <= r_back <= h_f && 1.0 <= c_back <= w_f
                push!(keypoints, (r_back, c_back))
            end
        end
    end

    # Downscale loop
    scale = Float64(downscale_factor)
    while min(h * scale, w * scale) >= min_dim
        scaled = ImageTransformations.imresize(img, ratio = scale)
        _add_scaled!(keypoints,
                     detect_fast(scaled; threshold = threshold, n = n),
                     scale)
        scale *= downscale_factor
    end

    # Upscale loop
    scale = Float64(upscale_factor)
    for _ in 1:upscale_iters
        scaled = ImageTransformations.imresize(img, ratio = scale)
        _add_scaled!(keypoints,
                     detect_fast(scaled; threshold = threshold, n = n),
                     scale)
        scale *= upscale_factor
    end

    return keypoints
end


"""
    detect_orb(image::AbstractMatrix;
               n_keypoints::Int = 500,
               n_fast::Int = 12,
               threshold::Real = 0.25,
               harris_factor::Real = 0.04,
               downsample::Real = 1.3,
               levels::Int = 8,
               sigma::Real = 1.2)
        -> Vector{PatternKeypoint}

Detect oriented FAST keypoints with rotated BRIEF (ORB) descriptors. The
underlying `ImageFeatures.ORB` builds an internal pyramid (`levels`
levels, downsampling factor `downsample`) for scale-invariant detection
and computes a 256-bit descriptor per keypoint.
"""
function detect_orb(image::AbstractMatrix;
                    n_keypoints::Int = 500,
                    n_fast::Int = 12,
                    threshold::Real = 0.25,
                    harris_factor::Real = 0.04,
                    downsample::Real = 1.3,
                    levels::Int = 8,
                    sigma::Real = 1.2)
    img = _to_gray(image)
    orb = ImageFeatures.ORB(num_keypoints = n_keypoints,
                              n_fast = n_fast,
                              threshold = Float64(threshold),
                              harris_factor = Float64(harris_factor),
                              downsample = Float64(downsample),
                              levels = levels,
                              sigma = Float64(sigma))
    desc, kp = ImageFeatures.create_descriptor(img, orb)
    return [PatternKeypoint((Float64(k[1]), Float64(k[2])), d)
            for (k, d) in zip(kp, desc)]
end
