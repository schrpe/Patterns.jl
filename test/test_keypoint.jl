# Unit tests for keypoint.jl.
# This file is supposed to be included from runtests.jl.

@testset "Keypoint" begin

    # A reusable checkerboard with strong corner-like features
    function _checkerboard(n::Int = 64, block::Int = 8)
        img = zeros(n, n)
        for i in 1:n, j in 1:n
            img[i, j] = ((i ÷ block) + (j ÷ block)) % 2 == 0 ? 1.0 : 0.0
        end
        return img
    end

    @testset "PatternKeypoint construction and equality" begin
        d = trues(256)
        kp1 = PatternKeypoint((10.0, 20.0), d)
        kp2 = PatternKeypoint((10.0, 20.0), d)
        @test kp1 == kp2
        @test kp1.position == (10.0, 20.0)
        @test length(kp1.descriptor) == 256

        kp3 = PatternKeypoint((10.0, 20.0), falses(256))
        @test kp1 != kp3
    end

    @testset "detect_fast — checkerboard" begin
        img = _checkerboard(64, 8)
        keypoints = detect_fast(img; threshold = 0.15, n = 12)
        @test !isempty(keypoints)
        # Checkerboard corners should be at the block intersections
        @test keypoints isa Vector{Tuple{Float64,Float64}}
    end

    @testset "detect_fast — threshold sensitivity" begin
        img = _checkerboard(64, 8)
        many = detect_fast(img; threshold = 0.05, n = 12)
        few  = detect_fast(img; threshold = 0.30, n = 12)
        # Lower threshold detects more corners
        @test length(many) >= length(few)
    end

    @testset "detect_fast — accepts Float32 input" begin
        img32 = Float32.(_checkerboard(64, 8))
        keypoints = detect_fast(img32; threshold = 0.15, n = 12)
        @test !isempty(keypoints)
    end

    @testset "detect_fast_scale_invariant — produces more keypoints" begin
        img = _checkerboard(80, 10)
        single_scale = detect_fast(img; threshold = 0.15, n = 12)
        multi_scale = detect_fast_scale_invariant(img;
                                                    threshold = 0.15, n = 12,
                                                    downscale_factor = 0.8,
                                                    upscale_factor = 1.25,
                                                    upscale_iters = 2,
                                                    min_dim = 16)
        @test length(multi_scale) >= length(single_scale)
        # Positions are still in the original frame: row/col within image bounds
        for (r, c) in multi_scale
            @test 0.0 <= r <= 80 + 1
            @test 0.0 <= c <= 80 + 1
        end
    end

    @testset "detect_fast_scale_invariant — input validation" begin
        img = _checkerboard(64, 8)
        @test_throws ArgumentError detect_fast_scale_invariant(img;
                                            downscale_factor = 1.0)
        @test_throws ArgumentError detect_fast_scale_invariant(img;
                                            downscale_factor = 0.0)
        @test_throws ArgumentError detect_fast_scale_invariant(img;
                                            upscale_factor = 1.0)
        @test_throws ArgumentError detect_fast_scale_invariant(img;
                                            upscale_iters = -1)
        @test_throws ArgumentError detect_fast_scale_invariant(img;
                                            min_dim = 5)
    end

    @testset "detect_orb — returns PatternKeypoint with 256-bit descriptors" begin
        img = _checkerboard(64, 8)
        keypoints = detect_orb(img; n_keypoints = 30, threshold = 0.15)
        @test keypoints isa Vector{PatternKeypoint}
        @test !isempty(keypoints)
        for kp in keypoints
            @test length(kp.descriptor) == 256
            @test kp.position[1] >= 1.0 && kp.position[1] <= 64.0
            @test kp.position[2] >= 1.0 && kp.position[2] <= 64.0
        end
    end

    @testset "detect_orb — limits keypoint count" begin
        img = _checkerboard(80, 10)
        # Set a tight cap; ORB should respect it
        kp_capped = detect_orb(img; n_keypoints = 10, threshold = 0.15)
        @test length(kp_capped) <= 10
    end

    @testset "detect_orb — accepts Int matrix" begin
        # Int matrix should be auto-converted via Float64
        img_int = [Int((i + j) % 2) for i in 1:64, j in 1:64]
        kps = detect_orb(img_int; n_keypoints = 20, threshold = 0.1)
        @test kps isa Vector{PatternKeypoint}
    end

    @testset "Two images, distinct keypoints" begin
        # Two checkerboards with the same block size but spatially shifted
        # produce different keypoint sets with high likelihood of differing
        # first-descriptor.
        img1 = _checkerboard(64, 4)
        # Shift by half a block — different feature positions
        img2 = circshift(img1, (2, 2))
        kp1 = detect_orb(img1; n_keypoints = 30, threshold = 0.15)
        kp2 = detect_orb(img2; n_keypoints = 30, threshold = 0.15)
        @test !isempty(kp1)
        @test !isempty(kp2)
        # The shift should produce keypoints at shifted positions
        if !isempty(kp1) && !isempty(kp2)
            # At least one keypoint should differ in either position or descriptor
            @test any(kp1[k].position != kp2[k].position
                      for k in 1:min(length(kp1), length(kp2)))
        end
    end

end
