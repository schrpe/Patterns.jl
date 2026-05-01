# Unit tests for keypoint_match.jl.
# This file is supposed to be included from runtests.jl.

@testset "Keypoint Match" begin

    # Helper: build a PatternKeypoint with a specified bit pattern in the descriptor.
    function _kp(pos, bits::Integer; n_bits::Int = 64)
        bv = BitVector(undef, n_bits)
        for i in 1:n_bits
            bv[i] = (bits >> (i - 1)) & 1 == 1
        end
        return PatternKeypoint(pos, bv)
    end

    @testset "match_keypoints — identical sets all match" begin
        kps = [_kp((1.0, 1.0), 0x01),
               _kp((2.0, 2.0), 0x02),
               _kp((3.0, 3.0), 0x04)]
        matches = match_keypoints(kps, kps; max_distance = 30, ratio = 0.8)
        @test length(matches) == 3
        for m in matches
            @test m.distance == 0
            @test m.model_pos == m.image_pos
        end
    end

    @testset "match_keypoints — Lowe's ratio rejects ambiguous" begin
        # Both image keypoints are equally close to the template — ratio test rejects
        template = [_kp((1.0, 1.0), 0x00)]
        image_ambig = [_kp((10.0, 10.0), 0x00),
                       _kp((20.0, 20.0), 0x00)]   # both identical to template
        matches = match_keypoints(template, image_ambig;
                                   max_distance = 10, ratio = 0.8)
        @test isempty(matches)

        # Template is much closer to one image keypoint — accepted
        image_clear = [_kp((10.0, 10.0), 0x00),     # exact match
                       _kp((20.0, 20.0), 0xff)]    # 8 bits different
        matches2 = match_keypoints(template, image_clear;
                                    max_distance = 30, ratio = 0.8)
        @test length(matches2) == 1
        @test matches2[1].image_pos == (10.0, 10.0)
        @test matches2[1].distance == 0
    end

    @testset "match_keypoints — max_distance filters" begin
        template = [_kp((1.0, 1.0), 0x00)]
        # All bits different in 64-bit descriptor → distance = 64
        image = [_kp((10.0, 10.0), 0xffffffffffffffff)]
        matches = match_keypoints(template, image;
                                   max_distance = 30, ratio = 0.8)
        @test isempty(matches)
    end

    @testset "match_keypoints — descriptor length mismatch" begin
        kp64 = _kp((1.0, 1.0), 0x01; n_bits = 64)
        kp32 = _kp((2.0, 2.0), 0x01; n_bits = 32)
        @test_throws DimensionMismatch match_keypoints([kp64], [kp32])
    end

    @testset "match_keypoints — single image keypoint uses max_distance fallback" begin
        # With only one image keypoint, second_best stays at max_distance.
        # Acceptance: best < max_distance * ratio.
        template = [_kp((1.0, 1.0), 0x00)]
        image = [_kp((10.0, 10.0), 0x00)]   # exact match, dist = 0
        matches = match_keypoints(template, image;
                                   max_distance = 10, ratio = 0.8)
        @test length(matches) == 1
        @test matches[1].distance == 0
    end

    @testset "match_keypoints — input validation" begin
        kp = _kp((1.0, 1.0), 0x01)
        @test_throws ArgumentError match_keypoints([kp], [kp]; ratio = 0.0)
        @test_throws ArgumentError match_keypoints([kp], [kp]; ratio = 1.5)
        @test_throws ArgumentError match_keypoints([kp], [kp]; max_distance = -1)
    end

    @testset "match_keypoints — multi-template" begin
        kp_a = _kp((1.0, 1.0), 0x01)
        kp_b = _kp((2.0, 2.0), 0x02)
        kp_c = _kp((3.0, 3.0), 0x04)

        templates = [[kp_a, kp_b], [kp_c]]
        image = [_kp((10.0, 10.0), 0x01),
                 _kp((11.0, 11.0), 0x02),
                 _kp((20.0, 20.0), 0x04)]

        results = match_keypoints(templates, image;
                                   max_distance = 30, ratio = 0.8)
        @test length(results) == 2
        @test length(results[1]) == 2
        @test length(results[2]) == 1
        @test results[1][1].image_pos == (10.0, 10.0)
        @test results[1][2].image_pos == (11.0, 11.0)
        @test results[2][1].image_pos == (20.0, 20.0)
    end

    @testset "match_keypoints — multi-template empty input" begin
        kp = _kp((1.0, 1.0), 0x01)
        @test_throws ArgumentError match_keypoints(Vector{PatternKeypoint}[],
                                                    [kp])
    end

    @testset "match_all_keypoints — no ratio test" begin
        template = [_kp((1.0, 1.0), 0x00)]
        # Three image keypoints at varying distances
        image = [_kp((10.0, 10.0), 0x00),     # dist 0
                 _kp((20.0, 20.0), 0x01),     # dist 1
                 _kp((30.0, 30.0), 0xff)]     # dist 8 (low byte all set)
        matches = match_all_keypoints(template, image; max_distance = 5)
        # Two matches under the distance bound
        @test length(matches) == 2
    end

    @testset "match_all_keypoints — collects all qualifying pairs" begin
        # Two templates, two image keypoints, all under the bound:
        # 4 matches expected.
        templates = [_kp((0.0, 0.0), 0x00), _kp((1.0, 1.0), 0x00)]
        images    = [_kp((10.0, 10.0), 0x00), _kp((11.0, 11.0), 0x01)]
        matches = match_all_keypoints(templates, images; max_distance = 5)
        @test length(matches) == 4
    end

    @testset "match_all_keypoints — input validation" begin
        kp = _kp((1.0, 1.0), 0x01)
        @test_throws ArgumentError match_all_keypoints([kp], [kp]; max_distance = -1)
    end

    @testset "End-to-end — ORB descriptors on a non-repetitive image" begin
        # A non-repetitive image gives distinct descriptors so Lowe's
        # ratio test can distinguish them. Note: a checkerboard produces
        # many visually-identical patches → identical descriptors, which
        # the ratio test correctly rejects as ambiguous.
        # Pseudo-random pattern via deterministic trigonometric mix:
        img = [0.5 + 0.5 * sin(0.13 * i + 0.27 * j) * cos(0.31 * i - 0.19 * j)
               for i in 1:120, j in 1:120]
        kp = detect_orb(img; n_keypoints = 30, threshold = 0.05)
        @test !isempty(kp)

        # Match against itself: distinct descriptors should pass the
        # ratio test, giving a self-match (distance 0) for most kps.
        matches = match_keypoints(kp, kp; max_distance = 80, ratio = 0.8)
        @test length(matches) >= 1
        @test any(m -> m.distance == 0, matches)
    end

end
