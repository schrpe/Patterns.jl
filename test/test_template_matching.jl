# Unit tests for template_matching.jl.
# This file is supposed to be included from runtests.jl.

@testset "Template Matching" begin

    # A reusable 3×3 template
    template = [1.0 2.0 3.0;
                4.0 5.0 6.0;
                7.0 8.0 9.0]

    # Source 20×20 with the template embedded at row 5, column 8
    source = zeros(20, 20)
    source[5:7, 8:10] .= template

    @testset "SAD — single template" begin
        scores = sad_match(source, template)
        @test size(scores) == (18, 18)
        @test isapprox(scores[5, 8], 0.0; atol = 1e-12)

        # Trough is unique at the embedding position
        min_pos = argmin(scores)
        @test Tuple(min_pos) == (5, 8)
    end

    @testset "SAD — early abort" begin
        scores = sad_match(source, template; max_score = 1.0)
        @test isapprox(scores[5, 8], 0.0; atol = 1e-12)

        # Distant positions should be aborted (zero source under the template
        # gives mean(|0 - template|) = mean(template) = 5.0 ≫ 1.0)
        @test scores[1, 1] == Inf
        @test scores[18, 18] == Inf
    end

    @testset "SAD — Float32 input" begin
        src32  = Float32.(source)
        tmpl32 = Float32.(template)
        scores = sad_match(src32, tmpl32)
        @test scores isa Matrix{Float64}
        @test isapprox(scores[5, 8], 0.0; atol = 1e-12)
    end

    @testset "SAD — mask" begin
        # Mask out the centre pixel of the template; perfect match should still hold
        mask = trues(3, 3)
        mask[2, 2] = false
        scores = sad_match(source, template; mask = mask)
        @test isapprox(scores[5, 8], 0.0; atol = 1e-12)
    end

    @testset "NCC — invariant to linear contrast change" begin
        # source = 3·template + 1 (within the embedding window) on a constant background
        source2 = fill(2.0, 20, 20)
        source2[5:7, 8:10] .= 3.0 .* template .+ 1.0

        scores = ncc_match(source2, template)
        @test size(scores) == (18, 18)
        @test isapprox(scores[5, 8], 1.0; atol = 1e-10)
    end

    @testset "NCC — perfect match against itself" begin
        scores = ncc_match(source, template)
        @test isapprox(scores[5, 8], 1.0; atol = 1e-10)
        @test maximum(scores) ≈ 1.0
        # Range constraint
        @test all(s -> -1.0 - 1e-12 <= s <= 1.0 + 1e-12, scores)
    end

    @testset "NCC — zero variance source returns 0" begin
        # Constant source: source variance is zero everywhere → NCC=0 everywhere
        const_src = fill(7.0, 10, 10)
        scores = ncc_match(const_src, template)
        @test all(s -> s == 0.0, scores)
    end

    @testset "NCC — mask" begin
        mask = trues(3, 3)
        mask[1, 1] = false  # ignore corner
        source3 = fill(2.0, 20, 20)
        source3[5:7, 8:10] .= template
        scores = ncc_match(source3, template; mask = mask)
        @test isapprox(scores[5, 8], 1.0; atol = 1e-10)
    end

    @testset "NCCR — perfect match for inverted contrast" begin
        # source-region = -1 · template + offset → NCC = -1, NCCR = 1
        source_inv = fill(2.0, 20, 20)
        source_inv[5:7, 8:10] .= -1.0 .* template .+ 12.0

        ncc = ncc_match(source_inv, template)
        nccr = nccr_match(source_inv, template)

        @test isapprox(ncc[5, 8], -1.0; atol = 1e-10)
        @test isapprox(nccr[5, 8],  1.0; atol = 1e-10)
    end

    # Three 2×2 templates with truly distinct mean-centered shapes (so NCC distinguishes them)
    distinct_t1 = [1.0 2.0; 3.0 4.0]
    distinct_t2 = [4.0 1.0; 2.0 3.0]
    distinct_t3 = [9.0 1.0; 1.0 9.0]

    @testset "Multi-model — per-model" begin
        src = zeros(30, 30)
        src[3:4,   3:4]   .= distinct_t1
        src[10:11, 15:16] .= distinct_t2
        src[20:21, 25:26] .= distinct_t3

        score_imgs = ncc_match(src, [distinct_t1, distinct_t2, distinct_t3])
        @test length(score_imgs) == 3
        @test isapprox(score_imgs[1][3,   3],  1.0; atol = 1e-10)
        @test isapprox(score_imgs[2][10, 15],  1.0; atol = 1e-10)
        @test isapprox(score_imgs[3][20, 25],  1.0; atol = 1e-10)
        # The templates' mean-centered shapes differ, so cross-NCC is < 1
        @test score_imgs[2][3, 3] < 0.5
        @test score_imgs[3][10, 15] < 0.999
    end

    @testset "Multi-model — pixelwise NCC" begin
        src = zeros(30, 30)
        src[3:4,   3:4]   .= distinct_t1
        src[10:11, 15:16] .= distinct_t2
        src[20:21, 25:26] .= distinct_t3

        max_scores, best_idx = ncc_match_pixelwise(src, [distinct_t1, distinct_t2, distinct_t3])

        @test size(max_scores) == size(best_idx)
        @test isapprox(max_scores[3,   3],  1.0; atol = 1e-10)
        @test isapprox(max_scores[10, 15],  1.0; atol = 1e-10)
        @test isapprox(max_scores[20, 25],  1.0; atol = 1e-10)
        @test best_idx[3,   3]  == 1
        @test best_idx[10, 15]  == 2
        @test best_idx[20, 25]  == 3
    end

    @testset "Multi-model — pixelwise SAD" begin
        src = zeros(30, 30)
        src[3:4,   3:4]   .= distinct_t1
        src[10:11, 15:16] .= distinct_t2
        src[20:21, 25:26] .= distinct_t3

        min_scores, best_idx = sad_match_pixelwise(src, [distinct_t1, distinct_t2, distinct_t3])

        @test isapprox(min_scores[3,   3], 0.0; atol = 1e-12)
        @test isapprox(min_scores[10, 15], 0.0; atol = 1e-12)
        @test isapprox(min_scores[20, 25], 0.0; atol = 1e-12)
        @test best_idx[3,   3]  == 1
        @test best_idx[10, 15]  == 2
        @test best_idx[20, 25]  == 3
    end

    @testset "Multi-model — pixelwise NCCR detects inverted contrast" begin
        t1 = [1.0 2.0; 3.0 4.0]

        # Source has the contrast-inverted template embedded — NCC would be -1, but NCCR = 1
        src = fill(2.0, 20, 20)
        src[8:9, 8:9] .= -1.0 .* t1 .+ 12.0

        max_scores, best_idx = nccr_match_pixelwise(src, [t1])
        @test isapprox(max_scores[8, 8], 1.0; atol = 1e-10)
        @test best_idx[8, 8] == 1
    end

    @testset "Input validation" begin
        # Source smaller than model
        @test_throws ArgumentError sad_match(rand(2, 2), rand(3, 3))
        @test_throws ArgumentError ncc_match(rand(2, 2), rand(3, 3))

        # Mask wrong size
        @test_throws ArgumentError sad_match(rand(10, 10), rand(3, 3); mask = trues(2, 2))
        @test_throws ArgumentError ncc_match(rand(10, 10), rand(3, 3); mask = trues(2, 2))

        # All-false mask
        @test_throws ArgumentError sad_match(rand(10, 10), rand(3, 3); mask = falses(3, 3))

        # Empty model list
        @test_throws ArgumentError ncc_match(rand(10, 10), Matrix{Float64}[])

        # Models of differing sizes
        @test_throws ArgumentError ncc_match(rand(10, 10), [rand(3, 3), rand(2, 2)])

        # Masks length mismatch
        @test_throws ArgumentError ncc_match(rand(10, 10),
                                              [rand(3, 3), rand(3, 3)];
                                              masks = [trues(3, 3)])
    end

end
