# Unit tests for gradient_matching.jl.
# This file is supposed to be included from runtests.jl.

@testset "Gradient Matching" begin

    # A diagonal-ramp source so that the Sobel gradient is non-zero and
    # smooth everywhere. Extracting a sub-region as the template guarantees
    # that interior gradients in source and template match exactly at the
    # extraction position (no boundary artifacts from embedding-into-zeros).
    h_src, w_src = 30, 30
    source = Float64[i + j for i in 1:h_src, j in 1:w_src]
    template = source[10:14, 15:19]   # 5×5 sub-region

    @testset "GDP — perfect match" begin
        scores = gdp_match(source, template)
        @test size(scores) == (h_src - 5 + 1, w_src - 5 + 1)
        @test isapprox(scores[10, 15], 1.0; atol = 1e-10)
        @test all(s -> -1e-12 <= s <= 1.0 + 1e-12, scores)
        # On a pure ramp the gradient direction is uniform — every interior position
        # gives GDP = 1; the position is therefore not uniquely identified by GDP alone.
        @test isapprox(maximum(scores), 1.0; atol = 1e-10)
    end

    @testset "GDP — score range and clamping" begin
        # GDP is in [0, 1]: contrast-inverted source clamps to 0 (no contrast reversal).
        source_inv = -source
        scores = gdp_match(source_inv, template)
        @test isapprox(scores[10, 15], 0.0; atol = 1e-10)
    end

    @testset "GDPR (global) — global contrast reversal" begin
        source_inv = -source
        gdp = gdp_match(source_inv, template)
        gdpr = gdpr_match(source_inv, template)
        # Global contrast reversal: |sum| / n = 1
        @test isapprox(gdpr[10, 15], 1.0; atol = 1e-10)
        # GDP at the same position is 0 (clamped)
        @test isapprox(gdp[10, 15], 0.0; atol = 1e-10)
        # Non-inverted source: GDPR also 1.0 at the perfect-match position
        @test isapprox(gdpr_match(source, template)[10, 15], 1.0; atol = 1e-10)
    end

    @testset "GDPR (lokal) — local contrast reversal" begin
        # For pure global inversion, GDPR-local also gives 1.0
        source_inv = -source
        gdpr_loc = gdpr_local_match(source_inv, template)
        @test isapprox(gdpr_loc[10, 15], 1.0; atol = 1e-10)
        # And for the un-inverted source
        @test isapprox(gdpr_local_match(source, template)[10, 15], 1.0; atol = 1e-10)
    end

    @testset "GDP — early abort returns 0 at clearly non-matching positions" begin
        # A source with the template embedded in noise; far away from the embedding,
        # gradient directions should not align consistently with the template,
        # so a high min_score should abort those positions.
        rng_src = Float64[(i + j) + 0.01 * sin(7 * i + 11 * j) for i in 1:h_src, j in 1:w_src]

        scores_full   = gdp_match(rng_src, template; min_score = 0.0)
        scores_abort  = gdp_match(rng_src, template; min_score = 0.95)

        # At the matching position both must be ≈ 1
        @test isapprox(scores_full[10, 15],  1.0; atol = 1e-3)
        @test isapprox(scores_abort[10, 15], 1.0; atol = 1e-3)

        # Aborted positions are reported as 0
        @test all(s -> s == 0.0 || s >= 0.95 - 1e-12, scores_abort)
    end

    @testset "Float32 input" begin
        src32  = Float32.(source)
        tmpl32 = Float32.(template)
        scores = gdp_match(src32, tmpl32)
        @test scores isa Matrix{Float64}
        @test isapprox(scores[10, 15], 1.0; atol = 1e-10)
    end

    @testset "Mask" begin
        # Mask out a corner of the template — should not affect the perfect match
        mask = trues(5, 5)
        mask[1, 1] = false
        scores = gdp_match(source, template; mask = mask)
        @test isapprox(scores[10, 15], 1.0; atol = 1e-10)
    end

    @testset "gradient_threshold filters out low-gradient pixels" begin
        # All Sobel-interior gradients in our ramp have the same magnitude.
        # Setting the threshold above that magnitude should leave no model points.
        # First find the actual gradient magnitude at the centre of the template.
        # For a unit-step ramp: gx = 4, gy = 4, so |grad| = sqrt(32) ≈ 5.66.
        scores = gdp_match(source, template; gradient_threshold = 1000.0)
        @test all(==(0.0), scores)
    end

    @testset "Multi-model GDP — per-model" begin
        # Three sub-regions of the same source as templates
        t1 = source[5:9,   5:9]
        t2 = source[15:19, 20:24]
        t3 = source[20:24, 5:9]
        score_imgs = gdp_match(source, [t1, t2, t3])
        @test length(score_imgs) == 3
        @test isapprox(score_imgs[1][5,   5], 1.0; atol = 1e-10)
        @test isapprox(score_imgs[2][15, 20], 1.0; atol = 1e-10)
        @test isapprox(score_imgs[3][20,  5], 1.0; atol = 1e-10)
    end

    @testset "Multi-model GDP — pixelwise" begin
        t1 = source[5:9,   5:9]
        t2 = source[15:19, 20:24]
        t3 = source[20:24, 5:9]
        max_scores, best_idx = gdp_match_pixelwise(source, [t1, t2, t3])
        # All three peaks should be ≈ 1 — but for a pure ramp the templates are
        # gradient-equivalent (same gradient direction everywhere), so all three
        # match at all three positions equally well. We verify the peak score and
        # that the index is in the valid range.
        @test isapprox(max_scores[5,   5],  1.0; atol = 1e-10)
        @test isapprox(max_scores[15, 20],  1.0; atol = 1e-10)
        @test isapprox(max_scores[20,  5],  1.0; atol = 1e-10)
        @test 1 <= best_idx[5,   5]  <= 3
        @test 1 <= best_idx[15, 20]  <= 3
        @test 1 <= best_idx[20,  5]  <= 3
    end

    @testset "Multi-model — distinct gradient patterns" begin
        # Build templates with truly distinct gradient orientations:
        # a horizontal-ramp, a vertical-ramp, and a diagonal-ramp source-region.
        h_ramp = Float64[j for i in 1:30, j in 1:30]
        v_ramp = Float64[i for i in 1:30, j in 1:30]
        d_ramp = Float64[i + j for i in 1:30, j in 1:30]

        # Compose a source: top-left quadrant horizontal, top-right vertical,
        # bottom-left diagonal. The templates are sub-regions of each ramp.
        src = zeros(40, 40)
        src[1:20, 1:20]   = h_ramp[1:20, 1:20]
        src[1:20, 21:40]  = v_ramp[1:20, 1:20]
        src[21:40, 1:20]  = d_ramp[1:20, 1:20]

        t_h = h_ramp[5:9,  5:9]
        t_v = v_ramp[5:9,  5:9]
        t_d = d_ramp[5:9,  5:9]

        max_scores, best_idx = gdp_match_pixelwise(src, [t_h, t_v, t_d])
        # In the horizontal-ramp region, t_h should win
        @test best_idx[7, 7] == 1
        # In the vertical-ramp region, t_v should win
        @test best_idx[7, 27] == 2
        # In the diagonal-ramp region, t_d should win
        @test best_idx[27, 7] == 3
    end

    @testset "Multi-model GDPR (global)" begin
        t1 = source[5:9,   5:9]
        score_imgs = gdpr_match(source, [t1])
        @test isapprox(score_imgs[1][5, 5], 1.0; atol = 1e-10)

        max_scores, best_idx = gdpr_match_pixelwise(source, [t1])
        @test isapprox(max_scores[5, 5], 1.0; atol = 1e-10)
        @test best_idx[5, 5] == 1
    end

    @testset "Multi-model GDPR (lokal)" begin
        t1 = source[5:9, 5:9]
        score_imgs = gdpr_local_match(source, [t1])
        @test isapprox(score_imgs[1][5, 5], 1.0; atol = 1e-10)

        max_scores, best_idx = gdpr_local_match_pixelwise(source, [t1])
        @test isapprox(max_scores[5, 5], 1.0; atol = 1e-10)
        @test best_idx[5, 5] == 1
    end

    @testset "Input validation" begin
        @test_throws ArgumentError gdp_match(rand(2, 2), rand(3, 3))
        @test_throws ArgumentError gdpr_match(rand(2, 2), rand(3, 3))
        @test_throws ArgumentError gdpr_local_match(rand(2, 2), rand(3, 3))
        @test_throws ArgumentError gdp_match(rand(10, 10), rand(3, 3); mask = trues(2, 2))
        @test_throws ArgumentError gdp_match(rand(10, 10), Matrix{Float64}[])
    end

end
