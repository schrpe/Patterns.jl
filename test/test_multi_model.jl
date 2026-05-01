# Unit tests for multi_model.jl.
# This file is supposed to be included from runtests.jl.

@testset "Multi-Model" begin

    @testset "search_templates — two distinct templates" begin
        # 64×64 ramp source with two extracted sub-region templates
        h_src, w_src = 64, 64
        src = Float64[i + j for i in 1:h_src, j in 1:w_src]
        t1 = src[10:18, 15:23]   # 9×9 sub-region
        t2 = src[40:48, 50:58]   # 9×9 sub-region

        results = search_templates(src, [t1, t2];
                                    metric = :ncc, min_score = 0.95,
                                    min_size = 4)
        @test length(results) == 2
        @test !isempty(results[1])
        @test !isempty(results[2])
        # Highest score is around 1.0 (template extracted from same source)
        @test maximum(m.score for m in results[1]) > 0.99
        @test maximum(m.score for m in results[2]) > 0.99
    end

    @testset "search_templates — distinct templates with NCCR" begin
        # Source contains a template and its contrast-inverted version
        h_src, w_src = 60, 60
        h_t = 9
        w_t = 9
        template = [Float64((i - 5)^2 + (j - 5)^2) for i in 1:h_t, j in 1:w_t]
        src = fill(0.5, h_src, w_src)
        src[10:18, 10:18] .= template
        src[30:38, 30:38] .= maximum(template) .- template   # inverted

        results = search_templates(src, [template, template];
                                    metric = :nccr, min_score = 0.7,
                                    min_size = 4)
        # Both lookups should find both occurrences (NCCR is contrast-symmetric)
        @test length(results) == 2
        @test !isempty(results[1])
        @test !isempty(results[2])
    end

    @testset "search_templates — masks per template" begin
        h_src, w_src = 50, 50
        src = zeros(h_src, w_src)
        h_t = 7
        template = [Float64((i - 4)^2 + (j - 4)^2) for i in 1:h_t, j in 1:h_t]
        src[10:16, 10:16] .= template
        src[30:36, 30:36] .= template

        m1 = trues(h_t, h_t); m1[1, 1] = false
        m2 = trues(h_t, h_t); m2[h_t, h_t] = false

        results = search_templates(src, [template, template];
                                    metric = :ncc, masks = [m1, m2],
                                    min_score = 0.95, min_size = 4)
        @test length(results) == 2
        @test !isempty(results[1])
        @test !isempty(results[2])
    end

    @testset "search_templates — empty templates throws" begin
        @test_throws ArgumentError search_templates(rand(20, 20),
                                                     Matrix{Float64}[])
    end

    @testset "search_templates — masks length mismatch throws" begin
        @test_throws ArgumentError search_templates(
            rand(20, 20), [rand(3, 3), rand(3, 3)];
            masks = [trues(3, 3)])
    end

    @testset "search_templates_pixelwise — NCC" begin
        # Three 2×2 templates with truly distinct mean-centered shapes
        t1 = [1.0 2.0; 3.0 4.0]
        t2 = [4.0 1.0; 2.0 3.0]
        t3 = [9.0 1.0; 1.0 9.0]

        src = zeros(30, 30)
        src[3:4,   3:4]   .= t1
        src[10:11, 15:16] .= t2
        src[20:21, 25:26] .= t3

        max_scores, best_idx = search_templates_pixelwise(
            src, [t1, t2, t3]; metric = :ncc)

        @test isapprox(max_scores[3, 3],   1.0; atol = 1e-10)
        @test isapprox(max_scores[10, 15], 1.0; atol = 1e-10)
        @test isapprox(max_scores[20, 25], 1.0; atol = 1e-10)
        @test best_idx[3, 3]   == 1
        @test best_idx[10, 15] == 2
        @test best_idx[20, 25] == 3
    end

    @testset "search_templates_pixelwise — SAD" begin
        t1 = [1.0 2.0; 3.0 4.0]
        t2 = [5.0 6.0; 7.0 8.0]
        src = zeros(20, 20)
        src[5:6, 5:6] .= t1
        src[10:11, 15:16] .= t2

        min_scores, best_idx = search_templates_pixelwise(
            src, [t1, t2]; metric = :sad)
        @test isapprox(min_scores[5, 5],   0.0; atol = 1e-12)
        @test isapprox(min_scores[10, 15], 0.0; atol = 1e-12)
        @test best_idx[5, 5]   == 1
        @test best_idx[10, 15] == 2
    end

    @testset "search_templates_pixelwise — GDP" begin
        # Use distinct gradient orientations
        h_ramp = Float64[j for i in 1:30, j in 1:30]
        v_ramp = Float64[i for i in 1:30, j in 1:30]

        src = zeros(40, 40)
        src[1:20, 1:20]   = h_ramp[1:20, 1:20]
        src[1:20, 21:40]  = v_ramp[1:20, 1:20]

        t_h = h_ramp[5:9, 5:9]
        t_v = v_ramp[5:9, 5:9]

        max_scores, best_idx = search_templates_pixelwise(
            src, [t_h, t_v]; metric = :gdp)

        # In horizontal-ramp region (cols 1-20) the horizontal template wins
        @test best_idx[10, 10] == 1
        # In vertical-ramp region (cols 21-40) the vertical template wins
        @test best_idx[10, 30] == 2
    end

    @testset "search_templates_pixelwise — empty templates throws" begin
        @test_throws ArgumentError search_templates_pixelwise(
            rand(20, 20), Matrix{Float64}[])
    end

    @testset "search_templates_pixelwise — unknown metric throws" begin
        @test_throws ArgumentError search_templates_pixelwise(
            rand(20, 20), [rand(3, 3)]; metric = :bogus)
    end

    @testset "search_shape_models — multiple models on one source" begin
        # Parabolic bowl is rotation- and scale-symmetric. Build two
        # models with different rotation grids; both should match
        # at the source centre.
        h, w = 41, 41
        center = ((h + 1) / 2, (w + 1) / 2)
        img = [(i - center[1])^2 + (j - center[2])^2 for i in 1:h, j in 1:w]
        mask = [sqrt((i - center[1])^2 + (j - center[2])^2) <= 8.0
                for i in 1:h, j in 1:w]

        m_no_rot = create_similarity_template(img;
                                               gradient_threshold = 1.0,
                                               origin = center, mask = mask)
        m_rot = create_similarity_template(img;
                                            rotation_range = (0.0, π/2),
                                            angular_granularity = π/4,
                                            gradient_threshold = 1.0,
                                            origin = center, mask = mask)

        results = search_shape_models(img, [m_no_rot, m_rot];
                                       variant = :gdp,
                                       min_score = 0.95,
                                       suppression_radius = 5)
        @test length(results) == 2
        @test !isempty(results[1])
        @test !isempty(results[2])
        # Each model's best match should be at the source centre
        for r in results
            sorted = sort(r, by = m -> -m.score)
            @test abs(sorted[1].pose.r - center[1]) <= 1.0
            @test abs(sorted[1].pose.c - center[2]) <= 1.0
            @test sorted[1].score > 0.99
        end
    end

    @testset "search_shape_models — empty models throws" begin
        @test_throws ArgumentError search_shape_models(
            rand(10, 10), RotatedScaledSearchModel[])
    end

end
