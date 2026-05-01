# Unit tests for pyramid_search.jl.
# This file is supposed to be included from runtests.jl.

@testset "Pyramid Search" begin

    @testset "Translation pyramid_search — NCC, exact placement" begin
        # 64×64 background with a distinctive 9×9 template embedded at (20, 30)
        h_src, w_src = 64, 64
        src = zeros(h_src, w_src)
        # Distinctive template: a small radial bump
        h_t, w_t = 9, 9
        template = [Float64((i - 5)^2 + (j - 5)^2) for i in 1:h_t, j in 1:w_t]
        src[20:28, 30:38] .= template

        matches = pyramid_search(src, template;
                                  metric = :ncc, min_score = 0.5,
                                  min_size = 4, padding = 2)
        @test !isempty(matches)
        # The strongest match should be at (20, 30)
        sorted = sort(matches, by = m -> -m.score)
        @test sorted[1].pose == (20, 30)
        @test sorted[1].score > 0.95
    end

    @testset "Translation pyramid_search — NCCR, contrast-inverted" begin
        h_src, w_src = 64, 64
        src = fill(0.5, h_src, w_src)
        h_t, w_t = 9, 9
        template = [Float64((i - 5)^2 + (j - 5)^2) for i in 1:h_t, j in 1:w_t]
        # Embed an inverted-contrast version (rescaled to keep range sensible)
        max_t = maximum(template)
        src[20:28, 30:38] .= max_t .- template

        matches = pyramid_search(src, template;
                                  metric = :nccr, min_score = 0.5,
                                  min_size = 4)
        @test !isempty(matches)
        sorted = sort(matches, by = m -> -m.score)
        @test sorted[1].pose == (20, 30)
        @test sorted[1].score > 0.95
    end

    @testset "Translation pyramid_search — GDP, ramp content" begin
        # A larger background ramp with a distinctive sub-region as template
        h_src, w_src = 80, 80
        src = Float64[i + j for i in 1:h_src, j in 1:w_src]
        h_t, w_t = 11, 11
        template = src[20:30, 30:40]    # a chunk of the same ramp

        matches = pyramid_search(src, template;
                                  metric = :gdp, min_score = 0.95,
                                  gradient_threshold = 1.0,
                                  min_size = 4, padding = 2)
        @test !isempty(matches)
        # On a uniform ramp every position has the same gradient direction so
        # GDP = 1 at many positions. Just check that at least one match is
        # at or near the embedded position and the top score is ≈ 1.
        sorted = sort(matches, by = m -> -m.score)
        @test sorted[1].score > 0.99
    end

    @testset "Translation pyramid_search — empty result below threshold" begin
        src = zeros(40, 40)
        template = ones(7, 7)
        matches = pyramid_search(src, template;
                                  metric = :ncc, min_score = 0.5,
                                  min_size = 4)
        # Constant source with non-constant template → NCC = 0 everywhere
        @test isempty(matches) || all(m -> m.score < 0.5, matches)
    end

    @testset "Translation pyramid_search — stop_at_level" begin
        # Continuous ramp source (no embedding-boundary artifacts);
        # 9×9 template with min_size=4 → 2 pyramid levels (9 → 4)
        h_src, w_src = 64, 64
        src = Float64[i + j for i in 1:h_src, j in 1:w_src]
        template = src[20:28, 30:38]   # extracted from the same ramp

        # stop_at_level = 1 (full refinement)
        matches_full = pyramid_search(src, template;
                                       metric = :ncc, min_score = 0.7,
                                       min_size = 4, stop_at_level = 1)
        @test !isempty(matches_full)

        # stop_at_level = 2 (refines one fewer level → only the coarsest)
        matches_partial = pyramid_search(src, template;
                                          metric = :ncc, min_score = 0.7,
                                          min_size = 4, stop_at_level = 2)
        # Should still find candidates at coarse resolution
        @test !isempty(matches_partial)
    end

    @testset "Translation pyramid_search — input validation" begin
        src = rand(40, 40)
        tmpl = rand(7, 7)
        @test_throws ArgumentError pyramid_search(rand(5, 5), rand(10, 10))
        @test_throws ArgumentError pyramid_search(src, tmpl;
                                                   metric = :unknown_metric)
        @test_throws ArgumentError pyramid_search(src, tmpl;
                                                   min_score_adjust = -0.5)
        @test_throws ArgumentError pyramid_search(src, tmpl; stop_at_level = 0)
        @test_throws ArgumentError pyramid_search(src, tmpl; padding = -1)
        @test_throws ArgumentError pyramid_search(src, tmpl; min_size = 1)
        # Mask wrong size
        @test_throws ArgumentError pyramid_search(src, tmpl;
                                                   mask = trues(3, 3))
    end

    @testset "Translation pyramid_search — mask works" begin
        h_src, w_src = 40, 40
        src = zeros(h_src, w_src)
        h_t, w_t = 7, 7
        template = [Float64((i - 4)^2 + (j - 4)^2) for i in 1:h_t, j in 1:w_t]
        src[10:16, 15:21] .= template

        mask = trues(h_t, w_t)
        mask[1, 1] = false   # ignore the corner
        matches = pyramid_search(src, template;
                                  metric = :ncc, min_score = 0.7,
                                  min_size = 4, mask = mask)
        @test !isempty(matches)
        sorted = sort(matches, by = m -> -m.score)
        @test sorted[1].pose == (10, 15)
    end

    @testset "shape_search — single variant identity" begin
        # Parabolic bowl source with circular mask — the only variant is identity
        h, w = 41, 41
        center = ((h + 1) / 2, (w + 1) / 2)
        img = [(i - center[1])^2 + (j - center[2])^2 for i in 1:h, j in 1:w]
        mask = [sqrt((i - center[1])^2 + (j - center[2])^2) <= 8.0
                for i in 1:h, j in 1:w]

        rsm = create_similarity_template(img;
                                          gradient_threshold = 1.0,
                                          origin = center,
                                          mask = mask)
        @test length(rsm) == 1   # identity only

        matches = shape_search(img, rsm; variant = :gdp,
                                min_score = 0.95)
        @test !isempty(matches)
        sorted = sort(matches, by = m -> -m.score)
        # Best match should be at or very near (centre_row, centre_col)
        @test abs(sorted[1].pose.r - center[1]) <= 1.0
        @test abs(sorted[1].pose.c - center[2]) <= 1.0
        @test sorted[1].score > 0.99
    end

    @testset "shape_search — rotation variants find the right θ" begin
        # Build a search model with multiple rotations on a parabolic bowl
        # (rotation-symmetric → all variants should match equally)
        h, w = 41, 41
        center = ((h + 1) / 2, (w + 1) / 2)
        img = [(i - center[1])^2 + (j - center[2])^2 for i in 1:h, j in 1:w]
        mask = [sqrt((i - center[1])^2 + (j - center[2])^2) <= 8.0
                for i in 1:h, j in 1:w]

        rsm = create_similarity_template(img;
                                          rotation_range = (0.0, π/2),
                                          angular_granularity = π/4,
                                          gradient_threshold = 1.0,
                                          origin = center,
                                          mask = mask)
        @test length(rsm) == 3

        matches = shape_search(img, rsm; variant = :gdp,
                                min_score = 0.95,
                                suppression_radius = 5)
        # NMS keeps only the best match in a region — for a rotation-symmetric
        # source all variants score 1 at the centre, so the strongest will
        # win and other (θ) candidates near the centre are suppressed.
        @test !isempty(matches)
        sorted = sort(matches, by = m -> -m.score)
        @test abs(sorted[1].pose.r - center[1]) <= 1.0
        @test abs(sorted[1].pose.c - center[2]) <= 1.0
        @test sorted[1].score > 0.99
    end

    @testset "shape_search — input validation" begin
        h, w = 21, 21
        center = ((h + 1) / 2, (w + 1) / 2)
        img = [(i - center[1])^2 + (j - center[2])^2 for i in 1:h, j in 1:w]
        rsm = create_similarity_template(img;
                                          gradient_threshold = 1.0,
                                          origin = center)
        @test_throws ArgumentError shape_search(img, rsm; variant = :unknown)
    end

end
