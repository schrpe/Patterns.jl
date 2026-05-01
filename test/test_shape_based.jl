# Unit tests for shape_based.jl.
# This file is supposed to be included from runtests.jl.

@testset "Shape-Based Matching" begin

    @testset "image_gradients — Sobel wrapper" begin
        img = Float64[i + j for i in 1:5, j in 1:5]
        gx, gy = image_gradients(img)
        @test size(gx) == size(img)
        @test size(gy) == size(img)
        # Boundary pixels are 0
        @test all(==(0.0), gx[1, :])
        @test all(==(0.0), gx[end, :])
        @test all(==(0.0), gx[:, 1])
        @test all(==(0.0), gx[:, end])
        # Interior of a uniform diagonal ramp: gx = gy = 8
        @test gx[3, 3] == 8.0
        @test gy[3, 3] == 8.0
    end

    @testset "build_shape_model" begin
        img = Float64[i + j for i in 1:7, j in 1:7]
        m = build_shape_model(img; gradient_threshold = 1.0)
        # Interior points only (5×5 = 25)
        @test length(m) == 25
        # Each gradient is non-zero
        for d in m.gradients
            @test sqrt(d[1]^2 + d[2]^2) > 1.0
        end

        # With a much higher threshold, no points qualify
        m_empty = build_shape_model(img; gradient_threshold = 100.0)
        @test length(m_empty) == 0

        # Mask filters out specific pixels
        mask = trues(7, 7)
        mask[3, 3] = false
        m_masked = build_shape_model(img; gradient_threshold = 1.0, mask = mask)
        @test length(m_masked) == 24

        # Custom origin shifts point coordinates
        m_origin = build_shape_model(img; gradient_threshold = 1.0,
                                          origin = (1.0, 1.0))
        # Default origin = (4.0, 4.0); difference between any point should be (3.0, 3.0)
        @test m_origin.points[1] - m.points[1] ≈ [3.0, 3.0]

        # Mask wrong size raises
        @test_throws ArgumentError build_shape_model(img; mask = trues(3, 3))
    end

    @testset "transform_shape_model — identity, rotation, scaling" begin
        m = ShapeModel([[1.0, 0.0], [0.0, 1.0]],
                       [[1.0, 0.0], [0.0, 1.0]])

        # Identity transform: model unchanged
        m_id = transform_shape_model(m, [1.0 0.0; 0.0 1.0])
        @test m_id.points == m.points
        @test m_id.gradients == m.gradients

        # 90° rotation: orthogonal, so gradients rotate the same way as points
        A_rot = [0.0 -1.0; 1.0 0.0]
        m_rot = transform_shape_model(m, A_rot)
        @test m_rot.points[1] ≈ [0.0,  1.0]
        @test m_rot.points[2] ≈ [-1.0, 0.0]
        @test m_rot.gradients[1] ≈ [0.0,  1.0]
        @test m_rot.gradients[2] ≈ [-1.0, 0.0]

        # Uniform scaling by 2: points double, gradients halve
        A_scale = [2.0 0.0; 0.0 2.0]
        m_sc = transform_shape_model(m, A_scale)
        @test m_sc.points[1] ≈ [2.0, 0.0]
        @test m_sc.gradients[1] ≈ [0.5, 0.0]

        # Input validation
        @test_throws ArgumentError transform_shape_model(m, ones(3, 3))
    end

    @testset "shape_score_gdp — perfect match at the model origin" begin
        # Diagonal-ramp image as both reference and source
        h, w = 21, 21
        img = Float64[i + j for i in 1:h, j in 1:w]
        center = ((h + 1) / 2, (w + 1) / 2)

        model = build_shape_model(img; gradient_threshold = 1.0, origin = center)
        src_gx, src_gy = image_gradients(img)

        pose = Pose2D([1.0 0.0; 0.0 1.0], [center[1], center[2]])
        score = shape_score_gdp(src_gx, src_gy, model, pose)
        @test isapprox(score, 1.0; atol = 1e-10)
    end

    @testset "shape_score_gdp — clamps inverted contrast" begin
        h, w = 21, 21
        img = Float64[i + j for i in 1:h, j in 1:w]
        center = ((h + 1) / 2, (w + 1) / 2)
        model = build_shape_model(img; gradient_threshold = 1.0, origin = center)

        # Source with all gradients inverted (negate the image)
        src_gx, src_gy = image_gradients(-img)
        pose = Pose2D([1.0 0.0; 0.0 1.0], [center[1], center[2]])
        @test isapprox(shape_score_gdp(src_gx, src_gy, model, pose),
                       0.0; atol = 1e-10)
    end

    @testset "shape_score_gdpr — global contrast reversal" begin
        h, w = 21, 21
        img = Float64[i + j for i in 1:h, j in 1:w]
        center = ((h + 1) / 2, (w + 1) / 2)
        model = build_shape_model(img; gradient_threshold = 1.0, origin = center)

        # Inverted-contrast source
        src_gx, src_gy = image_gradients(-img)
        pose = Pose2D([1.0 0.0; 0.0 1.0], [center[1], center[2]])
        @test isapprox(shape_score_gdpr(src_gx, src_gy, model, pose),
                       1.0; atol = 1e-10)

        # Direct match also gives 1.0
        src_gx2, src_gy2 = image_gradients(img)
        @test isapprox(shape_score_gdpr(src_gx2, src_gy2, model, pose),
                       1.0; atol = 1e-10)
    end

    @testset "shape_score_gdpr_local — local contrast reversal" begin
        h, w = 21, 21
        img = Float64[i + j for i in 1:h, j in 1:w]
        center = ((h + 1) / 2, (w + 1) / 2)
        model = build_shape_model(img; gradient_threshold = 1.0, origin = center)

        src_gx, src_gy = image_gradients(-img)
        pose = Pose2D([1.0 0.0; 0.0 1.0], [center[1], center[2]])
        @test isapprox(shape_score_gdpr_local(src_gx, src_gy, model, pose),
                       1.0; atol = 1e-10)
    end

    @testset "shape_score_gdp — translation-only convenience method" begin
        h, w = 21, 21
        img = Float64[i + j for i in 1:h, j in 1:w]
        center = ((h + 1) / 2, (w + 1) / 2)
        model = build_shape_model(img; gradient_threshold = 1.0, origin = center)
        src_gx, src_gy = image_gradients(img)

        # Both APIs produce the same score
        s_pose = shape_score_gdp(src_gx, src_gy, model,
                                 Pose2D([1.0 0.0; 0.0 1.0], [center[1], center[2]]))
        s_t = shape_score_gdp(src_gx, src_gy, model, [center[1], center[2]])
        @test s_pose == s_t
    end

    @testset "shape_score_gdp — rotation invariance via transform_shape_model" begin
        # Parabolic bowl source: f(p) = |p|² has gradient ∇f = 2·p, which the 3×3 Sobel
        # reproduces exactly up to a constant factor and bilinearly interpolated values
        # are exact at fractional positions. The image is rotationally symmetric about
        # its centre, so any rotation of the model should still match perfectly — provided
        # the rotated points stay inside the source. We restrict the model to a circular
        # mask of radius ≤ source-radius so all rotated points remain in bounds.
        h, w = 41, 41
        center = ((h + 1) / 2, (w + 1) / 2)
        img = [(i - center[1])^2 + (j - center[2])^2 for i in 1:h, j in 1:w]
        src_gx, src_gy = image_gradients(img)

        mask_radius = 15.0
        mask = [sqrt((i - center[1])^2 + (j - center[2])^2) <= mask_radius
                for i in 1:h, j in 1:w]

        model = build_shape_model(img; gradient_threshold = 1.0,
                                       origin = center, mask = mask)

        # Identity → score = 1 exactly
        score_id = shape_score_gdp(src_gx, src_gy, model,
                                   Pose2D([1.0 0.0; 0.0 1.0], [center[1], center[2]]))
        @test isapprox(score_id, 1.0; atol = 1e-10)

        # For rotations the bilinear interpolation gives the exact rotated direction
        # (because gx and gy are linear in the pixel coordinates for a parabolic bowl)
        # and all points stay inside the circular mask, so the score remains 1 exactly.
        for θ in (π/6, π/4, π/3, π/2, π)
            A_rot = [cos(θ) -sin(θ); sin(θ) cos(θ)]
            pose = Pose2D(A_rot, [center[1], center[2]])
            score = shape_score_gdp(src_gx, src_gy, model, pose)
            @test isapprox(score, 1.0; atol = 1e-10)
        end
    end

    @testset "shape_score_gdp — pre-transformed model" begin
        h, w = 41, 41
        center = ((h + 1) / 2, (w + 1) / 2)
        img = [(i - center[1])^2 + (j - center[2])^2 for i in 1:h, j in 1:w]
        src_gx, src_gy = image_gradients(img)
        mask = [sqrt((i - center[1])^2 + (j - center[2])^2) <= 15.0
                for i in 1:h, j in 1:w]
        model = build_shape_model(img; gradient_threshold = 1.0,
                                       origin = center, mask = mask)

        # Pre-rotate the model by 30° and evaluate with translation only
        θ = π / 6
        A = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        m_rot = transform_shape_model(model, A)

        s_pretransformed = shape_score_gdp(src_gx, src_gy, m_rot,
                                            [center[1], center[2]])
        s_inline = shape_score_gdp(src_gx, src_gy, model,
                                    Pose2D(A, [center[1], center[2]]))
        # The two paths must agree (modulo float arithmetic)
        @test isapprox(s_pretransformed, s_inline; atol = 1e-10)
    end

    @testset "shape_score — bias correction with min_source_gradient" begin
        # Build a small model from a clean reference
        h, w = 21, 21
        img = Float64[i + j for i in 1:h, j in 1:w]
        center = ((h + 1) / 2, (w + 1) / 2)
        model = build_shape_model(img; gradient_threshold = 1.0, origin = center)

        # Source: same content, but at the queried position the source gradient
        # is exactly the same → bias correction has no effect on score.
        src_gx, src_gy = image_gradients(img)
        pose = Pose2D([1.0 0.0; 0.0 1.0], [center[1], center[2]])

        s_no_bias  = shape_score_gdp(src_gx, src_gy, model, pose;
                                      min_source_gradient = 0.0)
        s_with_bias = shape_score_gdp(src_gx, src_gy, model, pose;
                                       min_source_gradient = 1.0)
        @test isapprox(s_no_bias, 1.0; atol = 1e-10)
        @test isapprox(s_with_bias, 1.0; atol = 1e-10)

        # An aggressive bias threshold removes ALL source contributions → score = 0
        s_max_bias = shape_score_gdp(src_gx, src_gy, model, pose;
                                     min_source_gradient = 1e6)
        @test s_max_bias == 0.0
    end

    @testset "shape_score — early abort returns 0 below min_score" begin
        h, w = 21, 21
        img = Float64[i + j for i in 1:h, j in 1:w]
        center = ((h + 1) / 2, (w + 1) / 2)
        model = build_shape_model(img; gradient_threshold = 1.0, origin = center)
        src_gx, src_gy = image_gradients(-img)  # inverted: GDP score will be 0

        pose = Pose2D([1.0 0.0; 0.0 1.0], [center[1], center[2]])
        score = shape_score_gdp(src_gx, src_gy, model, pose; min_score = 0.5)
        @test score == 0.0
    end

    @testset "shape_score — out-of-bounds points are skipped" begin
        h, w = 21, 21
        img = Float64[i + j for i in 1:h, j in 1:w]
        center = ((h + 1) / 2, (w + 1) / 2)
        model = build_shape_model(img; gradient_threshold = 1.0, origin = center)
        src_gx, src_gy = image_gradients(img)

        # Translation far outside the source: most/all model points fall outside
        pose = Pose2D([1.0 0.0; 0.0 1.0], [1000.0, 1000.0])
        score = shape_score_gdp(src_gx, src_gy, model, pose)
        @test score == 0.0
    end

    @testset "Empty model returns 0" begin
        empty_model = ShapeModel(Vector{Float64}[], Vector{Float64}[])
        src_gx, src_gy = image_gradients(zeros(10, 10))
        @test shape_score_gdp(src_gx, src_gy, empty_model,
                              Pose2D([1.0 0.0; 0.0 1.0], [1.0, 1.0])) == 0.0
        @test shape_score_gdpr(src_gx, src_gy, empty_model,
                               Pose2D([1.0 0.0; 0.0 1.0], [1.0, 1.0])) == 0.0
        @test shape_score_gdpr_local(src_gx, src_gy, empty_model,
                                     Pose2D([1.0 0.0; 0.0 1.0], [1.0, 1.0])) == 0.0
    end

end
