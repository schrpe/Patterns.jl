# Unit tests for subpixel localization.
# This file is supposed to be included from runtests.jl.

@testset "Subpixel" begin

    @testset "subpixel_peak — 2D quadratic" begin
        # Synthetic 5×5 with peak at row 3.3, col 2.8 (so the discrete max is at (3, 3)).
        scores = [1.0 - ((i - 3.3)^2 + (j - 2.8)^2) for i in 1:5, j in 1:5]
        dx, dy, refined = subpixel_peak(scores, CartesianIndex(3, 3))
        @test isapprox(dx,  0.3; atol = 1e-10)
        @test isapprox(dy, -0.2; atol = 1e-10)
        @test isapprox(refined, 1.0; atol = 1e-10)
    end

    @testset "subpixel_peak — peak with cross-term" begin
        # f(i,j) = 2 - (i - 3 - 0.4)² - (j - 3 - 0.1)² - 0.5·((i-3-0.4)·(j-3-0.1))
        scores = [2.0 - (i - 3 - 0.4)^2 - (j - 3 - 0.1)^2 -
                  0.5 * (i - 3 - 0.4) * (j - 3 - 0.1) for i in 1:5, j in 1:5]
        dx, dy, refined = subpixel_peak(scores, CartesianIndex(3, 3))
        @test isapprox(dx, 0.4; atol = 1e-10)
        @test isapprox(dy, 0.1; atol = 1e-10)
        @test isapprox(refined, 2.0; atol = 1e-10)
    end

    @testset "subpixel_peak — boundary fallback" begin
        scores = [1.0 - ((i - 3.3)^2 + (j - 2.8)^2) for i in 1:5, j in 1:5]

        # Corner: no full 3×3 neighborhood
        dx, dy, refined = subpixel_peak(scores, CartesianIndex(1, 1))
        @test dx == 0.0
        @test dy == 0.0
        @test refined == scores[1, 1]

        # Edge: same fallback
        dx, dy, _ = subpixel_peak(scores, CartesianIndex(1, 3))
        @test dx == 0.0
        @test dy == 0.0
    end

    @testset "subpixel_peak — degenerate cases" begin
        # Flat region: no curvature, fallback
        flat = ones(5, 5)
        dx, dy, refined = subpixel_peak(flat, CartesianIndex(3, 3))
        @test dx == 0.0
        @test dy == 0.0
        @test refined == 1.0

        # Saddle (positive curvature in one direction): fallback
        saddle = [Float64((i - 3)^2 - (j - 3)^2) for i in 1:5, j in 1:5]
        dx, dy, _ = subpixel_peak(saddle, CartesianIndex(3, 3))
        @test dx == 0.0
        @test dy == 0.0
    end

    @testset "subpixel_peak — large offset fallback" begin
        # An asymmetric profile that suggests the true max is far from the discrete peak.
        # Construct so that the analytical extremum lies > 1 pixel away.
        scores = [exp(-0.5 * ((i - 7)^2 + (j - 3)^2) / 4) for i in 1:5, j in 1:5]
        # Discrete max at (5, 3). The analytical fit will want to extrapolate beyond 1 pixel.
        dx, dy, refined = subpixel_peak(scores, CartesianIndex(5, 3))
        @test dx == 0.0
        @test dy == 0.0
        @test refined == scores[5, 3]
    end

    # Helper: construct rectangle edge points + axis-aligned gradients.
    # A rectangle gives well-conditioned LSQ for similarity (4 DOF) — unlike a circle
    # with radial gradients, where rotation about the circle centre is unobservable
    # in the perpendicular-to-tangent metric.
    function _rectangle_correspondences(true_pose; n_per_side::Int = 5,
                                        half_w::Float64 = 10.0,
                                        half_h::Float64 = 10.0,
                                        edge_noise::Float64 = 0.0)
        model_points = Vector{Vector{Float64}}()
        model_grads  = Vector{Vector{Float64}}()
        xs = collect(range(-half_w, half_w, length = n_per_side))
        ys = collect(range(-half_h, half_h, length = n_per_side))
        for x in xs
            push!(model_points, [x,  half_h]); push!(model_grads, [0.0,  1.0])
            push!(model_points, [x, -half_h]); push!(model_grads, [0.0, -1.0])
        end
        for y in ys
            push!(model_points, [ half_w, y]); push!(model_grads, [ 1.0, 0.0])
            push!(model_points, [-half_w, y]); push!(model_grads, [-1.0, 0.0])
        end
        n = length(model_points)
        image_edges = [apply_pose(true_pose, model_points[i]) .+
                       edge_noise * sin(7 * i) * model_grads[i]
                       for i in 1:n]
        return [(model_point = model_points[i],
                 model_gradient = model_grads[i],
                 image_edge = image_edges[i])
                for i in 1:n]
    end

    @testset "refine_pose_similarity — perfect data (rectangle)" begin
        # True similarity pose: 8° rotation, 1.05 scale, translation (50, -30)
        true_pose = similarity_pose(deg2rad(8), 1.05, 50.0, -30.0)
        correspondences = _rectangle_correspondences(true_pose)

        init_pose = similarity_pose(deg2rad(5), 1.02, 48.0, -28.0)
        refined, residual, iters = refine_pose_similarity(init_pose, correspondences)

        @test isapprox(refined.A, true_pose.A; atol = 1e-6)
        @test isapprox(refined.t, true_pose.t; atol = 1e-6)
        @test residual < 1e-6
        @test iters >= 1
        @test iters <= 20
    end

    @testset "refine_pose_similarity — start from identity, recover translation" begin
        # Pure translation, easier convergence
        true_pose = Pose2D([1.0 0.0; 0.0 1.0], [10.0, 5.0])

        # Random non-degenerate model points + gradients
        n = 8
        model_points = [[Float64(i), Float64(2 * i)] for i in 1:n]
        model_grads  = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0],
                        [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        image_edges = [apply_pose(true_pose, p) for p in model_points]

        correspondences = [
            (model_point = model_points[i],
             model_gradient = model_grads[i],
             image_edge = image_edges[i])
            for i in 1:n
        ]

        refined, residual, iters = refine_pose_similarity(identity_pose(), correspondences)
        @test isapprox(refined.A, true_pose.A; atol = 1e-6)
        @test isapprox(refined.t, true_pose.t; atol = 1e-6)
        @test residual < 1e-6
    end

    @testset "refine_pose_similarity — small noise in image edges (rectangle)" begin
        true_pose = similarity_pose(deg2rad(3), 1.0, 20.0, 10.0)
        # ~0.05 px noise along the gradient direction at each edge point
        correspondences = _rectangle_correspondences(true_pose;
                                                     n_per_side = 10,
                                                     edge_noise = 0.05)

        init_pose = similarity_pose(deg2rad(1), 0.99, 19.0, 11.0)
        refined, residual, _ = refine_pose_similarity(init_pose, correspondences)

        # With small noise, parameters should be close to truth (within a few percent)
        @test isapprox(refined.A, true_pose.A; atol = 5e-3)
        @test isapprox(refined.t, true_pose.t; atol = 0.1)
        # Residual: 40 points × 0.05 noise ≈ √40 · 0.05 ≈ 0.32 (RSS lower bound)
        @test residual < 1.0
    end

    @testset "refine_pose_similarity — input validation" begin
        @test_throws ArgumentError refine_pose_similarity(
            identity_pose(),
            [(model_point = [0.0, 0.0],
              model_gradient = [1.0, 0.0],
              image_edge = [0.0, 0.0])],
        )
    end

end
