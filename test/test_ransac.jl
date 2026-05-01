# Unit tests for ransac.jl.
# This file is supposed to be included from runtests.jl.

@testset "RANSAC" begin

    # Helper: synthesize matches by transforming model points with a known pose.
    function _matches_from_pose(pose::Pose2D,
                                 model_points::Vector{Tuple{Float64,Float64}};
                                 noise::Float64 = 0.0)
        matches = KeypointMatch[]
        for (k, p) in enumerate(model_points)
            tp = pose.A * [p[1], p[2]] + pose.t
            # Deterministic pseudo-noise so tests are reproducible
            r = tp[1] + noise * sin(7 * k)
            c = tp[2] + noise * cos(11 * k)
            push!(matches, KeypointMatch(p, (r, c), 0))
        end
        return matches
    end

    @testset "compute_similarity_pose — identity" begin
        m1 = KeypointMatch((1.0, 0.0), (1.0, 0.0), 0)
        m2 = KeypointMatch((0.0, 1.0), (0.0, 1.0), 0)
        pose = compute_similarity_pose(m1, m2)
        @test isapprox(pose.A, [1.0 0.0; 0.0 1.0]; atol = 1e-12)
        @test isapprox(pose.t, [0.0, 0.0]; atol = 1e-12)
    end

    @testset "compute_similarity_pose — pure rotation" begin
        # Rotate (1, 0) → (0, 1) and (0, 1) → (-1, 0) in (row, col) frame
        # In our convention with A = [a -b; b a]:
        #   A · (1, 0) = (a, b)   → (0, 1) ⇒ a = 0, b = 1   (90° rotation)
        #   A · (0, 1) = (-b, a) = (-1, 0) ✓
        m1 = KeypointMatch((1.0, 0.0), (0.0, 1.0), 0)
        m2 = KeypointMatch((0.0, 1.0), (-1.0, 0.0), 0)
        pose = compute_similarity_pose(m1, m2)
        @test isapprox(pose.A, [0.0 -1.0; 1.0 0.0]; atol = 1e-12)
        @test isapprox(pose.t, [0.0, 0.0]; atol = 1e-12)
        @test isapprox(sqrt(pose.A[1, 1]^2 + pose.A[2, 1]^2), 1.0; atol = 1e-12)
    end

    @testset "compute_similarity_pose — uniform scaling + translation" begin
        # Scale = 2, translation = (5, -3), no rotation
        true_pose = similarity_pose(0.0, 2.0, 5.0, -3.0)
        m1 = KeypointMatch((1.0, 0.0),
                            Tuple(true_pose.A * [1.0, 0.0] + true_pose.t),
                            0)
        m2 = KeypointMatch((0.0, 1.0),
                            Tuple(true_pose.A * [0.0, 1.0] + true_pose.t),
                            0)
        pose = compute_similarity_pose(m1, m2)
        @test isapprox(pose.A, true_pose.A; atol = 1e-12)
        @test isapprox(pose.t, true_pose.t; atol = 1e-12)
    end

    @testset "compute_similarity_pose — degenerate input" begin
        # Both matches share the same model point → cannot fit
        m1 = KeypointMatch((1.0, 1.0), (5.0, 5.0), 0)
        m2 = KeypointMatch((1.0, 1.0), (10.0, 10.0), 0)
        @test_throws ArgumentError compute_similarity_pose(m1, m2)
    end

    @testset "estimate_pose — clean inliers" begin
        true_pose = similarity_pose(deg2rad(15), 1.05, 50.0, -30.0)
        # 12 model points spread out
        model_points = [(Float64(i), Float64(2i)) for i in 1:12]
        matches = _matches_from_pose(true_pose, model_points)

        result = estimate_pose(matches; error = 1.0, min_scale = 0.8, max_scale = 1.3)

        @test length(result.inliers) == 12
        @test length(result.outliers) == 0
        @test isapprox(result.pose.A, true_pose.A; atol = 1e-6)
        @test isapprox(result.pose.t, true_pose.t; atol = 1e-6)
        @test result.score > 0.99
        @test result.certainty == 1.0
    end

    @testset "estimate_pose — with outliers" begin
        true_pose = similarity_pose(deg2rad(8), 1.0, 20.0, 10.0)
        # 10 inlier matches plus 4 obvious outliers
        model_points = [(Float64(i), Float64(j)) for i in 1:5, j in 1:2]
        inlier_matches = _matches_from_pose(true_pose, vec(model_points))

        # Outlier: model→image with random unrelated mapping
        outlier_matches = [
            KeypointMatch((100.0, 100.0), (0.0, 0.0), 0),
            KeypointMatch((101.0, 100.0), (0.0, 1.0), 0),
            KeypointMatch((100.0, 101.0), (200.0, 0.0), 0),
            KeypointMatch((101.0, 101.0), (5.0, 5.0), 0),
        ]
        all_matches = vcat(inlier_matches, outlier_matches)

        result = estimate_pose(all_matches; error = 1.0)
        @test length(result.inliers) >= 9
        @test length(result.outliers) >= 1
        @test isapprox(result.pose.A, true_pose.A; atol = 1e-3)
        @test isapprox(result.pose.t, true_pose.t; atol = 0.5)
    end

    @testset "estimate_pose — scale constraints reject out-of-range fits" begin
        # True scale = 3 — but min_scale = 0.8, max_scale = 1.2 → no candidate accepted
        true_pose = similarity_pose(0.0, 3.0, 0.0, 0.0)
        model_points = [(Float64(i), Float64(2i)) for i in 1:6]
        matches = _matches_from_pose(true_pose, model_points)
        result = estimate_pose(matches; error = 0.1,
                                min_scale = 0.8, max_scale = 1.2)
        @test length(result.inliers) == 0
        @test result.score == 0.0
        @test result.certainty == 0.0
    end

    @testset "estimate_pose — input validation" begin
        m = KeypointMatch((0.0, 0.0), (0.0, 0.0), 0)
        @test_throws ArgumentError estimate_pose(KeypointMatch[])
        @test_throws ArgumentError estimate_pose([m])
        @test_throws ArgumentError estimate_pose([m, m]; error = -1)
        @test_throws ArgumentError estimate_pose([m, m]; min_scale = 0)
        @test_throws ArgumentError estimate_pose([m, m]; min_scale = 1.5, max_scale = 0.5)
        @test_throws ArgumentError estimate_pose([m, m]; max_iterations = 0)
    end

    @testset "estimate_all_poses — single instance" begin
        true_pose = similarity_pose(deg2rad(5), 1.0, 30.0, 30.0)
        model_points = [(Float64(i), Float64(2i)) for i in 1:8]
        matches = _matches_from_pose(true_pose, model_points)

        results = estimate_all_poses(matches; error = 1.0, min_inliers = 4)
        @test length(results) == 1
        @test length(results[1].inliers) >= 6
        @test isapprox(results[1].pose.A, true_pose.A; atol = 1e-3)
    end

    @testset "estimate_all_poses — two distinct image instances" begin
        # Two instances of the same model at different image positions
        true_pose1 = similarity_pose(0.0, 1.0,   0.0,   0.0)   # near origin
        true_pose2 = similarity_pose(0.0, 1.0, 100.0, 100.0)  # 100 px away

        model_points = [(Float64(i), Float64(2i)) for i in 1:8]
        matches1 = _matches_from_pose(true_pose1, model_points)
        matches2 = _matches_from_pose(true_pose2, model_points)
        all_matches = vcat(matches1, matches2)

        results = estimate_all_poses(all_matches;
                                      error = 0.5, min_inliers = 4)
        @test length(results) == 2
        # Their translations should be far apart
        t_dist = sqrt((results[1].pose.t[1] - results[2].pose.t[1])^2 +
                      (results[1].pose.t[2] - results[2].pose.t[2])^2)
        @test t_dist > 50.0
    end

    @testset "estimate_all_poses — input validation" begin
        m = KeypointMatch((0.0, 0.0), (0.0, 0.0), 0)
        @test_throws ArgumentError estimate_all_poses([m, m]; min_inliers = 1)
    end

    @testset "estimate_all_poses — multi-model dispatch" begin
        true_pose1 = similarity_pose(0.0, 1.0, 10.0, 10.0)
        true_pose2 = similarity_pose(deg2rad(10), 1.0, 50.0, 50.0)

        model_points = [(Float64(i), Float64(2i)) for i in 1:8]
        list1 = _matches_from_pose(true_pose1, model_points)
        list2 = _matches_from_pose(true_pose2, model_points)

        all_results = estimate_all_poses([list1, list2]; error = 0.5)
        @test length(all_results) == 2
        @test !isempty(all_results[1])
        @test !isempty(all_results[2])
        @test isapprox(all_results[1][1].pose.t, true_pose1.t; atol = 0.5)
        @test isapprox(all_results[2][1].pose.t, true_pose2.t; atol = 0.5)
    end

    @testset "estimate_all_poses — multi-model empty input throws" begin
        @test_throws ArgumentError estimate_all_poses(Vector{KeypointMatch}[])
    end

end
