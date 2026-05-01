# Unit tests for the Patterns type definitions.
# This file is supposed to be included from runtests.jl.

@testset "Types" begin

    @testset "Pose2D" begin
        # identity
        p = identity_pose()
        @test p.A == [1.0 0.0; 0.0 1.0]
        @test p.t == [0.0, 0.0]
        @test apply_pose(p, [3.0, 4.0]) == [3.0, 4.0]

        # pure translation
        pt = Pose2D([1.0 0.0; 0.0 1.0], [3.0, 4.0])
        @test apply_pose(pt, [1.0, 1.0]) == [4.0, 5.0]

        # pure rotation 90°
        pr = similarity_pose(π/2, 1.0, 0.0, 0.0)
        @test apply_pose(pr, [1.0, 0.0]) ≈ [0.0, 1.0] atol=1e-12
        @test apply_pose(pr, [0.0, 1.0]) ≈ [-1.0, 0.0] atol=1e-12

        # uniform scaling
        ps = similarity_pose(0.0, 2.0, 0.0, 0.0)
        @test apply_pose(ps, [1.0, 0.0]) == [2.0, 0.0]
        @test apply_pose(ps, [0.0, 3.0]) == [0.0, 6.0]

        # rotation + scale + translation
        psim = similarity_pose(π, 2.0, 5.0, 0.0)
        @test apply_pose(psim, [1.0, 0.0]) ≈ [3.0, 0.0] atol=1e-12  # rotated to (-2,0), then +5 → (3,0)

        # equality
        @test identity_pose() == identity_pose()
        @test similarity_pose(0.0, 1.0, 0.0, 0.0) == identity_pose()
        @test identity_pose() != Pose2D([1.0 0.0; 0.0 1.0], [1.0, 0.0])

        # vector and tuple translation work
        @test similarity_pose(0.0, 1.0, [3.0, 4.0]) == similarity_pose(0.0, 1.0, 3.0, 4.0)

        # input validation
        @test_throws ArgumentError Pose2D([1.0 0.0; 0.0 1.0; 0.0 0.0], [0.0, 0.0])
        @test_throws ArgumentError Pose2D([1.0 0.0; 0.0 1.0], [0.0, 0.0, 0.0])
        @test_throws ArgumentError apply_pose(identity_pose(), [1.0, 2.0, 3.0])
    end

    @testset "Match" begin
        # tuple pose (translation only)
        m1 = Match(0.95, (10, 20))
        @test m1.score == 0.95
        @test m1.pose == (10, 20)
        @test m1 isa Match{Tuple{Int,Int}}

        # named-tuple pose (similarity)
        m2 = Match(0.8, (x=10, y=20, θ=0.1, s=1.2))
        @test m2.score == 0.8
        @test m2.pose.θ == 0.1
        @test m2.pose.s == 1.2

        # full Pose2D
        m3 = Match(0.7, identity_pose())
        @test m3.pose == identity_pose()

        # equality
        @test Match(0.5, (1, 2)) == Match(0.5, (1, 2))
        @test Match(0.5, (1, 2)) != Match(0.6, (1, 2))
        @test Match(0.5, (1, 2)) != Match(0.5, (1, 3))

        # score is coerced to Float64
        @test Match(1, (0, 0)).score === 1.0
    end

    @testset "ShapeModel" begin
        m = ShapeModel([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                       [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        @test length(m) == 3
        @test m.points[1] == [0.0, 0.0]
        @test m.gradients[3] == [1.0, 1.0]

        # empty model is allowed
        empty = ShapeModel(Vector{Float64}[], Vector{Float64}[])
        @test length(empty) == 0

        # equality
        m2 = ShapeModel([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        @test m == m2

        # input validation
        @test_throws ArgumentError ShapeModel([[0.0, 0.0]],
                                              [[1.0, 0.0], [0.0, 1.0]])
        @test_throws ArgumentError ShapeModel([[0.0, 0.0, 0.0]], [[1.0, 0.0]])
        @test_throws ArgumentError ShapeModel([[0.0, 0.0]], [[1.0, 0.0, 0.0]])
    end

    @testset "KeypointMatch" begin
        km = KeypointMatch((1.0, 2.0), (10.0, 20.0), 42)
        @test km.model_pos == (1.0, 2.0)
        @test km.image_pos == (10.0, 20.0)
        @test km.distance == 42

        # integer positions are coerced to Float64
        km2 = KeypointMatch((1, 2), (10, 20), 42)
        @test km2 == km

        # equality
        @test KeypointMatch((1.0, 2.0), (3.0, 4.0), 5) ==
              KeypointMatch((1.0, 2.0), (3.0, 4.0), 5)
        @test KeypointMatch((1.0, 2.0), (3.0, 4.0), 5) !=
              KeypointMatch((1.0, 2.0), (3.0, 4.0), 6)
    end

    @testset "TransformResult" begin
        km1 = KeypointMatch((1.0, 2.0), (10.0, 20.0), 5)
        km2 = KeypointMatch((3.0, 4.0), (30.0, 40.0), 8)

        r = TransformResult(identity_pose(), [km1, km2], [], 0.85, 0.7)
        @test r.pose == identity_pose()
        @test length(r.inliers) == 2
        @test length(r.outliers) == 0
        @test r.score == 0.85
        @test r.certainty == 0.7

        # outliers can be non-empty
        r2 = TransformResult(identity_pose(), [km1], [km2], 0.6, 0.5)
        @test length(r2.inliers) == 1
        @test length(r2.outliers) == 1

        # equality
        @test TransformResult(identity_pose(), [km1], [km2], 0.6, 0.5) ==
              TransformResult(identity_pose(), [km1], [km2], 0.6, 0.5)
    end

    @testset "AbstractSearchModel" begin
        # only the abstract type is defined at this stage
        @test AbstractSearchModel isa Type
        @test isabstracttype(AbstractSearchModel)
    end

end
