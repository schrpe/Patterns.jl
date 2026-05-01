# Unit tests for similarity_search.jl.
# This file is supposed to be included from runtests.jl.

@testset "Similarity Search" begin

    # Reusable parabolic-bowl reference image
    h, w = 41, 41
    center = ((h + 1) / 2, (w + 1) / 2)
    img = [(i - center[1])^2 + (j - center[2])^2 for i in 1:h, j in 1:w]
    mask = [sqrt((i - center[1])^2 + (j - center[2])^2) <= 15.0
            for i in 1:h, j in 1:w]

    @testset "Single variant — identity" begin
        rsm = create_similarity_template(img;
                                          gradient_threshold = 1.0,
                                          origin = center)
        @test length(rsm) == 1
        @test rsm.n_rotations == 1
        @test rsm.n_scales == 1
        v = rsm[1]
        @test v.θ == 0.0
        @test v.scale == 1.0
        # Identity variant has the same points/gradients as the base model
        @test v.model.points == rsm.base_model.points
        @test v.model.gradients == rsm.base_model.gradients
    end

    @testset "Rotation only" begin
        rsm = create_similarity_template(img;
                                          rotation_range = (0.0, π/2),
                                          angular_granularity = π/4,
                                          gradient_threshold = 1.0,
                                          origin = center)
        @test length(rsm) == 3            # 0, π/4, π/2
        @test rsm.n_rotations == 3
        @test rsm.n_scales == 1
        @test rsm[1].θ ≈ 0.0
        @test rsm[2].θ ≈ π/4
        @test rsm[3].θ ≈ π/2
        for v in rsm
            @test v.scale == 1.0
        end
    end

    @testset "Scale only" begin
        rsm = create_similarity_template(img;
                                          scale_range = (0.8, 1.2),
                                          scale_granularity = 0.2,
                                          gradient_threshold = 1.0,
                                          origin = center)
        @test length(rsm) == 3            # 0.8, 1.0, 1.2
        @test rsm.n_rotations == 1
        @test rsm.n_scales == 3
        @test rsm[1].scale ≈ 0.8
        @test rsm[2].scale ≈ 1.0
        @test rsm[3].scale ≈ 1.2
        for v in rsm
            @test v.θ == 0.0
        end
    end

    @testset "Rotation × scale grid" begin
        rsm = create_similarity_template(img;
                                          rotation_range = (0.0, π/2),
                                          angular_granularity = π/4,
                                          scale_range = (0.8, 1.2),
                                          scale_granularity = 0.2,
                                          gradient_threshold = 1.0,
                                          origin = center)
        @test length(rsm) == 9            # 3 × 3
        @test rsm.n_rotations == 3
        @test rsm.n_scales == 3

        # Variants are stored with outer θ, inner scale
        # So rsm[1] is (0.0, 0.8), rsm[2] is (0.0, 1.0), ..., rsm[4] is (π/4, 0.8)
        @test rsm[1].θ ≈ 0.0   && rsm[1].scale ≈ 0.8
        @test rsm[2].θ ≈ 0.0   && rsm[2].scale ≈ 1.0
        @test rsm[3].θ ≈ 0.0   && rsm[3].scale ≈ 1.2
        @test rsm[4].θ ≈ π/4  && rsm[4].scale ≈ 0.8
        @test rsm[9].θ ≈ π/2  && rsm[9].scale ≈ 1.2
    end

    @testset "Variant transformation matches transform_shape_model" begin
        rsm = create_similarity_template(img;
                                          rotation_range = (π/3, π/3),
                                          scale_range = (1.5, 1.5),
                                          gradient_threshold = 1.0,
                                          origin = center)
        @test length(rsm) == 1
        v = rsm[1]

        # Build the same transformation manually and compare
        θ, s = π/3, 1.5
        A = [s*cos(θ) -s*sin(θ); s*sin(θ) s*cos(θ)]
        m_expected = transform_shape_model(rsm.base_model, A)

        @test v.model.points    == m_expected.points
        @test v.model.gradients == m_expected.gradients
    end

    @testset "Iteration" begin
        rsm = create_similarity_template(img;
                                          rotation_range = (0.0, π/2),
                                          angular_granularity = π/4,
                                          gradient_threshold = 1.0,
                                          origin = center)
        collected = ShapeVariant[]
        for v in rsm
            push!(collected, v)
        end
        @test length(collected) == 3
        @test collected[1] == rsm[1]
        @test collected[end] == rsm[end]
    end

    @testset "nearest_variant" begin
        rsm = create_similarity_template(img;
                                          rotation_range = (0.0, π),
                                          angular_granularity = π/4,
                                          scale_range = (1.0, 2.0),
                                          scale_granularity = 0.5,
                                          gradient_threshold = 1.0,
                                          origin = center)
        # Exact grid point
        v = nearest_variant(rsm, π/4, 1.5)
        @test v.θ ≈ π/4
        @test v.scale ≈ 1.5

        # Slightly off — should round to the nearest
        v_round = nearest_variant(rsm, π/4 + 0.05, 1.45)
        @test v_round.θ ≈ π/4
        @test v_round.scale ≈ 1.5

        # Outside grid — clamped to boundary
        v_clamp = nearest_variant(rsm, -10.0, 0.1)
        @test v_clamp.θ ≈ 0.0       # min rotation
        @test v_clamp.scale ≈ 1.0   # min scale

        v_clamp_hi = nearest_variant(rsm, 10.0, 100.0)
        @test v_clamp_hi.θ ≈ π
        @test v_clamp_hi.scale ≈ 2.0
    end

    @testset "All variants match a rotation-symmetric source" begin
        # Parabolic bowl is rotation-symmetric: every rotated variant scores 1
        # at the centre (within the circular mask, no points fall out of bounds).
        rsm = create_similarity_template(img;
                                          rotation_range = (-π, π/2),
                                          angular_granularity = π/4,
                                          gradient_threshold = 1.0,
                                          origin = center,
                                          mask = mask)
        src_gx, src_gy = image_gradients(img)
        for v in rsm
            score = shape_score_gdp(src_gx, src_gy, v.model,
                                    [center[1], center[2]])
            @test isapprox(score, 1.0; atol = 1e-10)
        end
    end

    @testset "Uniform scale invariance under cosine matching" begin
        # Parabolic bowl is self-similar under uniform scaling: with gradients
        # normalized to direction, the cosine match remains 1 for any scale
        # variant — provided the scaled points still fall inside the source.
        # We use a smaller mask radius (8) so even scale 2.0 reaches at most
        # row 37 (well inside the 41×41 source).
        small_mask = [sqrt((i - center[1])^2 + (j - center[2])^2) <= 8.0
                      for i in 1:h, j in 1:w]
        rsm = create_similarity_template(img;
                                          scale_range = (0.5, 2.0),
                                          scale_granularity = 0.5,
                                          gradient_threshold = 1.0,
                                          origin = center,
                                          mask = small_mask)
        src_gx, src_gy = image_gradients(img)
        for v in rsm
            score = shape_score_gdp(src_gx, src_gy, v.model,
                                    [center[1], center[2]])
            @test isapprox(score, 1.0; atol = 1e-10)
        end
    end

    @testset "Input validation" begin
        @test_throws ArgumentError create_similarity_template(img;
                                            angular_granularity = 0.0)
        @test_throws ArgumentError create_similarity_template(img;
                                            angular_granularity = -0.1)
        @test_throws ArgumentError create_similarity_template(img;
                                            scale_granularity = 0.0)
        @test_throws ArgumentError create_similarity_template(img;
                                            rotation_range = (1.0, 0.0))
        @test_throws ArgumentError create_similarity_template(img;
                                            scale_range = (2.0, 1.0))
        @test_throws ArgumentError create_similarity_template(img;
                                            scale_range = (0.0, 1.0))
        @test_throws ArgumentError create_similarity_template(img;
                                            scale_range = (-1.0, 1.0))
    end

    @testset "Subtype of AbstractSearchModel" begin
        rsm = create_similarity_template(img;
                                          gradient_threshold = 1.0,
                                          origin = center)
        @test rsm isa AbstractSearchModel
    end

end
