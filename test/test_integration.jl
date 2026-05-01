# End-to-end integration tests across the full pipeline.
# This file is supposed to be included from runtests.jl.

@testset "Integration" begin

    # A reusable non-repetitive synthetic image: trigonometric mix gives
    # locally distinctive features that NCC, GDP and ORB can disambiguate.
    function _synthetic_image(h::Int = 100, w::Int = 100)
        return [0.5 + 0.5 * sin(0.13 * i + 0.27 * j) *
                       cos(0.31 * i - 0.19 * j)
                for i in 1:h, j in 1:w]
    end

    @testset "Translation pipeline — pyramid_search end-to-end" begin
        src = _synthetic_image(96, 96)
        # Extract a sub-region as the template
        template = src[20:30, 30:40]   # 11×11

        matches = pyramid_search(src, template;
                                  metric = :ncc,
                                  min_score = 0.95,
                                  min_size = 4)
        @test !isempty(matches)
        sorted = sort(matches, by = m -> -m.score)
        @test sorted[1].pose == (20, 30)
        @test sorted[1].score > 0.99
    end

    @testset "Multi-template pipeline — search_templates" begin
        src = _synthetic_image(120, 120)
        # Two templates from different regions — NCC should localize each
        t1 = src[20:30, 20:30]
        t2 = src[60:70, 60:70]

        results = search_templates(src, [t1, t2];
                                    metric = :ncc,
                                    min_score = 0.95,
                                    min_size = 4)
        @test length(results) == 2
        @test !isempty(results[1])
        @test !isempty(results[2])
        @test sort(results[1], by = m -> -m.score)[1].pose == (20, 20)
        @test sort(results[2], by = m -> -m.score)[1].pose == (60, 60)
    end

    @testset "Shape-based pipeline — model + search" begin
        # Parabolic bowl with a circular mask: rotation-symmetric source.
        h, w = 41, 41
        center = ((h + 1) / 2, (w + 1) / 2)
        img = [(i - center[1])^2 + (j - center[2])^2
               for i in 1:h, j in 1:w]
        mask = [sqrt((i - center[1])^2 + (j - center[2])^2) <= 8.0
                for i in 1:h, j in 1:w]

        model = create_similarity_template(img;
                                            rotation_range = (0.0, π/2),
                                            angular_granularity = π/4,
                                            gradient_threshold = 1.0,
                                            origin = center,
                                            mask = mask)
        @test length(model) == 3   # 3 angles, 1 scale

        matches = shape_search(img, model;
                                variant = :gdp,
                                min_score = 0.95,
                                suppression_radius = 5)
        @test !isempty(matches)
        sorted = sort(matches, by = m -> -m.score)
        @test abs(sorted[1].pose.r - center[1]) <= 1.0
        @test abs(sorted[1].pose.c - center[2]) <= 1.0
    end

    @testset "Keypoint pipeline — detect → match → RANSAC" begin
        # A single non-repetitive image gives distinct ORB descriptors so
        # Lowe's ratio test passes most matches; self-matching should
        # recover the identity similarity.
        img = _synthetic_image(160, 160)
        kp = detect_orb(img; n_keypoints = 80, threshold = 0.05)
        @test length(kp) >= 4

        matches = match_keypoints(kp, kp; max_distance = 80, ratio = 0.8)
        if length(matches) >= 4
            result = estimate_pose(matches;
                                    error = 2.0,
                                    min_scale = 0.95,
                                    max_scale = 1.05)
            # Identity recovered (within numerical precision)
            @test isapprox(result.pose.A, [1.0 0.0; 0.0 1.0]; atol = 0.05)
            @test isapprox(result.pose.t, [0.0, 0.0]; atol = 1.0)
            @test result.score > 0.5
            @test result.certainty == 1.0
        end
    end

    @testset "Multi-model keypoint pipeline" begin
        # Two distinct images → two distinct keypoint models → match each
        # against a "scene" image (here we use one of the two as the scene).
        img1 = _synthetic_image(120, 120)
        img2 = [0.5 + 0.5 * sin(0.21 * i - 0.19 * j) *
                       cos(0.17 * i + 0.29 * j)
                for i in 1:120, j in 1:120]

        kp1 = detect_orb(img1; n_keypoints = 50, threshold = 0.05)
        kp2 = detect_orb(img2; n_keypoints = 50, threshold = 0.05)
        scene_kp = detect_orb(img1; n_keypoints = 80, threshold = 0.05)

        per_model = match_keypoints([kp1, kp2], scene_kp;
                                     max_distance = 80, ratio = 0.8)
        @test length(per_model) == 2

        # Model 1 matches the scene (same image), model 2 should match worse
        @test length(per_model[1]) >= 1
        if length(per_model[1]) >= 4
            result = estimate_all_poses(per_model[1]; error = 2.0,
                                         min_scale = 0.95, max_scale = 1.05)
            @test !isempty(result)
        end
    end

    @testset "Subpixel refinement integration" begin
        # NCC peak refined to subpixel precision. The peak is at integer
        # coordinates (no sub-pixel offset), so the refined offsets should
        # be small and the refined score should still indicate a strong match.
        src = _synthetic_image(96, 96)
        template = src[20:30, 30:40]
        scores = ncc_match(src, template)
        peak = argmax(scores)
        @test Tuple(peak) == (20, 30)

        dx, dy, refined_score = subpixel_peak(scores, peak)
        @test abs(dx) < 1.0
        @test abs(dy) < 1.0
        # The LSQ-quadratic refined score is the value of the fitted parabola
        # at its extremum, not the discrete peak value — it can differ slightly
        # due to fit residuals. We only require that it remains a strong match.
        @test refined_score > 0.99
    end

    @testset "Module exports — sanity check" begin
        # All public symbols are accessible after `using Patterns`
        @test Pose2D isa DataType
        @test Match isa UnionAll
        @test ShapeModel isa DataType
        @test KeypointMatch isa DataType
        @test TransformResult isa DataType
        @test ShapeVariant isa DataType
        @test RotatedScaledSearchModel <: AbstractSearchModel
        @test ImagePyramid isa DataType
        @test MaskPyramid isa DataType
        @test PatternKeypoint isa DataType

        # Functions
        for fn in (apply_pose, identity_pose, similarity_pose, image_gradients,
                   sad_match, ncc_match, nccr_match,
                   gdp_match, gdpr_match, gdpr_local_match,
                   build_shape_model, transform_shape_model,
                   shape_score_gdp, shape_score_gdpr, shape_score_gdpr_local,
                   create_similarity_template, nearest_variant,
                   pyramid_search, shape_search,
                   search_templates, search_templates_pixelwise, search_shape_models,
                   detect_fast, detect_fast_scale_invariant, detect_orb,
                   match_keypoints, match_all_keypoints,
                   compute_similarity_pose, estimate_pose, estimate_all_poses,
                   subpixel_peak, refine_pose_similarity)
            @test fn isa Function
        end
    end

end
