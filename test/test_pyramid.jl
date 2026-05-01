# Unit tests for pyramid.jl.
# This file is supposed to be included from runtests.jl.

@testset "Pyramid" begin

    @testset "ImagePyramid — auto height" begin
        img = Float64[i + j for i in 1:32, j in 1:32]
        p = ImagePyramid(img; min_size = 8)
        # 32 → 16 → 8; next would be 4 < 8, stop. → 3 levels
        @test length(p) == 3
        @test size(p[1]) == (32, 32)
        @test size(p[2]) == (16, 16)
        @test size(p[3]) == (8, 8)
    end

    @testset "ImagePyramid — explicit n_levels" begin
        img = Float64[i + j for i in 1:32, j in 1:32]
        p = ImagePyramid(img, 4)
        @test length(p) == 4
        @test size(p[1]) == (32, 32)
        @test size(p[2]) == (16, 16)
        @test size(p[3]) == (8, 8)
        @test size(p[4]) == (4, 4)
    end

    @testset "ImagePyramid — single level when input is small" begin
        img = Float64[i + j for i in 1:6, j in 1:6]
        p = ImagePyramid(img; min_size = 8)
        # 6 ÷ 2 = 3 < 8 → don't add another level → 1 level only
        @test length(p) == 1
        @test size(p[1]) == (6, 6)
    end

    @testset "ImagePyramid — mean values are correct" begin
        img = Float64[i + j for i in 1:8, j in 1:8]
        p = ImagePyramid(img, 2)
        # Level 2, pixel (1,1) = mean of img[1:2, 1:2] = (2 + 3 + 3 + 4) / 4 = 3
        @test p[2][1, 1] == 3.0
        # Level 2, pixel (1,2) = mean of img[1:2, 3:4] = (4 + 5 + 5 + 6) / 4 = 5
        @test p[2][1, 2] == 5.0
        # Level 2, pixel (2,1) = mean of img[3:4, 1:2] = (4 + 5 + 5 + 6) / 4 = 5
        @test p[2][2, 1] == 5.0
    end

    @testset "ImagePyramid — odd sizes truncate" begin
        img = ones(7, 7)
        p = ImagePyramid(img, 2)
        # 7 ÷ 2 = 3 → level 2 is 3×3
        @test size(p[2]) == (3, 3)
        @test all(==(1.0), p[2])
    end

    @testset "ImagePyramid — Float32 input is converted" begin
        img32 = Float32[i + j for i in 1:8, j in 1:8]
        p = ImagePyramid(img32, 2)
        @test p[1] isa Matrix{Float64}
        @test p[2] isa Matrix{Float64}
    end

    @testset "ImagePyramid — Int input is converted" begin
        img = [i + j for i in 1:8, j in 1:8]   # Matrix{Int}
        p = ImagePyramid(img, 2)
        @test p[1] isa Matrix{Float64}
        @test p[2][1, 1] == 3.0
    end

    @testset "ImagePyramid — iteration" begin
        img = ones(16, 16)
        p = ImagePyramid(img, 3)
        sizes = Tuple{Int,Int}[]
        for level in p
            push!(sizes, size(level))
        end
        @test sizes == [(16, 16), (8, 8), (4, 4)]
    end

    @testset "ImagePyramid — input validation" begin
        img = ones(8, 8)
        @test_throws ArgumentError ImagePyramid(img; min_size = 0)
        @test_throws ArgumentError ImagePyramid(img; min_size = -1)
        @test_throws ArgumentError ImagePyramid(img, 0)
        @test_throws ArgumentError ImagePyramid(img, -3)
        # Cannot build 5 levels from an 8×8 image (8 → 4 → 2 → 1 → 0 fails)
        @test_throws ArgumentError ImagePyramid(img, 5)
    end

    @testset "MaskPyramid — auto height" begin
        mask = trues(32, 32)
        p = MaskPyramid(mask; min_size = 8)
        @test length(p) == 3
        @test size(p[1]) == (32, 32)
        @test size(p[2]) == (16, 16)
        @test size(p[3]) == (8, 8)
        # All-true input erodes to all-true at every level
        for level in p
            @test all(level)
        end
    end

    @testset "MaskPyramid — conservative erosion" begin
        # Mark a 2×2 corner as false. After 2× downsampling, the corresponding
        # pixel at level 2 should be false; the rest should be true.
        mask = trues(8, 8)
        mask[1:2, 1:2] .= false
        p = MaskPyramid(mask, 2)
        @test size(p[2]) == (4, 4)
        @test p[2][1, 1] == false
        @test p[2][2, 1] == true
        @test p[2][1, 2] == true
        @test p[2][2, 2] == true
    end

    @testset "MaskPyramid — partial coverage erodes away" begin
        # Only one of four source pixels is true → output pixel is false
        mask = falses(8, 8)
        mask[1, 1] = true
        p = MaskPyramid(mask, 2)
        @test p[2][1, 1] == false
        @test all(p[2] .== false)
    end

    @testset "MaskPyramid — accepts Matrix{Bool}" begin
        mask = trues(8, 8)
        p_bit  = MaskPyramid(mask, 2)
        p_bool = MaskPyramid(Matrix{Bool}(mask), 2)
        @test p_bit[1] == p_bool[1]
        @test p_bit[2] == p_bool[2]
    end

    @testset "MaskPyramid — explicit n_levels and validation" begin
        mask = trues(8, 8)
        @test_throws ArgumentError MaskPyramid(mask; min_size = 0)
        @test_throws ArgumentError MaskPyramid(mask, 0)
        @test_throws ArgumentError MaskPyramid(mask, 5)
    end

    @testset "MaskPyramid — iteration" begin
        p = MaskPyramid(trues(16, 16), 3)
        c = 0
        for level in p
            @test all(level)
            c += 1
        end
        @test c == 3
    end

    @testset "Pyramids and shape models on a parabolic bowl" begin
        # End-to-end sanity: build an image pyramid + a mask pyramid for
        # a parabolic bowl, then check that both pyramids have matching
        # heights and that the levels are consistent.
        h, w = 32, 32
        img = [(i - 16.5)^2 + (j - 16.5)^2 for i in 1:h, j in 1:w]
        mask = [(i - 16.5)^2 + (j - 16.5)^2 <= 100.0 for i in 1:h, j in 1:w]

        ip = ImagePyramid(img; min_size = 4)
        mp = MaskPyramid(mask; min_size = 4)

        @test length(ip) == length(mp)
        for k in eachindex(ip.levels)
            @test size(ip[k]) == size(mp[k])
        end
    end

end
