# Unit test suite for the Patterns.jl package.

using Patterns
using Test

@testset "Patterns" begin

    include("test_types.jl")
    include("test_subpixel.jl")
    include("test_template_matching.jl")
    include("test_gradient_matching.jl")
    include("test_shape_based.jl")
    include("test_similarity_search.jl")
    include("test_pyramid.jl")
    include("test_pyramid_search.jl")
    include("test_multi_model.jl")
    include("test_keypoint.jl")
    include("test_keypoint_match.jl")
    include("test_ransac.jl")
    include("test_integration.jl")

end
