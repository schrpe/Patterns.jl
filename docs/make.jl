# Build the docs from the repo root:
#   julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=docs docs/make.jl

using Documenter
using Patterns

DocMeta.setdocmeta!(
    Patterns,
    :DocTestSetup,
    :(using Patterns);
    recursive = true,
)

makedocs(
    modules  = [Patterns],
    authors  = "schrpe",
    sitename = "Patterns.jl",
    format   = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://schrpe.github.io/Patterns.jl",
    ),
    pages = [
        "Home"      => "index.md",
        "Reference" => "reference.md",
    ],
    checkdocs = :exports,
)

deploydocs(; repo="github.com/schrpe/Patterns.jl")
