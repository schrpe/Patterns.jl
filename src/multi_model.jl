#= ------------------------------------------------------------------------

    Mehrmodell-Suche

    Three top-level pipelines for searching multiple distinct models in
    one source image:

    `search_templates`           — for each template, run a separate
                                    `pyramid_search`. Returns
                                    `Vector{Vector{Match}}` (one match
                                    list per template).

    `search_templates_pixelwise` — pixelwise max-score consolidation
                                    over multiple templates of the same
                                    size. Wraps the existing
                                    `*_match_pixelwise` family with a
                                    unified `metric` keyword. Returns
                                    `(max_scores, best_index)`.

    `search_shape_models`        — for each `RotatedScaledSearchModel`,
                                    run a separate `shape_search`.
                                    Returns `Vector{Vector{Match}}`.

    The per-model variants build their own internal pyramids; an
    optimization to share the source pyramid across templates is left
    to a future refactor.

------------------------------------------------------------------------ =#

export search_templates, search_templates_pixelwise, search_shape_models


# ------------------------------------------------------------------------
# search_templates — per-model pyramid search
# ------------------------------------------------------------------------

"""
    search_templates(source::AbstractMatrix,
                     templates::AbstractVector{<:AbstractMatrix};
                     metric::Symbol = :ncc,
                     masks = nothing,
                     min_score::Real = 0.5,
                     min_score_adjust::Real = 0.9,
                     min_size::Int = 8,
                     stop_at_level::Int = 1,
                     padding::Int = 2,
                     gradient_threshold::Real = 0.0)
        -> Vector{Vector{Match{Tuple{Int,Int}}}}

Run `pyramid_search` for each template in `templates`, returning a vector
of match lists — one per template, in the same order. `masks`, if given,
must be a vector of the same length as `templates`; each entry may be
`nothing` or a Bool matrix the size of the corresponding template.

All other keyword arguments are forwarded to `pyramid_search` (one of
`:ncc`, `:nccr`, `:gdp`, `:gdpr`, `:gdpr_local` for `metric`).
"""
function search_templates(source::AbstractMatrix,
                          templates::AbstractVector{<:AbstractMatrix};
                          metric::Symbol = :ncc,
                          masks = nothing,
                          min_score::Real = 0.5,
                          min_score_adjust::Real = 0.9,
                          min_size::Int = 8,
                          stop_at_level::Int = 1,
                          padding::Int = 2,
                          gradient_threshold::Real = 0.0)
    isempty(templates) && throw(ArgumentError("templates must be non-empty"))
    if masks !== nothing
        length(masks) == length(templates) ||
            throw(ArgumentError("masks must have the same length as templates"))
    end

    return [
        pyramid_search(source, templates[k];
                       metric = metric,
                       mask = masks === nothing ? nothing : masks[k],
                       min_score = min_score,
                       min_score_adjust = min_score_adjust,
                       min_size = min_size,
                       stop_at_level = stop_at_level,
                       padding = padding,
                       gradient_threshold = gradient_threshold)
        for k in eachindex(templates)
    ]
end


# ------------------------------------------------------------------------
# search_templates_pixelwise — unified dispatcher over *_match_pixelwise
# ------------------------------------------------------------------------

"""
    search_templates_pixelwise(source::AbstractMatrix,
                               templates::AbstractVector{<:AbstractMatrix};
                               metric::Symbol = :ncc,
                               masks = nothing,
                               max_score::Real = Inf,
                               gradient_threshold::Real = 0.0)
        -> (scores::Matrix{Float64}, best_index::Matrix{Int})

Pixel-wise consolidated multi-template search. For each pixel of the
score image, returns the best score across all templates and the
1-based index of the template that achieved it.

All templates must have the same size. The interpretation of "best"
depends on the metric:

- `:sad`        — minimum SAD; uses `max_score` for early termination
- `:ncc`        — maximum NCC
- `:nccr`       — maximum NCCR
- `:gdp`        — maximum GDP (uses `gradient_threshold`)
- `:gdpr`       — maximum GDPR (uses `gradient_threshold`)
- `:gdpr_local` — maximum GDPR-local (uses `gradient_threshold`)

This wraps the corresponding `*_match_pixelwise` function. No pyramid
acceleration is applied.
"""
function search_templates_pixelwise(source::AbstractMatrix,
                                     templates::AbstractVector{<:AbstractMatrix};
                                     metric::Symbol = :ncc,
                                     masks = nothing,
                                     max_score::Real = Inf,
                                     gradient_threshold::Real = 0.0)
    isempty(templates) && throw(ArgumentError("templates must be non-empty"))

    if metric === :sad
        return sad_match_pixelwise(source, templates;
                                    masks = masks, max_score = max_score)
    elseif metric === :ncc
        return ncc_match_pixelwise(source, templates; masks = masks)
    elseif metric === :nccr
        return nccr_match_pixelwise(source, templates; masks = masks)
    elseif metric === :gdp
        return gdp_match_pixelwise(source, templates;
                                    masks = masks,
                                    gradient_threshold = gradient_threshold)
    elseif metric === :gdpr
        return gdpr_match_pixelwise(source, templates;
                                     masks = masks,
                                     gradient_threshold = gradient_threshold)
    elseif metric === :gdpr_local
        return gdpr_local_match_pixelwise(source, templates;
                                            masks = masks,
                                            gradient_threshold = gradient_threshold)
    end
    throw(ArgumentError("unsupported metric: $metric " *
                        "(use :sad, :ncc, :nccr, :gdp, :gdpr, :gdpr_local)"))
end


# ------------------------------------------------------------------------
# search_shape_models — per-model shape search
# ------------------------------------------------------------------------

"""
    search_shape_models(source::AbstractMatrix,
                        models::AbstractVector{<:RotatedScaledSearchModel};
                        variant::Symbol = :gdp,
                        min_score::Real = 0.5,
                        min_source_gradient::Real = 0.0,
                        suppression_radius::Int = -1)
        -> Vector{Vector{Match{NamedTuple{(:r, :c, :θ, :scale), …}}}}

Run `shape_search` for each `RotatedScaledSearchModel` in `models`,
returning one match list per model. All keyword arguments are forwarded
to `shape_search`.
"""
function search_shape_models(source::AbstractMatrix,
                              models::AbstractVector{<:RotatedScaledSearchModel};
                              variant::Symbol = :gdp,
                              min_score::Real = 0.5,
                              min_source_gradient::Real = 0.0,
                              suppression_radius::Int = -1)
    isempty(models) && throw(ArgumentError("models must be non-empty"))
    return [
        shape_search(source, m;
                     variant = variant,
                     min_score = min_score,
                     min_source_gradient = min_source_gradient,
                     suppression_radius = suppression_radius)
        for m in models
    ]
end
