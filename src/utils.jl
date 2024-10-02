function cosine(glove::TrainedGlove, i::Int32, j::Int32, ::Nothing)
    d, si, sj = eltype(glove.w)(0), eltype(glove.w)(0), eltype(glove.w)(0)
    @turbo for k in axes(glove.w, 1)
        d += glove.w[k, i] * glove.w[k, j]
        si += glove.w[k, i]^2
        sj += glove.w[k, j]^2
    end
    return d / sqrt(si * sj)
end

function cosine(glove::TrainedGlove, i::I, j::I, ns::Int) where {I<:Integer}
    actual = cosine(glove, i, j, nothing)
    neg = eltype(glove.w)(0.0)
    for k = 1:ns
        neg += cosine(glove, Int32(i), Int32(k), nothing)
    end
    neg /= ns
    return actual - neg
end

function cosine(
    glove::TrainedGlove,
    I::AbstractVector{IT},
    J::AbstractVector{IT};
    ns::Union{Int,Nothing} = nothing,
) where {IT<:Integer}
    res = zeros(size(I))
    for k in eachindex(I)
        res[k] = cosine(glove, I[k], J[k], ns)
    end
    return res
end

function dist(glove::TrainedGlove, i::Int32, j::Int32, ::Nothing)
    d = eltype(glove.w)(0.0)
    @turbo for k in axes(glove.w, 1)
        d += (glove.w[k, i] - glove.w[k, j])^2
    end
    return sqrt(d)
end


function dist(glove::TrainedGlove, i::I, j::I, ns::Int) where {I<:Integer}
    actual_dist = dist(glove, i, j, nothing)
    neg_dist = eltype(glove.w)(0.0)
    for k = 1:ns
        neg_dist += dist(glove, Int32(i), Int32(k), nothing)
    end
    neg_dist = neg_dist / ns
    return actual_dist / neg_dist
end

function dist(
    glove::TrainedGlove,
    I::AbstractVector{IT},
    J::AbstractVector{IT};
    ns::Union{Int,Nothing} = nothing,
) where {IT<:Integer}
    res = zeros(size(I))
    for k in eachindex(I)
        res[k] = dist(glove, I[k], J[k], ns)
    end
    return res
end
