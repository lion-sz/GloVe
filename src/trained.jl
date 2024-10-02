struct TrainedGlove{F<:AbstractFloat}
    w::AbstractMatrix{F}
end

function TrainedGlove(glove::AbstractGlove)
    w = zeros(Float32, size(glove.w))
    copyto!(w, glove.w)
    w .+= Matrix{Float32}(glove.w̃)
    return TrainedGlove(w)
end
