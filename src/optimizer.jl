abstract type AbstractOptimizer end

struct AdaGrad{F<:AbstractFloat} <: AbstractOptimizer
    η::F
    Gw::Matrix{F}
    Gw̃::Matrix{F}
    Gb::Vector{F}
    Gb̃::Vector{F}
end

function AdaGrad(n::Int, k::Int, η::Float64, F::Type)
    return AdaGrad(
        F(η),
        fill(F(1e-3), k, n),
        fill(F(1e-3), k, n),
        fill(F(1e-3), n),
        fill(F(1e-3), n),
    )
end

struct AdaGradCuda{F<:AbstractFloat,mat<:AbstractMatrix{F},vec<:AbstractVector{F}} <:
       AbstractOptimizer
    η::F
    Gw::mat
    Gw̃::mat
    Gb::vec
    Gb̃::vec
end
Adapt.@adapt_structure AdaGradCuda

function CUDA.cu(opt::AdaGrad{F})::AdaGradCuda{F} where {F<:AbstractFloat}
    return AdaGradCuda(opt.η, cu(opt.Gw), cu(opt.Gw̃), cu(opt.Gb), cu(opt.Gb̃))
end

cpu(::AdaGrad) = x -> x

function cpu(opt::AdaGradCuda)
    return AdaGrad(opt.η, Array(opt.Gw), Array(opt.Gw̃), Array(opt.Gb), Array(opt.Gb̃))
end
