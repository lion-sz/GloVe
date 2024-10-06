struct CoocRec
    i::Int32
    j::Int32
    x::Float32
end


abstract type AbstractGlove end

struct GloveModel{F<:AbstractFloat} <: AbstractGlove
    w::Matrix{F}
    w̃::Matrix{F}
    b::Vector{F}
    b̃::Vector{F}
end

function GloveModel(d::Int, k::Int, F::Type, range::Union{Float64,Nothing})
    if isnothing(range)
        range = F(1 / k)
    end
    return GloveModel(
        rand(F, k, d) .* F(range) .- F(range / 2),
        rand(F, k, d) .* F(range) .- F(range / 2),
        rand(F, d) .* F(range) .- F(range / 2),
        rand(F, d) .* F(range) .- F(range / 2),
    )
end

cpu(::GloveModel) = x -> x

"""
This function tries to hide the time spend loading the data by using async calls.
This makes the loop a bit more complex, but might be good especially later on for
the cuda kernels.
"""
function train_epoch!(glove::GloveModel, opt::AdaGrad, file::CoocFile)
    losses = Vector{Float64}(undef, file.n_chunks)
    ind = 1
    chunk, state = iterate(file)
    next_chunk = -1
    # The first position is the current chunk, 
    # The upcoming chunk is stored in the second position.
    chunks = Vector{Vector{CoocRec}}(undef, 2)
    load_time = @elapsed begin
        chunks[1] = load_chunk(file, chunk)
    end
    while true
        iter_res = iterate(file, state)
        if isnothing(iter_res)
            next_chunk = nothing
        else
            next_chunk, state = iter_res
            # Load the next chunk
            t = @async begin
                chunks[2] = load_chunk(file, next_chunk)
            end
        end
        # Perform the actual training.
        loss = 0.0
        records = chunks[1]
        @batch reduction = ((+, loss)) for i in eachindex(records)
            @inbounds loss += step!(glove, opt, records[i], false)
            @inbounds loss += step!(glove, opt, records[i], true)
        end
        losses[ind] = loss / length(records)
        # Set up next iteration.
        if isnothing(iter_res)
            break
        else
            load_time += @elapsed wait(t)
            ind += 1
            chunks[1] = chunks[2]
        end
    end
    println("Total time spend loading data $load_time")
    return losses
end


function step!(
    glove::GloveModel{F},
    opt::AdaGrad{F},
    crec::CoocRec,
    inverse::Bool,
)::F where {F<:AbstractFloat}
    i = inverse ? crec.j : crec.i
    j = inverse ? crec.i : crec.j
    # loss computation
    w = @view glove.w[:, i]
    w̃ = @view glove.w̃[:, j]
    d = glove.b[i] + glove.b̃[j] - log(F(crec.x))
    @turbo for k in eachindex(w)
        d += w[k] * w̃[k]
    end
    f = min(F((crec.x / 100)^0.75), F(1.0))
    # Update the running sums.
    tmp = f * d
    @turbo for k in eachindex(w)
        dw = tmp * w̃[k]
        dw̃ = tmp * w[k]
        opt.Gw[k, i] += dw^2
        opt.Gw̃[k, j] += dw̃^2
        w[k] -= dw * opt.η / sqrt(opt.Gw[k, i])
        w̃[k] -= dw̃ * opt.η / sqrt(opt.Gw̃[k, j])
    end
    opt.Gb[i] += tmp^2
    opt.Gb̃[j] += tmp^2
    glove.b[i] -= opt.η * tmp / sqrt(opt.Gb[i])
    glove.b̃[j] -= opt.η * tmp / sqrt(opt.Gb̃[j])
    return f * d^2
end
