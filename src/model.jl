abstract type AbstractGlove end

struct GloveModel{F<:AbstractFloat} <: AbstractGlove
    w::Matrix{F}
    w̃::Matrix{F}
    b::Vector{F}
    b̃::Vector{F}
end

function GloveModel(
    d::Int,
    k::Int,
    F::Type,
    range::Float64,
    lower::Union{Float64,Nothing} = nothing,
    b_range::Union{Float64,Nothing} = nothing,
)
    if isnothing(lower)
        lower = -F(range / 2)
    else
        lower = F(lower)
    end
    if isnothing(b_range)
        b_range = range
    end
    # I sample b such that the expected inner product corresponds
    # to a desired X (here set to 1).
    b_lower = (log(1) - k * (lower + range / 2)^2) / 2 - b_range / 2
    return GloveModel(
        rand(F, k, d) .* F(range) .+ lower,
        rand(F, k, d) .* F(range) .+ lower,
        rand(F, d) .* F(b_range) .+ F(b_lower),
        rand(F, d) .* F(b_range) .+ F(b_lower),
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
    # Load the first batch
    res = Vector{Vector{Int32}}(undef, 3)
    res_next = Vector{Vector{Int32}}(undef, 3)
    load_time = @elapsed begin
        res .= [load_chunk(file, i, chunk) for i in ["I", "J", "X"]]
    end
    while true
        iter_res = iterate(file, state)
        res_next = Vector{Vector{Int32}}(undef, 3)
        if !isnothing(iter_res)
            next_chunk, state = iter_res
            # Load the next chunk
            load_time += @elapsed t = @async begin
                if !isnothing(next_chunk)
                    @async res_next[1] = load_chunk(file, "I", next_chunk)
                    @async res_next[2] = load_chunk(file, "J", next_chunk)
                    @async res_next[3] = load_chunk(file, "X", next_chunk)
                end
            end
        end
        # Perform the actual training.
        loss = 0.0
        I, J, X = res
        @batch reduction = ((+, loss)) for i in eachindex(res[1])
            loss += step!(glove, opt, I[i], J[i], X[i])
            loss += step!(glove, opt, J[i], I[i], X[i])
        end
        losses[ind] = loss / length(res[1])
        # Set up next iteration.
        if isnothing(iter_res)
            break
        else
            load_time += @elapsed wait(t)
            ind += 1
            res .= res_next
        end
    end
    println("Total time spend loading data $load_time")
    return losses
end


function step!(
    glove::GloveModel{F},
    opt::AdaGrad{F},
    i::Int32,
    j::Int32,
    x::Int32,
)::F where {F<:AbstractFloat}
    # loss computation
    w = @view glove.w[:, i]
    w̃ = @view glove.w̃[:, j]
    #d = (dot(w, w̃) + glove.b[i] + glove.b̃[j] - log(F(x)))
    d = glove.b[i] + glove.b̃[j] - log(F(x))
    @turbo for k in eachindex(w)
        d += w[k] * w̃[k]
    end
    f = min(F((x / 100)^0.75), F(1.0))
    # Update the running sums.
    tmp = (F(2) * f * d)
    @turbo for k in eachindex(w)
        tw = w[k]
        tw̃ = w̃[k]
        opt.Gw[k, i] += (tmp * tw̃)^2
        opt.Gw̃[k, j] += (tmp * tw)^2
        w[k] = tw - tmp * tw̃ * opt.η / sqrt(opt.Gw[k, i])
        w̃[k] = tw̃ - tmp * tw * opt.η / sqrt(opt.Gw̃[k, j])
    end
    opt.Gb[i] += tmp^2
    opt.Gb̃[j] += tmp^2
    glove.b[i] -= tmp * opt.η / sqrt(opt.Gb[i])
    glove.b̃[j] -= tmp * opt.η / sqrt(opt.Gb̃[j])
    return f * d^2
end
