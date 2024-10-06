Adapt.@adapt_structure CoocRec

struct CudaGlove{F<:AbstractFloat,mat<:AbstractMatrix{F},vec<:AbstractVector{F}} <:
       AbstractGlove
    d::Int32
    k::Int32
    w::mat
    w̃::mat
    b::vec
    b̃::vec
end

Adapt.@adapt_structure CudaGlove

function CUDA.cu(model::GloveModel{F})::CudaGlove{F} where {F<:AbstractFloat}
    return CudaGlove(
        Int32(size(model.w, 2)),
        Int32(size(model.w, 1)),
        cu(model.w),
        cu(model.w̃),
        cu(model.b),
        cu(model.b̃),
    )
end

function cpu(m::CudaGlove{F}) where {F<:AbstractFloat}
    return GloveModel{F}(Array(m.w), Array(m.w̃), Array(m.b), Array(m.b̃))
end

function cu_dot(warp, a::AbstractVector, b::AbstractVector)::Float32
    k = CG.thread_rank(warp)
    acc = 0.0f0
    while k <= length(a)
        acc += a[k] * b[k]
        k += Int32(32)
    end
    # Shuffle results between the group.
    acc += CG.shfl_down(warp, acc, 16)
    acc += CG.shfl_down(warp, acc, 8)
    acc += CG.shfl_down(warp, acc, 4)
    acc += CG.shfl_down(warp, acc, 2)
    acc += CG.shfl_down(warp, acc, 1)
    acc = CG.shfl(warp, acc, 1)
    return acc
end


function cu_train_block!(
    glove::CudaGlove{F},
    opt::AdaGradCuda{F},
    crecs::AbstractVector{CoocRec},
    losses::CuDeviceMatrix{F},
) where {F<:AbstractFloat}
    # These control the outer loop.
    block_id = blockIdx().x
    n_warps = Int32(blockDim().x / 32)
    block_stride = gridDim().x * Int32(n_warps / 2)

    warp = CG.coalesced_threads()
    warp_id = floor(Int32, (threadIdx().x - 1) / 32)
    thread_id = CG.thread_rank(warp)

    shared = CuDynamicSharedArray(F, (glove.k, 2, n_warps))
    w = @view shared[:, 1i32, warp_id+1i32]
    w̃ = @view shared[:, 2i32, warp_id+1i32]

    ind = (block_id - 1i32) * Int32(n_warps / 2) + 1i32 + floor(Int32, warp_id / 2)
    @inbounds while ind <= length(crecs)
        # Odd numbered warps do the inverse step.
        crec = crecs[ind]
        i = (warp_id % 2i32) == 0i32 ? crec.i : crec.j
        j = (warp_id % 2i32) == 0i32 ? crec.j : crec.i
        x = crec.x
        # Copy to the local memory.
        k = thread_id
        while k <= glove.k
            w[k] = glove.w[k, i]
            w̃[k] = glove.w̃[k, j]
            k += 32i32
        end
        # Compute the distance
        d = cu_dot(warp, w, w̃) + glove.b[i] + glove.b̃[j] - log(F(x))
        f = min(F(x / 100i32)^F(0.75), F(1.0))
        tmp = f * d
        # Update w and w̃
        k = thread_id
        while k <= glove.k
            dw = tmp * w̃[k]
            dw̃ = tmp * w[k]
            opt.Gw[k, i] += dw^2
            opt.Gw̃[k, j] += dw̃^2
            glove.w[k, i] -= opt.η * dw / sqrt(opt.Gw[k, i])
            glove.w̃[k, j] -= opt.η * dw̃ / sqrt(opt.Gw̃[k, j])
            k += Int32(32)
        end
        # Update b and b̃. Only done by one thread in each block
        if thread_id == 1
            opt.Gb[i] += tmp^2
            opt.Gb̃[j] += tmp^2
            glove.b[i] -= opt.η * tmp / sqrt(opt.Gb[i])
            glove.b̃[j] -= opt.η * tmp / sqrt(opt.Gb̃[j])
            losses[block_id, warp_id+1] += f * d^2
        end
        ind += block_stride
    end
end


"""
This function tries to hide the time spend loading the data by using async calls.
This makes the loop a bit more complex, but might be good especially later on for
the cuda kernels.
"""
function train_epoch!(
    glove::CudaGlove{F},
    opt::AdaGradCuda{F},
    file::CoocFile;
    n_blocks::Int = 512,
) where {F<:AbstractFloat}
    losses = zeros(file.n_chunks)
    cu_losses = CUDA.zeros(F, (n_blocks, 4))
    ind = 1
    chunk, state = iterate(file)
    next_chunk = -1
    # Load the first batch
    chunks = Vector{CuVector{CoocRec}}(undef, 2)
    load_time = @elapsed begin
        chunks[1] = cu(load_chunk(file, chunk))
    end
    train_time = 0.0
    while true
        iter_res = iterate(file, state)
        if isnothing(iter_res)
            next_chunk = nothing
        else
            next_chunk, state = iter_res
            # Load the next chunk
            load_time += @elapsed t = @async begin
                chunks[2] = cu(load_chunk(file, next_chunk))
            end
        end
        # Perform the actual training.
        records = chunks[1]
        fill!(cu_losses, F(0.0))
        train_time += @elapsed CUDA.@sync begin
            @cuda shmem = glove.k * 2 * 4 * 4 threads = 128 blocks = n_blocks cu_train_block!(
                glove,
                opt,
                records,
                cu_losses,
            )
        end
        losses[ind] = sum(cu_losses) / length(records)
        # Set up next iteration.
        if isnothing(iter_res)
            break
        else
            load_time += @elapsed wait(t)
            ind += 1
            chunks[1] = chunks[2]
        end
    end
    println("Time spend training $train_time, time loading $load_time")
    return losses
end
