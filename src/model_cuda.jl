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


function cuda_train_block_single_warp!(
    glove::CudaGlove{F},
    opt::AdaGradCuda{F},
    I::CuDeviceVector{Int32},
    J::CuDeviceVector{Int32},
    X::CuDeviceVector{Int32},
    losses::CuDeviceVector{F},
) where {F<:AbstractFloat}
    thread_id = threadIdx().x
    stride = blockDim().x
    block_id = blockIdx().x
    block_stride = gridDim().x

    ind = block_id
    @inbounds while ind <= length(I)
        i = I[ind]
        j = J[ind]
        x = X[ind]
        # Compute the inner product
        inner = F(0.0)
        k = thread_id
        while k <= glove.k
            inner += glove.w[k, i] * glove.w[k, j]
            k += stride
        end
        inner += shfl_xor_sync(UInt32(0xFF), inner, 16)
        inner += shfl_xor_sync(UInt32(0xFF), inner, 8)
        inner += shfl_xor_sync(UInt32(0xFF), inner, 4)
        inner += shfl_xor_sync(UInt32(0xFF), inner, 2)
        inner += shfl_xor_sync(UInt32(0xFF), inner, 1)
        # Compute the distance
        d = inner + glove.b[i] + glove.b̃[j] - log(F(x))
        f = min(F(x / 100i32)^F(0.75), F(1.0))
        temp = F(2.0) * f * d
        # Update w and w̃
        k = thread_id
        while k <= glove.k
            opt.Gw[k, i] += (temp * glove.w̃[k, j])^2
            opt.Gw̃[k, j] += (temp * glove.w[k, i])^2
            glove.w[k, i] += temp * glove.w̃[k, j] / (sqrt(opt.Gw[k, i]) + F(1e-6)) * opt.η
            glove.w̃[k, j] += temp * glove.w[k, i] / (sqrt(opt.Gw̃[k, j]) + F(1e-6)) * opt.η
            k += stride
        end
        # Update b and b̃. Only done by one thread in each block
        if thread_id == 1
            opt.Gb[i] += temp^2
            opt.Gb̃[j] += temp^2
            glove.b[i] -= temp / (sqrt(opt.Gb[i]) + F(1e-6))
            glove.b̃[j] -= temp / (sqrt(opt.Gb̃[j]) + F(1e-6))
            losses[block_id] += f * d^2
        end
        ind += block_stride
    end
end


function cuda_train_block!(
    glove::CudaGlove{F},
    opt::AdaGradCuda{F},
    I::CuDeviceVector{Int32},
    J::CuDeviceVector{Int32},
    X::CuDeviceVector{Int32},
    losses::CuDeviceVector{F},
) where {F<:AbstractFloat}
    tb = CG.this_thread_block()
    warp = CG.coalesced_threads()

    thread_id = threadIdx().x
    stride = blockDim().x
    block_id = blockIdx().x
    block_stride = gridDim().x
    shared = CuStaticSharedArray(F, 32)
    if thread_id <= 32
        shared[thread_id] = F(0.0)
    end

    ind = block_id
    @inbounds while ind <= length(I)
        i = I[ind]
        j = J[ind]
        x = X[ind]
        # Compute the inner product
        inner = F(0.0)
        k = thread_id
        while k <= glove.k
            inner += glove.w[k, i] * glove.w[k, j]
            k += stride
        end
        inner += CG.shfl_down(warp, inner, 16)
        inner += CG.shfl_down(warp, inner, 8)
        inner += CG.shfl_down(warp, inner, 4)
        inner += CG.shfl_down(warp, inner, 2)
        inner += CG.shfl_down(warp, inner, 1)
        # Write to shared memory.
        if CG.thread_rank(warp) == 1
            shared[Int32((thread_id - 1) / 32i32)+1]
        end
        CG.sync(tb)
        # The first warp collects the results.
        if thread_id <= 32
            inner = shared[thread_id]
            inner += CG.shfl_down(warp, inner, 16)
            inner += CG.shfl_down(warp, inner, 8)
            inner += CG.shfl_down(warp, inner, 4)
            inner += CG.shfl_down(warp, inner, 2)
            inner += CG.shfl_down(warp, inner, 1)
        end
        if thread_id == 1
            shared[1] = inner
        end
        CG.sync(tb)
        # Compute the distance
        d = shared[1] + glove.b[i] + glove.b̃[j] - log(F(x))
        f = min(F(x / 100i32)^F(0.75), F(1.0))
        temp = F(2.0) * f * d
        # Update w and w̃
        k = thread_id
        while k <= glove.k
            opt.Gw[k, i] += (temp * glove.w̃[k, j])^2
            opt.Gw̃[k, j] += (temp * glove.w[k, i])^2
            glove.w[k, i] += temp * glove.w̃[k, j] / (sqrt(opt.Gw[k, i]) + F(1e-6)) * opt.η
            glove.w̃[k, j] += temp * glove.w[k, i] / (sqrt(opt.Gw̃[k, j]) + F(1e-6)) * opt.η
            k += stride
        end
        # Update b and b̃. Only done by one thread in each block
        if thread_id == 1
            opt.Gb[i] += temp^2
            opt.Gb̃[j] += temp^2
            glove.b[i] -= temp / (sqrt(opt.Gb[i]) + F(1e-6))
            glove.b̃[j] -= temp / (sqrt(opt.Gb̃[j]) + F(1e-6))
            losses[block_id] += f * d^2
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
    n_threads::Int = 64,
) where {F<:AbstractFloat}
    losses = zeros(file.n_chunks)
    cu_losses = CUDA.zeros(F, n_blocks)
    ind = 1
    chunk, state = iterate(file)
    next_chunk = -1
    # Load the first batch
    res = Vector{CuVector{Int32}}(undef, 3)
    res_next = Vector{CuVector{Int32}}(undef, 3)
    load_time = @elapsed begin
        res .= [cu(load_chunk(file, i, chunk)) for i in ["I", "J", "X"]]
    end
    while true
        iter_res = iterate(file, state)
        res_next = Vector{CuVector{Int32}}(undef, 3)
        if !isnothing(iter_res)
            next_chunk, state = iter_res
            # Load the next chunk
            load_time += @elapsed t = @async begin
                if !isnothing(next_chunk)
                    @async res_next[1] = cu(load_chunk(file, "I", next_chunk))
                    @async res_next[2] = cu(load_chunk(file, "J", next_chunk))
                    @async res_next[3] = cu(load_chunk(file, "X", next_chunk))
                end
            end
        end
        # Perform the actual training.
        I, J, X = res
        fill!(cu_losses, F(0.0))
        if n_threads == 32
            @cuda threads = 32 blocks = n_blocks cuda_train_block_single_warp!(
                glove,
                opt,
                I,
                J,
                X,
                cu_losses,
            )
            @cuda threads = 32 blocks = n_blocks cuda_train_block_single_warp!(
                glove,
                opt,
                J,
                I,
                X,
                cu_losses,
            )
        else
            @cuda threads = n_threads blocks = n_blocks cuda_train_block!(
                glove,
                opt,
                I,
                J,
                X,
                cu_losses,
            )
            @cuda threads = n_threads blocks = n_blocks cuda_train_block!(
                glove,
                opt,
                J,
                I,
                X,
                cu_losses,
            )
        end
        losses[ind] = sum(cu_losses) / length(I)
        # Set up next iteration.
        if isnothing(iter_res)
            break
        else
            load_time += @elapsed wait(t)
            ind += 1
            res .= res_next
        end
    end
    # println("Total time spend loading data $load_time")
    return losses
end
