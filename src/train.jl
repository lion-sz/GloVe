function train(
    file::CoocFile,
    n_epochs::Int;
    k::Int,
    range::Float64,
    lower::Union{Float64,Nothing} = nothing,
    b_range::Union{Float64,Nothing} = nothing,
    learning_rate::Float64,
    gpu::Bool = false,
    return_models::Bool = false,
)
    glove = GloveModel(file.N, k, Float32, range, lower, b_range)
    opt = AdaGrad(file.N, k, learning_rate, Float32)
    if gpu
        glove = cu(glove)
        opt = cu(opt)
    end

    tg, epoch_losses = train!(glove, opt, file, n_epochs)
    if return_models
        return tg, epoch_losses, cpu(glove), cpu(opt)
    else
        return tg, epoch_losses
    end
end

function train!(glove::AbstractGlove, opt::AbstractOptimizer, file::CoocFile, n_epochs::Int)

    epoch_losses = Vector{Float64}()
    for epoch = 1:n_epochs
        losses = train_epoch!(glove, opt, file)
        push!(epoch_losses, sum(losses) / length(losses))
    end

    return TrainedGlove(glove), epoch_losses
end

