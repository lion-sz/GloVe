struct TrainedGlove{F<:AbstractFloat}
    w::AbstractMatrix{F}
end

function TrainedGlove(glove::AbstractGlove)
    w = zeros(Float32, size(glove.w))
    copyto!(w, glove.w)
    w .+= Matrix{Float32}(glove.w̃)
    return TrainedGlove(w)
end

function save_text_file(glove::TrainedGlove, vocab_filename::String)
    vocab_file = open(vocab_filename)
    out_file = open("vectors.txt", "w")
    line = readline(vocab_file)
    i = 1
    while line != ""
        word = split(line, " ")[1]
        write(out_file, word)
        vec = join(" " .* string.(glove.w[:, i])) * "\n"
        write(out_file, vec)
        line = readline(vocab_file)
        i += 1
    end
    close(out_file)
    return i - 1
end
