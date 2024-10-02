struct CoocFile
    file_name::String
    file::HDF5.File
    N::Int
    n_chunks::Int
    chunksize::Int
end

function CoocFile(file_path::String)
    file = h5open(file_path)
    n = HDF5.read_attribute(file, "N")
    n_chunks = HDF5.get_num_chunks(file["I"])
    chunksize = Int(HDF5.get_chunk_length(file["I"]) / 4)
    return CoocFile(file_path, file, n, n_chunks, chunksize)
end

Base.show(io::IO, file::CoocFile) =
    print(io, "Cooc Matrix '$(file.file_name)' ($(file.n_chunks) chunks)")
Base.length(file::CoocFile) = file.n_chunks
Base.getindex(file::CoocFile, dataset::String) = file.file[dataset]


function Base.close(file::CoocFile)
    close(file.file)
    return
end

function Base.iterate(cooc::CoocFile)
    order = shuffle(0:(cooc.n_chunks-1))
    return (order[1], (1, order))
end

function Base.iterate(cooc::CoocFile, state)
    pos, order = state
    if pos < length(order)
        return (order[pos+1], (pos + 1, order))
    else
        return nothing
    end
end

function get_chunk_size(file::CoocFile, dataset::String, chunk::Int)
    if chunk == file.n_chunks - 1
        # Compute the number of nonzero items.
        n = length(file[dataset])
        last_chunk = HDF5.get_chunk_info_all(file[dataset])[end]
        return n - last_chunk.offset[1]
    else
        return Int(HDF5.get_chunk_length(file[dataset]) / 4)
    end
end

function load_chunk(file::CoocFile, dataset::String, chunk::Int)
    n = get_chunk_size(file, dataset, chunk)
    res = zeros(Int32, n)
    copyto!(res, file, dataset, chunk)
    return res
end

function Base.copyto!(
    dest::AbstractVector{Int32},
    file::CoocFile,
    dataset::String,
    chunk::Int,
)
    # Copy the data
    n = get_chunk_size(file, dataset, chunk)
    bytes = HDF5.read_chunk(file[dataset], chunk)
    copyto!(dest, @view reinterpret(Int32, bytes)[1:n])
    return n
end
