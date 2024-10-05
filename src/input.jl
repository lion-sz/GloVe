abstract type CoocFile end


"""Iterate over the chunks in random order"""
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


struct HDF5Cooc <: CoocFile
    file_name::String
    file::HDF5.File
    N::Int
    n_chunks::Int
    chunksize::Int
end

function HDF5Cooc(file_path::String)
    file = h5open(file_path)
    n = HDF5.read_attribute(file, "N")
    n_chunks = HDF5.get_num_chunks(file["I"])
    chunksize = Int(HDF5.get_chunk_length(file["I"]) / 4)
    return HDF5Cooc(file_path, file, n, n_chunks, chunksize)
end

Base.length(file::HDF5Cooc) = file.n_chunks
Base.show(io::IO, file::HDF5Cooc) =
    print(io, "Cooc Matrix '$(file.file_name)' ($(file.n_chunks) chunks)")
Base.close(file::HDF5Cooc) = close(file.file)


function get_chunk_size(file::HDF5Cooc, chunk::Int)
    dataset = "I"
    if chunk == file.n_chunks - 1
        # Compute the number of nonzero items.
        n = length(file.file[dataset])
        last_chunk = HDF5.get_chunk_info_all(file.file[dataset])[end]
        return n - last_chunk.offset[1]
    else
        return Int(HDF5.get_chunk_length(file.file[dataset]) / 4)
    end
end


function load_chunk(file::HDF5Cooc, chunk::Int)::Vector{CoocRec}
    n = get_chunk_size(file, chunk)

    I = reinterpret(Int32, HDF5.read_chunk(file.file["I"], chunk))
    J = reinterpret(Int32, HDF5.read_chunk(file.file["J"], chunk))
    X = reinterpret(Int32, HDF5.read_chunk(file.file["X"], chunk))

    records = Vector{CoocRec}(undef, n)
    for i = 1:n
        @inbounds records[i] = CoocRec(I[i], J[i], Float32(X[i]))
    end
    return records
end

function convert_rec_f32(infile_name::String, outfile_name::String)
    infile = open(infile_name)
    outfile = open(outfile_name, "w")
    n = Int32(filesize(infile) / 16)
    for i = 1:n
        # Write the indices.
        write(outfile, read(infile, 8))
        x = Float32(reinterpret(Float64, read(infile, 8))[1])
        write(outfile, x)
    end
    close(infile)
    close(outfile)
end


struct BinCooc <: CoocFile
    records_name::String
    vocab_name::String
    records_f::IOStream
    N::Int
    n_records::Int
    n_chunks::Int
    chunksize::Int
end

function BinCooc(
    records_file::String,
    vocab_file::String,
    F::Type = Float32,
    chunksize::Int = 10_000_000,
)
    file = open(records_file)
    n_records = Int32(filesize(file) / (3 * 4))
    n_chunks = floor(Int32, n_records / chunksize)
    cs = ceil(Int32, n_records / n_chunks)
    N = countlines(vocab_file)
    return BinCooc(records_file, vocab_file, file, N, n_records, n_chunks, cs)
end

Base.length(file::BinCooc) = file.n_chunks
Base.show(io::IO, file::BinCooc) =
    print(io, "Cooc Matrix '$(file.records_name)' ($(file.n_chunks) chunks)")
Base.close(file::BinCooc) = close(file.records_f)


function load_chunk(file::BinCooc, chunk::Int)::Vector{CoocRec}
    nb = 12
    if chunk == file.n_chunks - 1
        # The final chunk can have a different size.
        n = file.n_records - file.chunksize * (file.n_chunks - 1)
    else
        n = file.chunksize
    end
    # Set the file to the starting position of this chunk.
    # Note that chunks are 0 indexed.
    start = nb * file.chunksize * chunk
    seek(file.records_f, start)
    return reinterpret(CoocRec, read(file.records_f, file.chunksize * nb))
end
