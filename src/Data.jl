module Data

using Random: shuffle
using DataFrames: DataFrame
export TimeSeriesItem, BufferedLoader


struct TimeSeriesItem{D, T}
    data::D
    keys::T
end

# constructor
TimeSeriesItem(data) = TimeSeriesItem(data, shuffle(collect(keys(data))))

# getobs and numobs, see https://github.com/JuliaML/MLUtils.jl for details
getobs(ts::TimeSeriesItem, idx) = ts.data[ts.keys[idx]]
numobs(ts::TimeSeriesItem) = length(ts.keys)

# split dataframe into stepsized chunks
function split_ts(ts, stepsize)
    startidx = collect(1:stepsize:size(ts)[1])[1:end-1]
    idx = [i:i+stepsize-1 for i in startidx]
    [ts[i, :] for i in idx]
end

#iterator
Base.iterate(ts::TimeSeriesItem, state=1) = state > length(ts.keys) ? nothing : (getobs(ts, state), state+1)

# Buffer
mutable struct BufferedLoader
    ts::TimeSeriesItem
    batchsize::Int
    buffer::Vector{DataFrame}
    count::Int
end

# constructor
BufferedLoader(ts::TimeSeriesItem, batchsize::Int) = BufferedLoader(ts, batchsize, [], length(ts.keys))

function split_ts(ts, stepsize)
    startidx = collect(1:stepsize:size(ts)[1])[1:end-1]
    idx = [i:i+stepsize-1 for i in startidx]
    [ts[i, :] for i in idx]
end

"""
This empties the buffer
"""
function (b::BufferedLoader)()
    # dont run if everything is consumed
    if b.count == 0
        return nothing
    end

    # if the buffer is too small
    if length(b.buffer) < b.batchsize
        # while we can consume items
        while (b.count > 0) & (length(b.buffer) < b.batchsize)
            # get and observation
            ts = getobs(b.ts, b.count)
            # split into stepsizes
            for tsx in split_ts(ts, 120)
                push!(b.buffer, tsx)
            end
            # consume the count
            b.count = b.count - 1
        end
    end

    # if we need to reduce the buffer
    if length(b.buffer) >= b.batchsize
        out = b.buffer[1:b.batchsize]
        b.buffer = b.buffer[b.batchsize:end]
    else
        out = b.buffer
    end

    return out
end

Base.iterate(b::BufferedLoader, state=1) = b.count <= 0 ? nothing : (b(), state+1)

end