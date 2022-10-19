module Data

using Random: shuffle
using DataFrames: DataFrame, select, combine, GroupedDataFrame, AbstractDataFrame
using Base: @kwdef
export TimeSeriesItem, BufferedLoader, sizes


struct TimeSeriesItem{
  D1<:GroupedDataFrame,
  D2<:GroupedDataFrame,
  K<:Vector{Tuple{Int,Int}},
}
  dynamic::D1
  static::D2
  keys::K
end

# constructor
TimeSeriesItem(data, config) = TimeSeriesItem(
  select(data, vcat(config.dyn_cat, config.dyn_real), ungroup = false),
  combine(
    data,
    vcat(config.stat_cat, config.stat_real) .=> unique,
    ungroup = false,
    renamecols = false,
  ),
  shuffle(Tuple.(keys(data))),
)

@kwdef struct Observation{D<:AbstractDataFrame,S<:AbstractDataFrame}
  dynamic::D
  static::S
end

# getobs and numobs, see https://github.com/JuliaML/MLUtils.jl for details
getobs(ts::TimeSeriesItem, idx) =
  Observation(ts.dynamic[ts.keys[idx]], ts.static[ts.keys[idx]])
numobs(ts::TimeSeriesItem) = length(ts.keys)

# split dataframe into stepsized chunks
function split_df(ts::AbstractDataFrame, stepsize::Int)
  startidx = collect(1:stepsize:size(ts)[1])[1:end-1]
  idx = [i:i+stepsize-1 for i in startidx]
  [ts[i, :] for i in idx]
end

#iterator
Base.iterate(ts::TimeSeriesItem, state = 1) =
  state > length(ts.keys) ? nothing : (getobs(ts, state), state + 1)

# Buffer
mutable struct BufferedLoader
  ts::TimeSeriesItem
  batchsize::Int
  buffer::Vector{Observation}
  count::Int
  stepsize::Int
end

# constructor
BufferedLoader(ts::TimeSeriesItem, batchsize::Int, stepsize::Int) =
  BufferedLoader(ts, batchsize, [], length(ts.keys), stepsize)

"""
This empties the buffer
"""
function (b::BufferedLoader)()::Union{Vector{Observation},Nothing}
  # dont run if everything is consumed
  if b.count == 0
    return nothing
  end

  # if the buffer is too small
  if length(b.buffer) < b.batchsize
    # while we can consume items
    while (b.count > 0) & (length(b.buffer) < b.batchsize)
      # get and observation
      obs = getobs(b.ts, b.count)
      # split into stepsizes
      for tsx in split_df(obs.dynamic, b.stepsize)
        push!(b.buffer, Observation(tsx, obs.static))
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
    return nothing
    # out = b.buffer
  end

  return out
end

# Base.iterate(b::BufferedLoader, state=1) = b.count <= 0 ? nothing : (b(), state+1)
Base.iterate(b::BufferedLoader, state = 1) = b.count <= 0 ? nothing : (b(), state + 1)

function sizes(x, prefix = ">")
  if eltype(x) <: AbstractArray
    container = first(split(string(typeof(x)), "{"))
    println(prefix, container * ": " * string(size(x)))
    for xi in x
      sizes(xi, prefix * prefix)
    end
  else
    container = first(split(string(typeof(x)), "{"))
    println(prefix, container * ": " * string(size(x)))
  end
end

end
