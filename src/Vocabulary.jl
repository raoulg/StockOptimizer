module Vocabulary

using DataStructures
using ProgressBars
using DataFrames
using FileIO
using Dates

using ..Settings

mutable struct Vocab{D<:Union{DefaultDict{String, Int, Int}, DefaultDict{Int, Int, Int}}}
  data::D
  len::Int
  Vocab(d::DefaultDict) = new{typeof(d)}(d, length(d))
end

Base.getindex(v::Vocab, key) = v.data[key]

function build_vocab(corpus::Vector{T}) where T<:Union{String, Int}
  d = DefaultDict{T, Int, Int}(1)
  if T == String
      d["<OOV>"] = 1
  else
      d[typemax(T)] = 1
  end

  for key in unique(corpus)
      d[key] = length(d) + 1
  end
  return Vocab(d)
end

function Base.show(io::IO, v::Vocab{DefaultDict{T, Int, Int}}) where {T}
  println(io, "Vocab($T=>Int, default=$(v.data.d.default))")
end


function extract_vocabs(df::DataFrame, dataconfig::Settings.DataConfig, dir)::Dict{Symbol,Vocab}
  cols = vcat(dataconfig.dyn_cat , dataconfig.stat_cat)
  vocabs = Dict{Symbol,Vocab}()
  for col in ProgressBar(names(df[!, cols]))
    vals = collect(skipmissing(unique(df[!, col])))
    v = build_vocab(vals)
    vocabs[Symbol(col)] = v
  end
  tag = Dates.format(now(), "yyyymmdd-HHMMSS")
  vocabpath = dir / (tag * "-vocabs.jld2")
  @info "Saving vocabs to $(vocabpath.path)"
  save(vocabpath.path, Dict("vocabs" => vocabs))
  return vocabs
end

end
