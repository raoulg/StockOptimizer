module Vocabulary

using DataStructures
using ProgressBars
using DataFrames
using FileIO

using ..Settings

mutable struct Vocab
    data::DefaultDict{String,Int,Int}
    len::Int
    Vocab(x::DefaultDict{String,Int,Int}) = new(x, length(x))
end

Base.getindex(v::Vocab, key::String) = v.data[key]

function build_vocab(corpus::Vector{String})::Vocab
    d = DefaultDict{String,Int}(0)
    d["<OOV>"] = 0
    for key in unique(corpus)
        d[string(key)] = length(d)
    end
    return Vocab(d)
end

isstringcol(x) = eltype(x) <: String

function extract_vocabs(df::DataFrame, config::Settings.Config)::Dict{String,Vocab}
    stringcols = findall(isstringcol, eachcol(df))
    vocabs = Dict{String,Vocab}()
    for col in ProgressBar(names(df[!, stringcols]))
        vals = unique(df[!, col])
        v = build_vocab(vals)
        vocabs[String(col)] = v
    end
    vocabpath = config.processeddir / "vocabs.jld2"
    @info "Saving vocabs to $(vocabpath.path)"
    save(vocabpath.path, Dict("vocabs" => vocabs))
    return vocabs
end

end