module TFT

using Flux: Parallel, sigmoid, Chain, Dense, elu, @functor, Embedding, softmax
using ..Vocabulary: Vocab
using ..Layers


function GLU(in, out)
    Chain(
        Parallel(Layers.Hadamard, Dense(in => out, sigmoid), Dense(in => out)),
    )
end

function GRN_context(in, out)
    Chain(
        Parallel(+, Dense(in => in, bias=false), Dense(in => in, bias=true)),
        elu,
        GLU(in, out)
    )
end

function GRN_nocontext(in, out)
    Chain(
        Dense(in => in),
        elu,
        GLU(in, out)
    )
end

# MultiEmbedding
struct MultiEmbedding{T,F}
    statical::Vector{String}
    vocabs::Dict{String,Vocab}
    out::Int
    paths::T
    combine::F
end

# constructor
MultiEmbedding(vocabs, out) = MultiEmbedding(
    collect(keys(vocabs)), 
    vocabs, 
    out, 
    [Embedding(vocabs[key].len, out) for key in collect(keys(vocabs))], 
    vcat)

@functor MultiEmbedding

# forward pass
(m::MultiEmbedding)(xs::Matrix) = m.combine(map((f, x) -> f(x), m.paths, Layers.splitmatrix(xs))...)

# project ME
function MultiProject(vocabs, out)
    hidden = length(vocabs) * out
    me = MultiEmbedding(vocabs, out)

    model = Chain(
        me,
        Dense(hidden => out)
    )
    return model
end

# Variable Selection
struct VariableSelection
    n_cat::Int
    d::Int
end

# forward pass
function (vs::VariableSelection)(v, Ξ)
    vₓ = reshape(v, (vs.n_cat, 1, size(v)[end]))
    ξ = reshape(Ξ, (vs.n_cat, vs.d, size(Ξ)[end]))
    dropdims(sum(vₓ.*ξ, dims=1); dims=1)
end

# overloading for tuples
(vs::VariableSelection)(xs::Tuple) = (vs::VariableSelection)(xs[1], xs[2])

function VariableVector(n_cat, d)
    Chain(
        GRN_nocontext(n_cat*d, n_cat),
        softmax
    )
end

function VariableSelectionNetwork(n_cat, d, vocabs)
    Chain(
        MultiEmbedding(vocabs, d),
        Layers.Split(VariableVector(n_cat, d), identity),
        VariableSelection(n_cat, d)
    )
end

end

