using Transformers: Basic
import Flux


Θ = rand(16, 120, 32);

mha = Basic.MultiheadAttention(4, 16, 16, 16, future=false, pdrop=0.1)
glu = TFT.GLU(16, 16)
fut = Θ[:, 91:end, :]

B = mha(fut, Θ, Θ)
Flux.normalise(fut + glu(B))

struct SkippedAttention
    mha
    glu
    past
end

Flux.@functor SkippedAttention

function SkippedAttention(heads::Int, dims::Int, drop::Float64, past::Int) 
    mha = Basic.MultiheadAttention(heads, dims, dims, dims, future=false, pdrop=drop)
    glu = TFT.GLU(dims, dims)
    SkippedAttention(mha, glu, past)
end

function (sa::SkippedAttention)(Θ::AbstractArray)
    fut = Θ[:, sa.past+1:end, :]
    Flux.normalise(fut + sa.glu(sa.mha(fut, Θ, Θ)))
end

sa = SkippedAttention(4, 16, 0.1, 90)
δ = sa(Θ)


FFL(dims) = TFT.GRN(dims, dims, dims, context=false)

struct SkippedFFL
    nn
    glu
    SkippedFFL(dims::Int) = new(FFL(dims), TFT.GLU(dims, dims))
end

Flux.@functor SkippedFFL

(sf::SkippedFFL)(δ::AbstractArray) = Flux.normalise(δ + sf.glu(sf.nn(δ)))

sffl = SkippedFFL(16)

ψ = sffl(δ)


