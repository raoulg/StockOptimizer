module TFT

using Flux: Parallel, sigmoid, Chain, Dense, elu, @functor, Embedding, softmax, SkipConnection, normalise, identity
import Flux
using ..Vocabulary: Vocab
using ..Layers
using ..Layers: Abstract3DArray
using ..Data: Observation
using ..Settings: DataConfig

expand1(x) = reshape(x, (1, size(x)...))
expand2(x) = reshape(x, (size(x, 1), 1, size(x)[2:end]...))
stack_embeddings(x) = vcat(map(expand1, x)...);
flat_embeddings(x) = vcat(x...)

create_vector(x) = softmax(expand2(x), dims=1); # selection vector
variable_select(v, ξ) = dropdims(sum(v .* ξ, dims = 1); dims = 1)

# projection helpers
split_dyn_cat(x::Abstract3DArray) = [x[i, :, :] for i in axes(x)[1]]
split_real(x::Abstract3DArray) = [x[i:i, :, :] for i in axes(x)[1]]
preprocesscat(batch::Vector{Observation}, cols::Vector{Symbol}) = split_dyn_cat(extract_reshape_dynamic(batch, cols))
densepaths(in::Int, out::Int) = [Dense(1, out) for _ = 1:in]

function extract_static(batch::Vector{Observation}, cols::Vector{Symbol})
  hcat([Matrix(batch[i].static[:, cols])' for i in eachindex(batch)]...)
end

split_stat_cat(x::Matrix) = [x[i, :] for i in axes(x)[1]]
preprocess_stat(batch::Vector{Observation}, cols::Vector{Symbol}) = split_stat_cat(extract_static(batch, cols))

"""
Extract the dynamic part, defined by a list of column names, from a Vector of Observations.
The dynamic part is transposed to have dimensions n_categories x timesteps.

Finally, this is reshaped into n_categories x timesteps x batchsize
usage:
```jldoctest
julia> buff = Data.BufferedLoader(items, 32)
julia> batch = buff()
julia> real_cols = [:unit_sales, :dcoilwtico]
julia> x_dyn = TFT.extract_reshape_dynamic(batch, real_cols)
julia> size(x_dyn) == (2, 120, 32)
true
```

"""
function extract_reshape_dynamic(batch::Vector{Observation}, cols::Vector{Symbol}, shape::Tuple)::Abstract3DArray
  m = [transpose(Matrix(v.dynamic[:, cols])) for v in batch]
  reshape(reduce(hcat, m), shape)
end

getshape(batch::Vector{Observation}, cols) = (length(cols), size(first(batch).dynamic, 1), length(batch))

extract_reshape_dynamic(batch::Vector{Observation}, cols) = extract_reshape_dynamic(batch, cols, getshape(batch, cols))

function GLU(in, out)
  Chain(Flux.Parallel(Layers.Hadamard, Dense(in => out, sigmoid), Dense(in => out)))
end

# Layers
## GRN

struct GRN
  model
  project
end

@functor GRN

_grnContext(in::Int, hidden::Int, out::Int) = Chain(
  Parallel(
    .+,
    Dense(in, hidden, bias = false),
    Chain(Dense(hidden => hidden), expand2),
  ),
  elu,
  TFT.GLU(hidden, out),
)

_grnNoContext(in, hidden, out) = Chain(Dense(in => hidden), elu, TFT.GLU(hidden, out))

function GRN(in::Int, hidden::Int, out::Int ;context::Bool) 
  context ? model = TFT._grnContext(in, hidden, out) : model = TFT._grnNoContext(in, hidden, out)
  (in == out) ? project = identity : project = Dense(in, out)
  GRN(model, project)
  # context ? skip = (mx, (x, _)) -> project(x) .+ mx : skip = (mx, x) -> project(x) .+ mx
end

(g::GRN)(x::Tuple{AbstractArray, Matrix}) = normalise(g.project(x[1]) .+ g.model(x))
(g::GRN)(x::AbstractArray) = normalise(g.project(x) .+ g.model(x))

# GRN(in::Int, hidden::Int, out::Int; context::Bool) =
#   context ? GRN(in, hidden, out, _grnContext(in, hidden, out)) : GRN(in, hidden, out, _grnNoContext(in, hidden, out))

# (g::GRN)(xs) = g.path(xs)

## MultiEmbedding
struct MultiEmbedding{F, N}
  cat_cols::Vector{Symbol}
  static::Bool
  dims::Int
  preprocess::F
  paths::N
end

@functor MultiEmbedding

function MultiEmbedding(cat_cols::Vector{Symbol}, vocabs::Dict{Symbol, Vocab}, dims::Int; static::Bool)
  static ? preprocess = preprocess_stat : preprocess = preprocesscat
  MultiEmbedding(cat_cols, static, dims, preprocess, 
    [Embedding(vocabs[key].len, dims) for key in cat_cols]
    )
end

(m::MultiEmbedding)(batch::Vector{Observation}) =
  map((f, x) -> f(x), m.paths, m.preprocess(batch, m.cat_cols))

## Projection of real values
struct ProjectRealTime
  real_cols::Vector{Symbol}
  paths::Vector{Dense}
end

@functor ProjectRealTime

# constructor
ProjectRealTime(real_cols::Vector{Symbol}, dims::Int) =
  ProjectRealTime(real_cols, densepaths(length(real_cols), dims))

# forward pass
function (p::ProjectRealTime)(batch::Vector{Observation})
  map(|>, split_real(extract_reshape_dynamic(batch, p.real_cols)), p.paths)
end


# dynamic Variable selection
struct VariableSelectionNetwork
  project
  reshape::Layers.Split
  context_network
  create_vector::Function
  reduction::Function
end

@functor VariableSelectionNetwork

function dynamic_vsn(dataconfig::DataConfig, vocabs::Dict{Symbol, Vocab}, hidden::Int)
    VariableSelectionNetwork(
      Parallel(
        vcat,
        TFT.MultiEmbedding(dataconfig.dyn_cat, vocabs, hidden, static=false),
        TFT.ProjectRealTime(dataconfig.dyn_real, hidden)
      ),
      Layers.Split(TFT.stack_embeddings, TFT.flat_embeddings),
      TFT.GRN(
        (length(dataconfig.dyn_cat) + length(dataconfig.dyn_real)) * hidden,
        hidden, 
        (length(dataconfig.dyn_cat) + length(dataconfig.dyn_real)),
        context=true), 
      TFT.create_vector,
      TFT.variable_select)
end


function static_vsn(dataconfig::DataConfig, vocabs::Dict{Symbol, Vocab}, hidden::Int)
    cols = vcat(dataconfig.stat_cat)
    vars = length(cols)
    VariableSelectionNetwork(
      TFT.MultiEmbedding(cols, vocabs, hidden, static=true),
      Layers.Split(TFT.stack_embeddings, TFT.flat_embeddings),
      TFT.GRN(vars * hidden, hidden, vars, context=false), 
      TFT.create_vector,
      TFT.variable_select)
end

function VariableSelectionNetwork(dataconfig::DataConfig, vocabs::Dict{Symbol, Vocab}, hidden::Int; static::Bool)
  static ? static_vsn(dataconfig, vocabs, hidden) : dynamic_vsn(dataconfig, vocabs, hidden)
end

function (vsn::VariableSelectionNetwork)(batch::Vector{Observation}, c::Matrix)
  Ξ = vsn.reshape(vsn.project(batch))
  vₓ = vsn.create_vector(vsn.context_network((Ξ[2], c)))
  vsn.reduction(vₓ, Ξ[1])
end

function (vsn::VariableSelectionNetwork)(batch::Vector{Observation})
  Ξ = vsn.reshape(vsn.project(batch))
  vₓ = vsn.create_vector(vsn.context_network((Ξ[2])))
  vsn.reduction(vₓ, Ξ[1])
end

function StaticCovariates(dims)
  paths = [TFT.GRN(dims, dims, dims, context=false) for _ in 1:4]
  Chain(
      Layers.Split(paths)
  )
end

mutable struct LocalityEnhancement lstm::Flux.Recur
  glu::Flux.Chain
end

@functor LocalityEnhancement

LocalityEnhancement(dims) = LocalityEnhancement(
  Flux.LSTM(dims, dims),
  TFT.GLU(dims, dims)
)

timecast(ξₜ)  = [ξₜ[:, i, :] for i in axes(ξₜ)[2]]
uncast(x) = reshape(vcat(x...), (size(x[1], 1), length(x), size(x[1], 2)))
# 1. calling LocalityEnhancement will call ϕ(m, ξ, c)
# 2. calling ϕ(m, ξ, c) will initialize the lstm.state with c
# 3. and continu to call ϕ(m, ξ)
# 4. calling ϕ(m, ξ) will run the lstm on the timeseries ξ
(m::LocalityEnhancement)(x::Tuple) = m(x[1], x[2])
(m::LocalityEnhancement)(ξₜ::AbstractArray, state::Tuple{Matrix, Matrix}) = m.glu(ϕ(m, ξₜ,state))
function ϕ(m::LocalityEnhancement, ξₜ::AbstractArray{T, 3}, state::Tuple{Matrix, Matrix}) where {T}
  m.lstm.state = state
  ϕ(m, ξₜ)
end

ϕ(m::LocalityEnhancement, ξₜ) = uncast([m.lstm(x) for x in timecast(ξₜ)])

function Seq2seq(dims)
  model = TFT.LocalityEnhancement(dims)
  SkipConnection(model, (mx, (x,_)) -> normalise(mx + x))
end

staticEnhancement(dims) = TFT.GRN(dims, dims, dims, context=true)

end
