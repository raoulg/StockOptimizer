using StockOptimizer: filterdir, DataConfig, Data, Path, Vocab
using StockOptimizer: TFT, Abstract3DArray, Layers, Observation 
using FileIO: load
using Flux: Dense, @functor, params
using Flux: Parallel, Chain, elu, softmax, normalise  
using Flux: Embedding, LSTM, SkipConnection
using Random: rand
import Flux

dataconfig = DataConfig(
    dyn_cat = [:onpromotion],
    dyn_real = [:unit_sales, :dcoilwtico],
    stat_cat = [:family, :class, :city, :state, :type, :cluster, :perishable],
    stat_real = []
)

function load_example(dataconfig)
  files = filterdir(Path("data/processed"), "jld2")
  n = 2
  tag = split(files[n].name, ".")[1]
  @info "Loading $(files[n]) timeseries"
  gdf = load(files[n].path)[tag];

  n = 1
  @info "Loading vocabularies $(files[n])"
  tag = split(files[n].name, ".")[1]
  vocabs = load(files[n].path)["vocabs"];

  @info "Create buffered dataloader"
  items = Data.TimeSeriesItem(gdf, dataconfig)
  buff = Data.BufferedLoader(items, 32, 120);
  return buff, vocabs
end

buff, vocabs = load_example(dataconfig);
batch = buff();

# Variable Selection Network
staticvsn = TFT.VariableSelectionNetwork(
  dataconfig, vocabs, 16, static=true
);
ζ = staticvsn(batch);
size(ζ) == (16, 32)

scv = TFT.StaticCovariates(16);
c₁, c₂ , c₃, c₄ = scv(ζ); 


dynamicvsn = TFT.VariableSelectionNetwork(
  dataconfig, vocabs, 16, static=false
);

ξₜ = dynamicvsn(batch, c₁);

# Encoder Decoder


le = TFT.LocalityEnhancement(16)
size.(le.lstm.state)


state = (c₂, c₃)


encoder = TFT.Seq2seq(16)
size.(params(encoder))
x = encoder((ξₜ, state))


se = TFT.staticEnhancement(16)
Θ = se((x, c₄))






## Debug
for ps in params(staticvsn)
  println(size(ps))
end
dynamicvsn.project
size.(params(dynamicvsn.project))
size.(params(dynamicvsn.context_network))

batch[1].dynamic
x = dynamicvsn.project(batch)
x

typeof.(x)
x = dynamicvsn.reshape(x)
typeof.(x)
dynamicvsn.context_network((x[2], c))


for ps in params(dynamicvsn)
  println(size(ps))
end

for key in dataconfig.stat_cat
  println(key, " : ", vocabs[key].len)
end


## Debug


me = TFT.MultiEmbedding(cols, vocabs, 16, static=true)
x = me(batch)
resh = Layers.Split(TFT.stack_embeddings, TFT.flat_embeddings);
Ξ = resh(x);
size.(Ξ)
vars = length(cols)
dims = 16
grn = TFT.GRN(vars * dims, dims, vars, context=false)
v = TFT.create_vector(grn(Ξ[2]))

TFT.variable_select(v, Ξ[1])



dims=16
grn = TFT.GRN(length(cols) * dims, dims, dims, context=false)
using Flux
for ps in Flux.params(grn)
  println(size(ps))
end

for ps in Flux.params(dynamicvsn)
  println(size(ps))
end

for ps in Flux.params(staticvsn)
  println(size(ps))
end
c = cv(batch)

## Debugging


me = TFT.MultiEmbedding(dataconfig.dyn_cat, vocabs, 16, static=false)
prt = TFT.ProjectRealTime(dataconfig.dyn_real, 16);
pd = Parallel(vcat, me, prt)

x = pd(batch);
size.(x)

reshape_emb = Layers.Split(TFT.stack_embeddings, TFT.flat_embeddings);
dyn_vars = 4
dims = 16

grn = TFT.GRN(dyn_vars * dims, dims, dyn_vars, context = true);

x = pd(batch);
length(x) == dyn_vars
Ξ = reshape_emb(pd(batch));
# size.(Ξ)
c = rand(16, 32);
v = TFT.create_vector(grn((Ξ[2], c)));
ξₜ = TFT.variable_select(v, Ξ[1]);
size(ξₜ) == (16, 120, 32)


struct VariableSelectionNetwork
  project
  reshape::Layers.Split
  context_network::TFT.GRN
  create_vector::Function
  reduction::Function
  VariableSelectionNetwork(config::DataConfig, vocabs::Dict{Symbol, Vocab}, hidden::Int) =
    new(
      Parallel(
        vcat,
        TFT.MultiEmbedding(config.dyn_cat, vocabs, hidden, static=false),
        TFT.ProjectRealTime(config.dyn_real, hidden)
      ),
      reshape_emb,
      TFT.GRN(
        (length(dataconfig.dyn_cat) + length(dataconfig.dyn_real)) * hidden,
        hidden, 
        (length(dataconfig.dyn_cat) + length(dataconfig.dyn_real)),
        context=true), 
      TFT.create_vector,
      TFT.variable_select)
end

function (vsn::VariableSelectionNetwork)(batch, c)
  Ξ = vsn.reshape(vsn.project(batch))
  vₓ = vsn.create_vector(vsn.context_network((Ξ[2], c)))
  vsn.reduction(vₓ, Ξ[1])
end

vsn = VariableSelectionNetwork(dataconfig, vocabs, 16)
c = rand(16, 32)
ξₜ = vsn(batch, c)

## static
vocabs
dataconfig

function reshape_static(batch::Vector{Observation}, cols::Vector{Symbol})
  TFT.extract_reshape_static(batch, cols) |> Layers.splitmatrix
end


for (k, d) in vocabs
  println(typeof(d))
  println(k)
en


col = [:family, :city, :state, :type]
me = TFT.MultiEmbedding(
  String.(cols),
  16, 
  reshape_split,
  [Embedding(vocabs[string(key)].len, 16) for key in cols]
)

me(batch, cols)

## dynamic

sizesdict = Dict(k => d.len for (k, d) in vocabs)
sizesdict["perishable"] = 3
sizesdict
cols = [:onpromotion]
batch[1].dynamic


extract_time(batch, cols) = TFT.split_3D_on_first(TFT.extract_reshape_dynamic(batch, cols))
cols
me = TFT.MultiEmbedding(
  String.(cols),
  16, 
  extract_time,
  [Embedding(sizesdict[string(key)], 16) for key in cols]
)

me(batch, cols)

