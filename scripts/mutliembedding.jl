using DataFrames, FileIO
using StockOptimizer: filterdir, Path
using Flux: elu, Dense, Chain, Parallel, sigmoid
using StockOptimizer: Vocab, TFT, Layers

using Random


# load files
files = filterdir(Path("data/processed"), "jld2")
vocabs = load(files[end].path)["vocabs"]
statical = collect(keys(vocabs))

tag = split(files[1].name, ".")[1]
gdf = load(files[1].path)[tag]

k = first(keys(gdf))
df = gdf[k]

# create batch
batch = collect(transpose(Matrix(df[!, statical][1:32, :])))
batch
# test split
x = Layers.splitmatrix(batch)
size(x) == (5, )

# test MultiEmbedding
me = TFT.MultiEmbedding(vocabs, 16)
Ξ = me(batch)
size(Ξ) == (80, 32)

vv = TFT.VariableVector(5, 16)
v = vv(Ξ)
size(v) == (5, 32)

vs = TFT.VariableSelection(5, 16)
size(vs(v, Ξ)) == (16, 32)

model = Chain(
    TFT.MultiEmbedding(vocabs, 16),
    Layers.Split(TFT.VariableVector(5, 16), identity),
    TFT.VariableSelection(5, 16)
)
b = model(batch)
size(b) == (16, 32)

using Flux: params
ps = params(model)
for p in ps
    print(size(p))
end







