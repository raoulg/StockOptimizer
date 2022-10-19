using StockOptimizer: filterdir, Path, TFT, Layers
using Flux: params, Chain


files = filterdir(Path("data/processed"), "jld2")
vocabs = load(files[end].path)["vocabs"]
statical = collect(keys(vocabs))

tag = split(files[1].name, ".")[1]
df = load(files[1].path)[tag]
names(df)

# create batch
batch = collect(transpose(Matrix(df[!, statical][1:32, :])))
size(batch) == (5, 32)

vsn = TFT.VariableSelectionNetwork(5, 16, vocabs)

Ξ = vsn(batch)
size(Ξ) == (16, 32)

grn = TFT.GRN_nocontext(16, 16)

function StaticCovariates(in, out)
    paths = [TFT.GRN_nocontext(in, out) for _ in 1:4]
    Chain(
        Layers.Split(paths)
    )
end
scv = StaticCovariates(16, 16)
c1, c2, c3, c4 = scv(Ξ) 
size(c1) == (16, 32)

using DataFrames, Parquet
using StockOptimizer: Preprocess
using Impute: Impute
using Pipe: @pipe
df = load(files[1].path)[tag]
oil = DataFrame(read_parquet("data/raw/oil.parq"))
transform!(oil, :date => ByRow(Preprocess.cast_date) => :date)

gdf = @pipe df |>
    leftjoin(_, oil, on=:date) |>
    sort(_, [:store_nbr, :item_nbr, :date]) |>
    transform(_, :dcoilwtico .=> Impute.locf .=> :dcoilwtico) |>
    groupby(_, [:store_nbr, :item_nbr])
k = keys(gdf)

ts = dropmissing(gdf[k[1]])
ts

dyn_cols = [:unit_sales, :dcoilwtico]

using Flux

xd = convert.(Float32, collect(transpose(Matrix(ts[1:90, dyn_cols]))))
x = [rand(Float32, (2, 32)) for _ in 1:90];
size(x[1]) == (2, 32)

project = Dense(2, 16)
x = project.(x)
size(x[1]) == (16, 32)


lstm = LSTM(16, 32)
Flux.reset!(lstm)

xt = project(xd)
seq = [xt[:, i] for i in 1:size(xt, 2)]
out = [lstm(xi) for xi in seq]

hcat(out...)