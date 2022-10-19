using FileIO
using Base:@kwdef
using StockOptimizer: Path, filterdir
using StockOptimizer: Config
using StockOptimizer: build_vocab, Vocab
using DataFrames, Parquet
using DataStructures
using ProgressBars


config = Config(
    datadir=Path("data/raw"),
    processeddir=Path("data/processed"),
    suffix="parq",
    staticfiles=[Path("items.parq"), Path("stores.parq")],
    dynamicfiles=[Path("oil.parq")],
    trainfile=Path("train.parq"),
    dynamic_map=Dict("oil" => (datecol = "date", valcol = ["dcoilwtico"])),
    col_map=Dict("items.parq" => :item_nbr, "stores.parq" => :store_nbr),
    minlen=600
)


files = filterdir(config.processeddir, "jld2")

vocabs = load(files[end].path)["vocabs"]
vocabs

df = load(files[end-1].path)["20221003-165027"]

df = load(files[2].path)["tag"]
df

gdf = groupby(df, [:store_nbr, :item_nbr])
k = keys(gdf)
tsitem = gdf[k[2]]
for key in keys(vocabs)
    transform!(gdf, key => ByRow(x -> vocabs[key][x]) => key)
end

gdf
gdf[k[1000]]