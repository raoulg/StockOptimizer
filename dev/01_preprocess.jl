using StockOptimizer: Path, Load, preprocess
using StockOptimizer: Config, PreprocessConfig, DataConfig
using StockOptimizer: Preprocess
using StockOptimizer: filterdir
using DataFrames

config = Config(
    datadir=Path("data/raw"),
    processeddir=Path("data/processed"),
    suffix="parq",
    staticfiles=[Path("items.parq"), Path("stores.parq")],
    dynamicfiles=[Path("oil.parq")],
    trainfile=Path("train.parq"),
    datecol=:date,
    dynamic_map=Dict("oil" => (datecol = "date", valcol = ["dcoilwtico"])),
    col_map=Dict("items.parq" => :item_nbr, "stores.parq" => :store_nbr, "oil.parq" => :date),
    minlen=600
)

ppconfig = PreprocessConfig(
    datecol=:date,
    none_cols=[:onpromotion],
    locf_cols=[:store_nbr, :item_nbr, :family, :class, :perishable, :city, :state, :type, :cluster, :dcoilwtico],
    zero_cols=[:unit_sales],
    drop_cols=[:id],
    group_cols=[:store_nbr, :item_nbr],
    stepsize=1000,
    smoketest=2000,
)

dataconfig = DataConfig(
    dyn_cat = [:onpromotion, :perishable],
    dyn_real = [:unit_sales, :dcoilwtico],
    stat_cat = [:family, :class, :city, :state, :type, :cluster],
    stat_real = []
)

@time Preprocess.batch_and_save(config, ppconfig, dataconfig)


## Preprocess batch_and_save unpacked
files = filterdir(config.datadir, config.suffix)
df = Load.load_train(config);
first(df, 5)
df = Load.join_static(df, config)
df = Load.join_dynamic(df, config)
df = Preprocess.impute_none(df, ppconfig.none_cols)
first(df, 50)

dropmissing!(df)
find_float64(df) = findall(col -> eltype(col) <: Union{Missing, Float64}, eachcol(df))
df
cols = find_float64(df)

x = [1.0, 2.0, missing]
convert.(Float32, x)

x = convert(Vector{Union{Missing, Float32}}, x)

convertFloat32(x) = convert(Vector{Union{Missing, Float32}}, x)
transform!(df, cols .=> convertFloat32, renamecols=false)

gdf = groupby(df, ppconfig.group_cols)
gdf = filter(x -> nrow(x) > config.minlen, gdf)
k = keys(gdf)
using Random
k = shuffle(k)
z = k[1:10]


typeof(gdf[k[1]]) <: AbstractDataFrame
out = combine(gdf[z], x -> preprocess(x, ppconfig), ungroup = false)

Preprocess.find_float64(out)

subset = df[1:100, :]
dropmissing!(subset)

find_float64(df) = findall(col -> eltype(col) <: Float64, eachcol(df))

cols = find_float64(subset)
transform(subset, cols .=> x -> convert.(Float32, x), renamecols=false)


sort(df, [:store_nbr, :item_nbr])
# dropmissing!(df)

using StockOptimizer: Vocabulary
pwd()
vocabs = Vocabulary.extract_vocabs(df, dataconfig, config.processeddir)

v = vocabs[:cluster]
v.len
v.data


gdf = groupby(df, ppconfig.group_cols)
k = keys(gdf)[1:10]
out = combine(gdf[k], x -> preprocess(x, ppconfig), ungroup=false)







