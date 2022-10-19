module Preprocess


using Dates, DataFrames
using FileIO
using Impute: Impute
using Pipe: @pipe
using ProgressBars
using Random

using ..Load
using ..Vocabulary
using ..Settings: Config, PreprocessConfig, DataConfig

export preprocess

function stringify_na(x)
  ismissing(x) ? "none" : string(x)
end

function impute_none(df::DataFrame, cols::Vector{Symbol})
  return transform(df, cols .=> ByRow(stringify_na) .=> cols)
end

function fillnawithzero(df::DataFrame, col::Vector{Symbol})
  transform(df, col .=> ByRow(x -> ismissing(x) ? 0.0 : x) .=> col)
end

find_float64(df) = findall(col -> eltype(col) <: Union{Missing, Float64}, eachcol(df))
convertFloat32(x) = convert(Vector{Float32}, x)

function preprocess(df::AbstractDataFrame, config::PreprocessConfig)::AbstractDataFrame
  allDates = first(df[!, config.datecol]):Dates.Day(1):last(df[!, config.datecol])
  idx = vcat(config.group_cols, config.datecol)
  out = @pipe DataFrame(date = allDates) |>
        leftjoin(_, df, on = config.datecol) |>
        sort(_, idx) |>
        transform(_, config.locf_cols .=> Impute.locf .=> config.locf_cols) |>
        fillnawithzero(_, config.zero_cols) |>
        impute_none(_, config.none_cols) |>
        _[!, Not(config.drop_cols)] |>
        dropmissing(_) |>
        sort(_, idx)


  return out
end


function batch_and_save(config::Config, ppconfig::PreprocessConfig, dataconfig::DataConfig)
  @info "Loading data"
  df = Load.load_train(config)
  df = Load.join_static(df, config)
  df = Load.join_dynamic(df, config)
  df = impute_none(df, ppconfig.none_cols)
  @info "Extracting vocabularies."
  vocabs = Vocabulary.extract_vocabs(df, dataconfig, config.processeddir)


  @info "Grouping data"
  gdf = groupby(df, ppconfig.group_cols)
  gdf = filter(x -> nrow(x) > config.minlen, gdf)
  k = keys(gdf)
  k = shuffle(k)
  stepsize = ppconfig.stepsize
  if ppconfig.smoketest > 1
    @info "Using smoketest of $(ppconfig.smoketest)"
    k = k[1:ppconfig.smoketest]
  end

  for idx in ProgressBar(1:stepsize:length(k))
    idxmax = min(idx + stepsize, length(k))
    z = k[idx:idxmax]
    out = combine(gdf[z], x -> preprocess(x, ppconfig), ungroup = false)

    cols = find_float64(out)
    @info "Converting Float64 to Float32 for $(length(cols)) columns."
    transform!(out, cols .=> convertFloat32, renamecols=false)

    for key in keys(vocabs)
      transform!(out, key => ByRow(x -> vocabs[key][x]) => key)
    end

    tag = Dates.format(now(), "yyyymmdd-HHMMSS")
    filename = tag * ".jld2"
    outfile = config.processeddir / filename
    @info "Saving $(outfile)"
    save(outfile.path, Dict(tag => out), compress = true)
  end
end


end
