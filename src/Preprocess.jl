module Preprocess


using Dates, DataFrames
using FileIO
using Impute: Impute
using Pipe: @pipe
using ProgressBars
using Random

using ..Load
using ..Vocabulary

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


function cast_date(x::Int64)::DateTime
    unix2datetime(x / 10^6)
end


function preprocess(df, config)
    ts = transform(df, config.datecol => ByRow(cast_date) => config.datecol)
    allDates = first(ts[!, config.datecol]):Dates.Day(1):last(ts[!, config.datecol])
    @pipe DataFrame(date=allDates) |>
        leftjoin(_, ts, on=:date) |>
        sort(_, :date) |>
        transform(_, config.locf_cols .=> Impute.locf .=> config.locf_cols) |>
        fillnawithzero(_, config.zero_cols) |>
        dropmissing(_) |>
        _[!, Not(config.drop_cols)]
end


function batch_and_save(config, ppconfig) 
    @info "Loading data"
    df = Load.join_static(config)
    @info "Grouping data"
    df = impute_none(df, ppconfig.none_cols)
    dropmissing!(df)
    vocabs = Vocabulary.extract_vocabs(df, config)

    gdf = groupby(df, ppconfig.group_cols)
    gdf = filter(x -> nrow(x) > config.minlen, gdf)
    k = keys(gdf)
    k = shuffle(k)
    stepsize=ppconfig.stepsize
    if ppconfig.smoketest > 1
        @info "Using smoketest of $(ppconfig.smoketest)"
        k = k[1:ppconfig.smoketest]
    end

    for idx in ProgressBar(1:stepsize:length(k))
        idxmax = min(idx+stepsize, length(k))
        z = k[idx:idxmax]
        out = combine(gdf[z], x -> preprocess(x, ppconfig))

        for key in keys(vocabs)
            transform!(out, key => ByRow(x -> vocabs[key][x]) => key)
        end

        tag = Dates.format(now(), "yyyymmdd-HHMMSS") 
        filename = tag * ".jld2"
        outfile = config.processeddir / filename 
        @info "Saving $(outfile)"
        save(outfile.path, Dict(tag => out), compress=true)
    end
end


end