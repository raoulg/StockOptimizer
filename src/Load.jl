module Load

export filterdir

using DataFrames, Parquet

using ..Pathlib: hassuffix, Path


function filterdir(dir::Path, suffix::String)
    [dir / f for f in readdir(dir.path) if hassuffix(f, suffix)]
end 

function join_static(config)
    trainfile = config.datadir / config.trainfile
    files = filterdir(config.datadir, config.suffix)
    static = [f for f in files if f.name in config.staticfiles]
    df = DataFrame(read_parquet(trainfile.path))

    for static_file in static
        stat_df = DataFrame(read_parquet(static_file.path))
        idx = config.col_map[static_file.name]
        leftjoin!(df, stat_df, on=idx)
    end
    df
end

end