module Load

export filterdir

using DataFrames, Parquet
using Dates

using ..Pathlib: hassuffix, Path


function filterdir(dir::Path, suffix::String)
  [dir / f for f in readdir(dir.path) if hassuffix(f, suffix)]
end

function cast_date(x::Int64)::DateTime
  unix2datetime(x / 10^6)
end

function load_train(config)
  trainfile = config.datadir / config.trainfile
  df = DataFrame(read_parquet(trainfile.path))
  transform(df, config.datecol => ByRow(cast_date) => config.datecol)
end

function join_static(df, config)
  files = filterdir(config.datadir, config.suffix)
  static = [f for f in files if f.name in config.staticfiles]

  for static_file in static
    stat_df = DataFrame(read_parquet(static_file.path))
    idx = config.col_map[static_file.name]
    leftjoin!(df, stat_df, on = idx)
  end
  df
end

function join_dynamic(df, config)
  files = filterdir(config.datadir, config.suffix)
  dynamic = [f for f in files if f.name in config.dynamicfiles]

  for dyn_file in dynamic
    dyn_df = DataFrame(read_parquet(dyn_file.path))
    transform!(dyn_df, config.datecol => ByRow(cast_date) => config.datecol)
    idx = config.col_map[dyn_file.name]
    leftjoin!(df, dyn_df, on = idx)
  end
  df
end

end
