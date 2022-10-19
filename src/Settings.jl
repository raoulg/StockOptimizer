module Settings

export Config
import Base.@kwdef
using ..Pathlib

@kwdef struct Config
  datadir::Path
  processeddir::Path
  suffix::String
  staticfiles::Vector{Pathlib.Path}
  dynamicfiles::Vector{Pathlib.Path}
  trainfile::Pathlib.Path
  datecol::Symbol
  dynamic_map::Dict{String,NamedTuple{(:datecol, :valcol),Tuple{String,Vector{String}}}}
  col_map::Dict{String,Symbol}
  minlen::Int
end

@kwdef struct PreprocessConfig
  datecol::Symbol
  none_cols::Vector{Symbol}
  locf_cols::Vector{Symbol}
  zero_cols::Vector{Symbol}
  drop_cols::Vector{Symbol}
  group_cols::Vector{Symbol}
  stepsize::Int
  smoketest::Int
end

@kwdef struct DataConfig
  dyn_cat::Vector{Symbol}
  dyn_real::Vector{Symbol}
  stat_cat::Vector{Symbol}
  stat_real::Vector{Symbol}
end

end
