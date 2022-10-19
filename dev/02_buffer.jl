using StockOptimizer: Preprocess, Path, filterdir
using FileIO
using Random: shuffle, rand
using StockOptimizer: Data, DataConfig
using BenchmarkTools
# using Base:@kwdef
using DataFrames

n = 2
files = filterdir(Path("data/processed"), "jld2")
tag = split(files[n].name, ".")[1]
gdf = load(files[n].path)[tag];

dataconfig = DataConfig(
    dyn_cat = [:onpromotion, :perishable],
    dyn_real = [:unit_sales, :dcoilwtico],
    stat_cat = [:family, :class, :city, :state, :type, :cluster],
    stat_real = []
)

# a timeseries has two grouped dataframes
# static, and dynamic
# the keys refer to the groups
items = Data.TimeSeriesItem(gdf, dataconfig);
fieldnames(Data.TimeSeriesItem)

# getobs returns the first (shuffeled) key
@benchmark ts = Data.getobs(items, 1)

# getobs returns an Observation type
obs = Data.getobs(items, 1)
typeof(obs)
# it has a dynamic and a static observation
fieldnames(typeof(obs))

# the bufferedloader takes in an item, and will fill a buffer with 
# 32 examples, and stepsize 120
buff = Data.BufferedLoader(items, 32, 120);
typeof(buff)
# we can empty the TimeSeriesItem in batches of 32 until there is no timeseries of
# stepsize left to fill a buffer
for batch in buff
    if !isnothing(batch)
        println(size(batch), buff.count)
    end
end

@benchmark batch = buff() setup=(buff = Data.BufferedLoader(items, 32, 120))
buff = Data.BufferedLoader(items, 32, 120);
batch = buff();
# a batch is a Vector of Observations
typeof(batch)


# Deprecated

function reshape_batch(batch, cols::Vector{Symbol}, shape::Tuple)::Array{Float64, 3}
    m = [transpose(Matrix(v.dynamic[:, cols])) for v in batch]
    reshape(reduce(hcat, m), shape)
end

cat_cols = [:onpromotion, :perishable]
length(batch)
reshape_batch(batch, cat_cols, (120, 2, 32))


function test_reshape(realcol::Vector{Symbol}, shape::Tuple)
    d, t, b = shape
    # dyn = [:unit_sales, :dcoilwtico]
    batch = [Data.Observation(DataFrame(rand(t, d), realcol), DataFrame()) for _ in 1:b]
    z = reshape_batch(batch, realcol, (d, t, b));

    for k in 1:b
        y = collect(transpose(Matrix(batch[k].dynamic[:, realcol])))
        @assert (y == z[:, :, k])
        @assert sum(y - z[:, :, k]) == 0 
    end
end

realcol = [:unit_sales, :dcoilwtico]
shape = (2, 120, 32)
test_reshape(realcol, shape)
buff = Data.BufferedLoader(items, 32);
batch = buff()
batch

using Flux: Dense

realcol = [:unit_sales, :dcoilwtico]
m = reshape_batch(batch, realcol, shape)
nn = Dense(2 => 16)

function bmm(nn, m)
    # m = (d, t, b)
    # reshape (d, b * t)
    m_ = reshape(m, size(m, 1), :)
    # m_ = reshape(permutedims(m, (2, 3, 1)), size(m, 2), :)
    x = nn(m_) # (d_new, b*t) 

    # reshape (d_new, b, t)
    reshape(x, :, size(m, 2), size(m, 3))
    # permutedims(reshape(x, :, size(m, 3), size(m, 1)), (3, 1, 2))
end



size(bmm(nn, m)) == (16, 120, 32)

using Random: rand
function test_bmm(d, t, b)
    x = rand(d, t, b)
    nn = Dense(d => 16)
    bx = bmm(nn, x)
    for i in 1:b
        @assert bx[:, :, i] == nn(x[:, :, i])
    end
end
test_bmm(120, 2, 32)

names(batch[1].dynamic)
length(batch)


getshape(batch, cols) = (length(cols), size(first(batch).dynamic, 1), length(batch))
reshape_batch(batch, cols) = reshape_batch(batch, cols, getshape(batch, cols))
realcols = [:unit_sales, :dcoilwtico]

dyn_real = reshape_batch(batch, realcols)
using StockOptimizer: TFT, Layers
using Flux: Dense, Parallel
using Flux:glorot_uniform
dims = 16

const Abstract3DArray{T} = AbstractArray{T, 3}

create_bias(weights::AbstractArray) = fill!(similar(weights, size(weights, 1)), 0)

struct TimeDense{M<:AbstractMatrix, B}
    W::M
    b::B
    function TimeDense(W::M) where {M<:AbstractMatrix}
        b = create_bias(W)
        new{M, typeof(b)}(W, b)
    end
end

TimeDense(in::Int, out::Int) = TimeDense(glorot_uniform(out, in))

function (nn::TimeDense)(x::A) where {T, A <: Abstract3DArray{T}}
    newsize = (size(nn.W, 1), size(x, 2), size(x, 3))
    reshape(nn.W * reshape(x, size(x, 1), :) .+ nn.b, newsize)
end

dyn_real[1:1, :, :]

split_td(x) = [x[i:i, :, :] for i in 1:size(x, 1)]
x = split_td(dyn_real)

paths(dyn_real, dims) = [TimeDense(1, dims) for _ in 1:dyn_real]
p = paths(2, 16)

using Flux
f(x...) = (x)
model = Parallel(vcat, TimeDense(2, 16), TimeDense(2, 16))

out = model(rand(2, 10, 10), rand(2, 10, 10))

size(out[1])
size(out[2])






