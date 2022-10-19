
using Flux
using StockOptimizer: TFT

model = Chain(
    Parallel(
        .+,
        Dense(16 => 16),
        Chain(Dense(16 => 16), TFT.expand2),
    ),
    elu,
    TFT.GLU(16, 16)
)

x = rand(16, 90, 32);
c = rand(16, 32);



function GRN(in::Int, hidden::Int, out::Int ;context::Bool) 
    context ? model = TFT._grnContext(in, hidden, out) : model = TFT._grnNoContext(in, hidden, out)
    context ? skip = (mx, (x,_)) -> x .+ mx : skip = (mx, x) -> x .+ mx
    Chain(
        SkipConnection(model, skip),
        Flux.normalise
    )
end


grn = GRN(16, 16, 16, context=true)
z = grn((x, c));
size(z)

grn = GRN(16, 16, 16, context=false)
z = grn(x);
size(z)
z = grn((x, c));


