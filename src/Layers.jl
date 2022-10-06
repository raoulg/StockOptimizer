module Layers

function Hadamard(x1, x2)
    x1.*x2
end

function splitmatrix(m)
    return [m[i, :] for i in 1:size(m, 1)]
end


struct Split{T}
    paths::T
end
  
Split(paths...) = Split(paths)
  
using Flux: @functor
@functor Split
  
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)


end