module Layers

using Flux: @functor, glorot_uniform
function Hadamard(x1, x2)
  x1 .* x2
end

function splitmatrix(m)
  return [m[i, :] for i = 1:size(m, 1)]
end


struct Split{T}
  paths::T
end

Split(paths...) = Split(paths)

@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

# type for 3D data
const Abstract3DArray{T} = AbstractArray{T,3}

# fast fill of a bias, with size of the first dimension of the weights
create_bias(weights::AbstractArray) = fill!(similar(weights, size(weights, 1)), 0)

end
