module Metrics

using Distributions: Normal, Sk

softplus(x) = log(exp(x) + 1)

normalloss(y::Vector{T}; μ::Vector{T}, σ::Vector{T}) where {T<:AbstractFloat} =
  -sum(map((d, x) -> logpdf(d, x), Normal.(μ, softplus.(σ)), y))

normalloss(y::Vector{T}, ŷ::Matrix{T}) where {T<:AbstractFloat} =
  normalloss(y, μ = ŷ[1, :], σ = ŷ[2, :])

skewnormalloss(
  y::A;
  ξ::A,
  ω::A,
  α::A,
) where {A<:AbstractArray{T}} where {T<:AbstractFloat} =
  -sum(map((d, x) -> logpdf(d, x), SkewNormal.(ξ, ω, α), y))

skewnormalloss(y::Vector{T}, ŷ::Matrix{T}) where {T<:AbstractFloat} =
  skewnormalloss(y, ξ = ŷ[1, :], ω = ŷ[2, :], α = ŷ[3, :])

end
