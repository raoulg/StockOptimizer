using Distributions
using Plots


ŷ = rand(2, 32)
y = rand(32)


softplus(x) = log(exp(x) + 1)

μ = ŷ[1, :]
σ = ŷ[2, :]


normalloss(y::Vector{T}; μ::Vector{T}, σ::Vector{T}) where {T<:AbstractFloat} =
  -sum(map((d, x) -> logpdf(d, x), Normal.(μ, softplus.(σ)), y))

normalloss(y::Vector{T}, ŷ::Matrix{T}) where {T<:AbstractFloat} =
  normalloss(y, μ = ŷ[1, :], σ = ŷ[2, :])

loss = normalloss(y, μ = μ, σ = σ)
loss = normalloss(y, ŷ)

d = SkewNormal(0, 1, 4)
x = LinRange(-2, 2, 100)
y = pdf.(d, x)
plot(x, y)

logpdf.(d, -2:2)

skewnormalloss(
  y::A;
  ξ::A,
  ω::A,
  α::A,
) where {A<:AbstractArray{T}} where {T<:AbstractFloat} =
  -sum(map((d, x) -> logpdf(d, x), SkewNormal.(ξ, ω, α), y))
  
skewnormalloss(y::Vector{T}, ŷ::Matrix{T}) where {T<:AbstractFloat} =
  skewnormalloss(y, ξ = ŷ[1, :], ω = ŷ[2, :], α = ŷ[3, :])

y = rand(32)
ŷ = rand(3, 32)
ξ = ŷ[1, :]
ω = ŷ[2, :]
α = ŷ[3, :]


skewnormalloss(y, ξ = ξ, ω = ω, α = α)
skewnormalloss(y, ŷ)
