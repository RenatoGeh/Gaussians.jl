@inline function ∂μ_lower(μ::Real, σ::Real, θ::Real, x::Real)::Real
  ϕ = normpdf(μ, σ, x)
  ϕ_θ = normpdf(μ, σ, θ)
  Φ = normcdf(μ, σ, θ)
  return ϕ*((x-μ)*Φ/(σ^2)+ϕ_θ)/(Φ^2)
end

@inline function ∂σ_lower(μ::Real, σ::Real, θ::Real, x::Real)::Real
  ϕ = normpdf(μ, σ, x)
  ϕ_θ = normpdf(μ, σ, θ)
  Φ = normcdf(μ, σ, θ)
  return ϕ*(((((x-μ)/σ)^2)-1)*Φ+(θ-μ)*ϕ_θ)/(σ*(Φ^2))
end

@inline function ∂μ_upper(μ::Real, σ::Real, θ::Real, x::Real)::Real
  ϕ = normpdf(μ, σ, x)
  ϕ_θ = normpdf(μ, σ, θ)
  Φ_i = normccdf(μ, σ, θ)
  return ϕ*(((x-μ)/(σ^2))*Φ_i-ϕ_θ)/(Φ_i^2)
end

@inline function ∂σ_upper(μ::Real, σ::Real, θ::Real, x::Real)::Real
  ϕ = normpdf(μ, σ, x)
  ϕ_θ = normpdf(μ, σ, θ)
  Φ_i = normccdf(μ, σ, θ)
  return ϕ*(((((x-μ)/σ)^2)-1)*Φ_i+((μ-θ)*ϕ_θ))/(σ*(Φ_i^2))
end

@inline function ∂μ(ψ::TruncGaussian, x::Real)::Real
  μ, σ, θ = ψ.μ, ψ.σ, ψ.θ
  return ψ.lower ? ∂μ_lower(μ, σ, θ, x) : ∂μ_upper(μ, σ, θ, x)
end

@inline function ∂σ(ψ::TruncGaussian, x::Real)::Real
  μ, σ, θ = ψ.μ, ψ.σ, ψ.θ
  return ψ.lower ? ∂σ_lower(μ, σ, θ, x) : ∂σ_upper(μ, σ, θ, x)
end

@inline ∇(ψ::TruncGaussian, x::Real)::Tuple{Real, Real} = ∂μ(ψ, x), ∂σ(ψ, x)

function ℒ(ψ::TruncGaussian, X::Vector{<:Real})::Real
  n = length(X)
  if n > 100
    L = Vector{Float64}(undef, n)
    Threads.@threads for i ∈ 1:n L[i] = logpdf(ψ, X[i]) end
  else L = logpdf.(Ref(ψ), X) end
  return sum(L)/n
end
export ℒ

@inline ∂μ_logcdf(ψ::TruncGaussian, x::Real)::Real = -normpdf(ψ.μ, ψ.σ, x)/normcdf(ψ.μ, ψ.σ, x)
@inline ∂σ_logcdf(ψ::TruncGaussian, x::Real)::Real = ((ψ.μ-x)*normpdf(ψ.μ, ψ.σ, x))/(ψ.σ*normcdf(ψ.μ, ψ.σ, x))
@inline ∂μ_logpdf(ψ::TruncGaussian, x::Real)::Real = (x-ψ.μ)/(ψ.σ^2)
@inline ∂σ_logpdf(ψ::TruncGaussian, x::Real)::Real = ((((x-ψ.μ)/ψ.σ)^2)-1)/ψ.σ

function ∂μ_ℒ_lower(ψ::TruncGaussian, X::Vector{<:Real})::Real
  δ, n = 0.0, length(X)
  μ, σ, θ = ψ.μ, ψ.σ, ψ.θ
  v = σ^2
  for i ∈ 1:n δ += X[i]-μ end
  δ = δ/v + n*normpdf(μ, σ, θ)/normcdf(μ, σ, θ)
  return δ / n
end

function ∂σ_ℒ_lower(ψ::TruncGaussian, X::Vector{<:Real})::Real
  δ, n = 0.0, length(X)
  μ, σ, θ = ψ.μ, ψ.σ, ψ.θ
  v = σ^2
  for i ∈ 1:n δ += (X[i]-μ)^2 end
  δ = δ/v - n + n*((θ-μ)*normpdf(μ, σ, θ))/normcdf(μ, σ, θ)
  return δ / (σ*n)
end

function ∂μ_ℒ_upper(ψ::TruncGaussian, X::Vector{<:Real})::Real
  δ, n = 0.0, length(X)
  μ, σ, θ = ψ.μ, ψ.σ, ψ.θ
  v = σ^2
  for i ∈ 1:n δ += X[i]-μ end
  δ = δ/v - n*normpdf(μ, σ, θ)/normccdf(μ, σ, θ)
  return δ / n
end

function ∂σ_ℒ_upper(ψ::TruncGaussian, X::Vector{<:Real})::Real
  δ, n = 0.0, length(X)
  μ, σ, θ = ψ.μ, ψ.σ, ψ.θ
  v = σ^2
  for i ∈ 1:n δ += (X[i]-μ)^2 end
  δ = δ/v - n + n*((μ-θ)*normpdf(μ, σ, θ))/normccdf(μ, σ, θ)
  return δ / (σ*n)
end

@inline ∇ℒ_lower(ψ::TruncGaussian, X::Vector{<:Real})::Tuple{Real, Real} = ∂μ_ℒ_lower(ψ, X), ∂σ_ℒ_lower(ψ, X)
@inline ∇ℒ_upper(ψ::TruncGaussian, X::Vector{<:Real})::Tuple{Real, Real} = ∂μ_ℒ_upper(ψ, X), ∂σ_ℒ_upper(ψ, X)
@inline ∇ℒ(ψ::TruncGaussian, X::Vector{<:Real})::Tuple{Real, Real} = ψ.lower ? ∇ℒ_lower(ψ, X) : ∇ℒ_upper(ψ, X)
export ∇ℒ
