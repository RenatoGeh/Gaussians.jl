@inline function gradient_lower!(ψ::TruncGaussian, X::Vector{<:Real}, k::Integer; η::Real = 1.0)
  for i ∈ 1:k
    δ_μ, δ_σ = ∇ℒ_lower(ψ, X)
    ψ.μ, ψ.σ = min(ψ.μ + η*δ_μ, ψ.θ), ψ.σ + η*δ_σ
  end
  return nothing
end

@inline function gradient_upper!(ψ::TruncGaussian, X::Vector{<:Real}, k::Integer; η::Real = 1.0)
  for i ∈ 1:k
    δ_μ, δ_σ = ∇ℒ_upper(ψ, X)
    ψ.μ, ψ.σ = max(ψ.θ, ψ.μ + η*δ_μ), ψ.σ + η*δ_σ
  end
  return nothing
end

@inline gradient!(ψ::TruncGaussian, X::Vector{<:Real}, k::Integer; η::Real = 1.0) = ψ.lower ? gradient_lower!(ψ, X, k; η) : gradient_upper!(ψ, X, k; η)
export gradient!

@inline initialize(X::Vector{<:Real}, θ::Real, lower::Bool)::TruncGaussian = (σ = std(X); TruncGaussian(lower ? mean(X)-σ : mean(X)+σ, σ, θ, lower))
export initialize
