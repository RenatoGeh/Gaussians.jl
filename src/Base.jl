using StatsFuns

mutable struct Gaussian
  μ::Float64
  σ::Float64
end
export Gaussian

mutable struct TruncGaussian
  μ::Float64
  σ::Float64
  θ::Float64
  lower::Bool
end
export TruncGaussian

@inline lowertruncgauss(μ::Real, σ::Real, θ::Real, x::Real)::Real = x < θ ? normpdf(μ, σ, x)/normcdf(μ, σ, θ) : 0.0
@inline uppertruncgauss(μ::Real, σ::Real, θ::Real, x::Real)::Real = x ≥ θ ? normpdf(μ, σ, x)/normccdf(μ, σ, θ) : 0.0
@inline loglowertruncgauss(μ::Real, σ::Real, θ::Real, x::Real)::Real = x < θ ? normlogpdf(μ, σ, x) - normlogcdf(μ, σ, θ) : -Inf
@inline loguppertruncgauss(μ::Real, σ::Real, θ::Real, x::Real)::Real = x ≥ θ ? normlogpdf(μ, σ, x) - lognormccdf(μ, σ, θ) : -Inf

(ϕ::Gaussian)(x::Real)::Real = ismissing(x) ? 1.0 : normpdf(ϕ.μ, ϕ.σ, x)
(ψ::TruncGaussian)(x::Real)::Real = ismissing(x) ? 1.0 : (ψ.lower ? lowertruncgauss(ψ.μ, ψ.σ, ψ.θ, x) :
                                                                    uppertruncgauss(ψ.μ, ψ.σ, ψ.θ, x))

@inline pdf(ϕ::Gaussian, x::Real)::Real = ϕ(x)
@inline pdf(ψ::TruncGaussian, x::Real)::Real = ψ(x)
@inline logpdf(ϕ::Gaussian, x::Real)::Real = ismissing(x) ? 0.0 : normlogpdf(ϕ.μ, ϕ.σ, x)
@inline logpdf(ψ::TruncGaussian, x::Real)::Real = ismissing(x) ? 0.0 : (ψ.lower ? loglowertruncgauss(ψ.μ, ψ.σ, ψ.θ, x) :
                                                                                  loguppertruncgauss(ψ.μ, ψ.σ, ψ.θ, x))
export pdf, logpdf

@inline Base.rand(ϕ::Gaussian, n::Integer = 1)::Union{Float64, Vector{Float64}} = n == 1 ? randn() * ϕ.σ + ϕ.μ : randn(n) .* ϕ.σ .+ ϕ.μ
function Base.rand(ψ::TruncGaussian, n::Integer = 1)::Union{Float64, Vector{Float64}}
  # Inneficient rejection method, since sampling will only be used for debugging.
  if ψ.lower l, u = -Inf, ψ.θ
  else l, u = ψ.θ, Inf end
  if n == 1
    s = randn() * ψ.σ + ψ.μ
    while (s < l) || (s > u) s = randn() * ψ.σ + ψ.μ end
    return s
  end
  S = Vector{Float64}(undef, n)
  for i ∈ 1:n
    s = randn() * ψ.σ + ψ.μ
    while (s < l) || (s > u) s = randn() * ψ.σ + ψ.μ end
    S[i] = s
  end
  return S
end
