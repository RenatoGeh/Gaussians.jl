using Plots
using Statistics

@inline Plots.plot(ϕ::Gaussian; fillrange = 0.0, fillalpha = 0.35, kwargs...) = plot(x -> pdf(ϕ, x); fillrange, fillalpha, kwargs...)
@inline function Plots.plot(ψ::TruncGaussian; fillrange = 0.0, fillalpha = 0.35, seriescolor = 1,
    thresholdcolor = nothing, kwargs...)
  P = plot(x -> pdf(ψ, x); fillrange, fillalpha, kwargs...)
  tcolor = isnothing(thresholdcolor) ? :red : seriescolor + 1
  vline!([ψ.θ]; seriescolor = tcolor)
  t = ψ.σ/2
  annotate!(ψ.θ + (ψ.lower ? t : -t), mean(pdf.(Ref(ψ), -ψ.σ:0.05:ψ.σ)), text("θ=$(ψ.θ)", tcolor, ψ.lower ? :left : :right, 10);
            seriescolor = tcolor)
  return P
end
@inline function Plots.plot!(ψ::TruncGaussian; fillrange = 0.0, fillalpha = 0.35, seriescolor = 1,
    thresholdcolor = nothing, kwargs...)
  P = plot!(x -> pdf(ψ, x); fillrange, fillalpha, kwargs...)
  tcolor = isnothing(thresholdcolor) ? :red : seriescolor + 1
  vline!([ψ.θ]; seriescolor = tcolor)
  t = ψ.σ/2
  annotate!(ψ.θ + (ψ.lower ? t : -t), mean(pdf.(Ref(ψ), -ψ.σ:0.05:ψ.σ)), text("θ=$(ψ.θ)", tcolor, ψ.lower ? :left : :right, 10);
            seriescolor = tcolor)
  return P
end
