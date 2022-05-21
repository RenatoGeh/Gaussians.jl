# Gaussians.jl

A simple package for learning (semi-finite) truncated Gaussian distributions from data by gradient
ascent. Mainly for personal use as playground or testing grounds. Use at own risk.

## Example

Say we know some data `S` to come from a ground-truth truncated Gaussian distribution
`T(μ = 3.0, σ = 2.5)` whose support is defined over the `(-∞, θ = 4.75]` interval. Let's simulate
this by constructing `T` and sampling `S` from `T`.

```julia
using Gaussians

# Define our support (-∞, θ].
θ, lower = 4.75, true

# The fourth argument here specifies whether this is a lower (set to true), i.e. (-∞, θ]; or upper
# (set to false), i.e. [θ, ∞), interval. In this case, we want to set it to true.
T = TruncGaussian(3.0, 2.5, θ, lower)
# Sample 1000 values from T.
S = rand(T, 1000)
```

Now, assuming that we only know of the support given by `θ` and of the data `S`, let us learn a new
truncated Gaussian from our knowledge. We shall first create a new truncated Gaussian by
initializing its mean to more or less the middle of the center of mass of `S` and set its standard
deviation to that of `S`.

```julia
# Initialize mean such that it is within the support and set standard deviation to something more
# or less reasonable.
H = initialize(S, θ, lower)
```

We may now start learning by gradient ascent, which in our case maximizes the log-likelihood.

```julia
# Run 100 iterations of gradient ascent with learning rate η = 1.
gradient!(H, S, 100; η = 1)

# Compare log-likelihoods (write ℒ in Julia with \scrL).
println("ℒ(T, S) = ", ℒ(T, S), "\nℒ(H, S) = ", ℒ(H, S))
```

We could do the same with `[θ, ∞)` as the support. We only have to set it accordingly when
constructing/initializing the Gaussian; `gradient!` then takes care of the rest.
