```@meta
CurrentModule = UncertaintyOptimization
```

# Sampling

UncertaintyOptimization uses Markov chain Monte Carlo (MCMC) to infer a
posterior distribution for the parameters marked `uncertain` in the YAML
model. The default sampler is Turing's No-U-Turn Sampler (NUTS), an adaptive
form of Hamiltonian Monte Carlo (HMC). NUTS is particularly efficient for
continuous, differentiable models because it uses posterior gradients to move
through many correlated parameters without the slow random walk produced by
simple proposal-based samplers.

Efficient sampling is not controlled by one setting. It depends on the
posterior geometry, the NUTS trajectory settings, initial parameter values,
the numerical ODE solve, and the smoothness of the model equations. This page
explains those parts and gives a practical order in which to improve them.

## How NUTS Explores A Posterior

Let ``q`` contain the uncertain parameters and let ``p`` be an auxiliary
momentum. HMC defines a Hamiltonian

```math
H(q,p) = U(q) + K(p)
       = -\log \pi(q) + \frac{1}{2}p^\mathsf{T}M^{-1}p,
```

where ``\pi(q)`` is the posterior density and ``M`` is the mass matrix, also
called the metric. A leapfrog integrator follows the Hamiltonian dynamics using
the gradient ``\nabla_q\log\pi(q)``. A final Metropolis correction removes the
small bias introduced by numerical integration.

NUTS makes HMC easier to use by constructing a binary tree of forward and
backward leapfrog steps. It stops extending the trajectory when it detects a
U-turn, rather than requiring the user to choose a fixed trajectory length.
This lets NUTS take long moves through broad posterior regions and short moves
where the trajectory turns quickly.

!!! tip "Intuition: a ball on the posterior landscape"
    Imagine a ball whose position is the vector of uncertain parameters. The
    posterior forms a landscape: high-probability parameter combinations are
    valleys in the corresponding negative log-posterior. The ball needs the
    partial derivative with respect to every parameter to know the local slope,
    but it also has its own velocity, represented by the momentum ``p``. The
    slope bends the trajectory while the momentum carries the ball across the
    valley. This combination allows HMC to travel much farther than a sampler
    that repeatedly takes small, directionless steps.

## The Main NUTS Controls

`TuringSpec` accepts a configured Turing sampler through its `sampler` field.
For example:

```julia
using AdvancedHMC: DenseEuclideanMetric
using Turing: NUTS

sampler = NUTS(
    1000,
    0.8;
    max_depth = 9,
    init_ϵ = 0.0,
    metricT = DenseEuclideanMetric,
)

turing_spec = TuringSpec(
    simulation = simulation,
    data = observed_data,
    noise_initial = 3.0,
    sampler = sampler,
    n_samples = 3000,
    n_chains = 4,
)
```

The first positional value is the number of adaptation, or warmup, iterations.
The second is the target acceptance probability ``\delta``. `init_ϵ = 0.0`
asks Turing to find an initial step size automatically.

### Target Acceptance And Step Size

At the end of an HMC trajectory, the proposal is accepted with probability

```math
\alpha = \min\left(1, \exp(-\Delta H)\right).
```

An exact trajectory would conserve ``H`` and have ``\alpha=1``. Leapfrog
integration introduces an energy error ``\Delta H``, so the acceptance rate is
a direct signal of trajectory accuracy. During adaptation, NUTS changes its
step size to approach the target `δ`.

- A lower target, such as `0.65`, normally permits a larger step size and fewer
  gradient evaluations, but it can produce more rejected or divergent moves.
- A higher target, such as `0.8` or `0.9`, normally produces a smaller, more
  accurate step size. It is often safer for difficult ODE posteriors but costs
  more per effective sample.
- An extremely high target is not automatically better. Tiny steps can make
  every trajectory expensive while hiding a poor parameterization.

The mean `acceptance_rate` after warmup should be reasonably close to the
target. Persistently low acceptance, `numerical_error` transitions, or large
energy errors suggest that the step size is too large, the equations are not
smooth, or some proposed ODE solves are unstable. Raise `δ` before manually
forcing a small `init_ϵ`; manual `init_ϵ` is mainly useful when automatic
initialization repeatedly starts at an unsafe scale.

### Adaptation Length

Warmup learns the step size and metric. More adaptation is useful when there
are many uncertain parameters, their scales differ greatly, a dense metric is
used, or chains start far from typical posterior values. Too little adaptation
leaves production sampling with poor geometry. More adaptation does not repair
an invalid model and does not add posterior draws: it is an up-front cost.

Compare adapted step sizes and metrics across chains. If they vary greatly,
increase warmup and check whether the chains have reached the same posterior
region. Reducing adaptation is sensible only after repeated runs show stable
adapted settings from good initial points.

### Tree Depth

`max_depth` limits how many times NUTS may double its trajectory tree. A depth
of ``d`` permits up to roughly ``2^d`` leapfrog steps, so increasing depth by
one can approximately double the worst-case work for a draw. A draw that often
reaches the configured limit has been stopped by the limit rather than by the
No-U-Turn criterion.

!!! tip "Intuition: a ball travelling through valleys"
    Tree depth limits how long NUTS lets the ball travel while it searches both
    directions along a valley. A shallow tree stops the ball before it has
    crossed a long valley, producing nearby, correlated samples. A deeper tree
    gives it time to travel around a curved or elongated valley until its path
    naturally starts to turn back. If the valley is extremely narrow or badly
    scaled, however, repeatedly extending the journey is wasteful: reshape the
    coordinates or metric instead of only allowing a longer trip.

Inspect both `tree_depth` and `n_steps`. Increase `max_depth` when many valid
transitions hit the cap and effective sample size improves with longer
trajectories. First improve the metric or parameterization when depth
saturation accompanies tiny step sizes, numerical errors, or strong posterior
correlations. Tree depth is a safety and cost limit, not a general cure.

### Other Controls

`Δ_max` is the maximum allowed energy error before Turing treats a trajectory
as divergent. The default is normally preferable: increasing it can hide bad
trajectories instead of improving them. `n_samples` controls retained posterior
draws, whereas the adaptation count controls warmup. More retained draws reduce
Monte Carlo error only after the chain mixes; they do not rescue a stuck chain.

Use multiple chains with dispersed, valid starting points. Agreement between
chains is stronger evidence of convergence than one long chain. Random seeds
are important for reproducibility but should not be tuned to obtain a preferred
posterior.

## Metric And Parameter Scale

The metric determines how parameter displacement is measured and how momentum
is assigned. If one uncertain parameter ranges around ``10^{-3}`` and another
around ``10^3``, the same Euclidean step cannot suit both. The posterior then
looks like a long, narrow valley: a step small enough not to cross its narrow
side advances very slowly along its length.

Ideally the sampling coordinates make plausible changes in every parameter
roughly comparable. For example,

```math
z_i = \frac{q_i - \mu_i}{s_i}
```

uses a meaningful location ``\mu_i`` and scale ``s_i``. Positive parameters
that span orders of magnitude are often better represented by
``z_i=\log q_i``; bounded parameters can use a logit transformation. Priors
must be transformed consistently. Turing also transforms constrained
distributions internally, but a scientifically appropriate model
parameterization can still make the posterior much easier to sample.

AdvancedHMC provides three useful Euclidean metrics:

- `UnitEuclideanMetric` uses an identity metric. It is cheap but assumes that
  every transformed parameter already has a similar scale and is nearly
  uncorrelated.
- `DiagEuclideanMetric` adapts one scale per parameter. It is the robust default
  and handles different parameter ranges, but not rotated or correlated
  valleys.
- `DenseEuclideanMetric` adapts a full matrix. It can rescale parameters and
  represent pairwise correlations, allowing much more direct movement through
  an angled valley. It needs more warmup and its estimation becomes expensive
  and noisy as dimension grows.

Start with the diagonal metric. Try the dense metric when posterior plots show
strong correlations, tree depth is high despite sensible marginal scales, and
there are enough warmup samples to estimate a covariance matrix reliably. Use
the unit metric only when parameters have already been standardized. Compare
metrics by effective samples per second, numerical errors, depth saturation,
and agreement of posterior summaries, not by runtime alone.

## Initial Parameter Values

Initial values matter because every first trajectory needs a finite prior,
likelihood, gradient, and successful ODE solution. A point near a constrained
boundary or in an unstable simulation region can force a tiny step size or make
warmup fail before NUTS reaches the typical set.

For `run_inference`, uncertain parameters start from
`SimulationSpec.uncertain_param_values` when provided, otherwise from their
scalar YAML `value`. Observation noise starts from `TuringSpec.noise_initial`.
An initial value for a uniform prior must be strictly inside its bounds; a YAML
value on the boundary falls back to the prior mean, while an invalid explicit
override raises an error.

Choose finite, scientifically plausible values inside the prior support and
test one simulation at those values before sampling. The initial point need not
be the posterior mode, but it should avoid pathological ODE regions. Use
several dispersed starts to detect separate posterior modes rather than placing
every chain at exactly the same point.

After a pilot posterior is available, its draws can provide better starts for
later runs. Transform and standardize the posterior draws, cluster them, and
select a high-density representative or medoid from each cluster. These points
can recover stable starts from distinct posterior regions without choosing an
arbitrary draw. Preserve multiple clusters when they represent genuine modes;
collapsing them to one start can conceal multimodality. Representatives can be
placed in YAML or `uncertain_param_values` for later runs. Supplying a different
representative to every chain currently requires separate `run_inference`
calls or direct Turing initialization because the package-generated start is
shared across chains.

## Equations Must Have A Smooth Landscape

NUTS can only use gradients that the model makes available. The common
positivity guard

```math
x_+ = \max(x,0)
```

is continuous but not differentiable at zero. Its derivative changes abruptly
from zero to one. When an ODE trajectory crosses zero, this corner can make
sensitivities discontinuous, damage leapfrog energy conservation, and lower
acceptance. It can also create a flat negative region that gives NUTS no useful
gradient back toward physically meaningful values.

Prefer a model that enforces positivity by construction when scientifically
appropriate. A smooth approximation to the positive part is

```math
\operatorname{smoothpos}_\varepsilon(x)
    = \frac{x + \sqrt{x^2 + \varepsilon}}{2},
```

where ``\varepsilon`` controls the width of the smooth transition. A softplus
or a positive-state parameterization is another option. The smoothing scale
must be small relative to meaningful state values but not so small that it is
numerically indistinguishable from the original corner. Any replacement changes
the model near zero and must therefore be checked against simulations and
posterior predictive behavior.

Improving the posterior geometry in this way can make NUTS more efficient, not
just more stable. Removing a nonsmooth construct such as `max(0, x)` gives the
gradient a more continuous landscape, which can allow a larger step size, fewer
numerical errors and divergences, and more effective samples per unit of
computation. The replacement must still represent the intended scientific model.

### Hill Equations In The Log Domain

For positive ``x``, a Hill activation can be written as

```math
h(x;K,n) = \frac{x^n}{K^n+x^n}
         = \frac{1}{1+\exp\{n(\log K-\log x)\}}.
```

The second expression is the same Hill model evaluated through a logistic
function of log concentrations. A Hill repression is

```math
r(x;K,n) = \frac{1}{1+(x/K)^n}
         = \frac{1}{1+\exp\{n(\log x-\log K)\}}.
```

The log-Hill form avoids directly raising very small or very large values to a
power, reducing overflow, underflow, and unstable ratios. In code, use a
carefully chosen positive floor or smooth-positive input before taking `log`.
Compared with wrapping powers in `max(x, 0)`, a smooth positive input followed
by a log-Hill evaluation gives NUTS a smoother and usually more numerically
stable landscape. It also makes the ratio ``x/K`` explicit, which is useful
when parameters span several orders of magnitude.

The log form improves numerical evaluation; it does not justify a different
biological equation. Confirm that activation, repression, zero behavior, and
units still match the intended mechanism.

## Diagnose Before Tuning

The returned chain can contain NUTS diagnostics including `acceptance_rate`,
`tree_depth`, `n_steps`, `numerical_error`, `hamiltonian_energy`, and
`step_size`. Also inspect trace plots, rank or density plots, effective sample
size (ESS), and ``\widehat R`` across chains.

| Symptom | Likely response |
|:---|:---|
| Low acceptance or numerical errors | Increase target `δ`; then inspect equation smoothness and ODE failures. |
| Frequent maximum tree depth | Improve posterior geometry, especially parameter scaling or the metric, first; increase `max_depth` if trajectories remain valid. |
| High autocorrelation and low ESS | Improve parameterization or metric; then consider longer trajectories or more draws. |
| Chains disagree or ``\widehat R`` remains high | Run longer warmup from dispersed starts and investigate multimodality. |
| Dense metric is unstable | Add warmup, reduce dimension, regularize the model, or return to a diagonal metric. |
| Clean diagnostics but high Monte Carlo error | Increase `n_samples` or the number of chains. |

A useful tuning order is:

1. Make every initial simulation finite and place starts inside prior support.
2. Remove discontinuities and numerically unstable equation forms.
3. Transform badly ranged positive or bounded parameters.
4. Run adequate warmup with a diagonal metric and several chains.
5. Tune target acceptance in the approximate range `0.65` to `0.9` using
   acceptance and numerical-error diagnostics.
6. Address repeated depth saturation by improving posterior geometry through
   better scaling, metric choice, or parameterization before raising `max_depth`.
7. Compare diagonal and dense metrics by ESS per second and posterior agreement.
8. Increase retained samples only after the chains mix reliably.

Change one group of settings at a time and compare posterior summaries, not
only speed. The fastest chain is not useful if it explores the wrong region,
and a high acceptance rate is not useful if each effective sample requires an
excessively long trajectory.
