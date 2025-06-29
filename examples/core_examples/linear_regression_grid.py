#!/usr/bin/env python
"""
An example of how to use bilby to perform parameter estimation for
fitting a linear function to data with background Gaussian noise.
This will compare the output of using a stochastic sampling method
to evaluating the posterior on a grid.
"""
import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import random

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
random.seed(123)

# A few simple setup steps
label = "linear_regression_grid"
outdir = "outdir"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# Now we define the injection parameters which we make simulated data with
injection_parameters = dict()
injection_parameters["c"] = 0.2
injection_parameters["m"] = 0.5

# For this example, we'll use standard Gaussian noise

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_parameters when calling the model function.
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
N = len(time)
sigma = 3.0
data = model(time, **injection_parameters) + random.rng.normal(0, sigma, N)

# We quickly plot the data to check it looks sensible
fig, ax = plt.subplots()
ax.plot(time, data, "o", label="data")
ax.plot(time, model(time, **injection_parameters), "--r", label="signal")
ax.set_xlabel("time")
ax.set_ylabel("y")
ax.legend()
fig.savefig("{}/{}_data.png".format(outdir, label))

# Now lets instantiate a version of our GaussianLikelihood, giving it
# the time, data and signal model
likelihood = bilby.likelihood.GaussianLikelihood(time, data, model, sigma)

# From hereon, the syntax is exactly equivalent to other bilby examples
# We make a prior
priors = bilby.core.prior.PriorDict()
priors["m"] = bilby.core.prior.Uniform(0, 5, "m")
priors["c"] = bilby.core.prior.Uniform(-2, 2, "c")

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=500,
    sample="unif",
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)
fig = result.plot_corner(parameters=injection_parameters, save=False)

grid = bilby.core.grid.Grid(likelihood, priors, grid_size={"m": 200, "c": 100})

# overplot the grid estimates
grid_evidence = grid.log_evidence
axes = fig.get_axes()
axes[0].plot(
    grid.sample_points["c"],
    np.exp(grid.marginalize_ln_posterior(not_parameters="c") - grid_evidence),
    "k--",
)
axes[3].plot(
    grid.sample_points["m"],
    np.exp(grid.marginalize_ln_posterior(not_parameters="m") - grid_evidence),
    "k--",
)
axes[2].contour(
    grid.mesh_grid[1],
    grid.mesh_grid[0],
    np.exp(grid.ln_posterior - np.max(grid.ln_posterior)),
)

fig.savefig("{}/{}_corner.png".format(outdir, label), dpi=300)

# compare evidences
print("Dynesty log(evidence): {}".format(result.log_evidence))
print("Grid log(evidence): {}".format(grid_evidence))
