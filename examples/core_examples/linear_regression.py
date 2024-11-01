#!/usr/bin/env python
"""
An example of how to use bilby to perform parameter estimation for
non-gravitational wave data. In this case, fitting a linear function to
data with background Gaussian noise

"""
import bilby
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from bilby.core.utils.random import rng, seed
from dynesty.utils import get_random_generator

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

# A few simple setup steps
label = "linear_regression"
outdir = "outdir"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# First, we define our "signal model", in this case a simple linear function
def model(time, m, c):
    return time * m + c


# Now we define the injection parameters which we make simulated data with
injection_parameters = dict(m=0.5, c=0.2)

# For this example, we'll use standard Gaussian noise

# These lines of code generate the fake data. Note the ** just unpacks the
# contents of the injection_parameters when calling the model function.
sampling_frequency = 10
time_duration = 10
time = jnp.arange(0, time_duration, 1 / sampling_frequency)
N = len(time)
sigma = rng.normal(1, 0.01, N)
data = model(time, **injection_parameters) + jnp.array(rng.normal(0, sigma, N))

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
priors = dict()
priors["m"] = bilby.core.prior.Uniform(0, 5, "m")
priors["c"] = bilby.core.prior.Uniform(-2, 2, "c")

# And run sampler
if True:
    # with jax.log_compiles():
    # with jax.explain_cache_misses():
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=500,
        injection_parameters=injection_parameters,
        outdir=outdir,
        sample="acceptance-walk-jax",
        label=label,
        queue_size=500,
        rstate=get_random_generator(jax.random.PRNGKey(10)),
    )

# Finally plot a corner plot: all outputs are stored in outdir
result.plot_corner()
