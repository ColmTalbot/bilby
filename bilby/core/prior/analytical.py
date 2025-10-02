import os

import numpy as np
os.environ["SCIPY_ARRAY_API"] = "1"  # noqa  # flag for scipy backend switching
from scipy.special import (
    betainc,
    betaincinv,
    betaln,
    erf,
    # erfinv,  # erfinv is not currently backend agnostic
    gammaincinv,
    gammainc,
    gammaln,
    stdtr,
    stdtrit,
    xlogy,
)

from .base import Prior
from ..utils import logger
from ...compat.utils import xp_wrap


class DeltaFunction(Prior):

    def __init__(self, peak, name=None, latex_label=None, unit=None):
        """Dirac delta function prior, this always returns peak.

        Parameters
        ==========
        peak: float
            Peak value of the delta function
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass

        """
        super(DeltaFunction, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                            minimum=peak, maximum=peak, check_range_nonzero=False)
        self.peak = peak
        self._is_fixed = True
        self.least_recently_sampled = peak

    def rescale(self, val):
        """Rescale everything to the peak with the correct shape.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Rescaled probability, equivalent to peak
        """
        return self.peak * val ** 0

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
         Union[float, array_like]: np.inf if val = peak, 0 otherwise

        """
        at_peak = (val == self.peak)
        return at_peak * 1.0

    def cdf(self, val):
        return 1.0 * (val > self.peak)


class PowerLaw(Prior):

    def __init__(self, alpha, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Power law with bounds and alpha, spectral index

        Parameters
        ==========
        alpha: float
            Power law exponent parameter
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(PowerLaw, self).__init__(name=name, latex_label=latex_label,
                                       minimum=minimum, maximum=maximum, unit=unit,
                                       boundary=boundary)
        self.alpha = alpha

    @xp_wrap
    def rescale(self, val, *, xp=np):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ==========
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        =======
        Union[float, array_like]: Rescaled probability
        """
        if self.alpha == -1:
            return self.minimum * xp.exp(val * xp.log(self.maximum / self.minimum))
        else:
            return (self.minimum ** (1 + self.alpha) + val *
                    (self.maximum ** (1 + self.alpha) - self.minimum ** (1 + self.alpha))) ** (1. / (1 + self.alpha))

    @xp_wrap
    def prob(self, val, *, xp=np):
        """Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        if self.alpha == -1:
            return xp.nan_to_num(1 / val / xp.log(self.maximum / self.minimum)) * self.is_in_prior_range(val)
        else:
            return xp.nan_to_num(val ** self.alpha * (1 + self.alpha) /
                                 (self.maximum ** (1 + self.alpha) -
                                  self.minimum ** (1 + self.alpha))) * self.is_in_prior_range(val)

    @xp_wrap
    def ln_prob(self, val, *, xp=np):
        """Return the logarithmic prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float:

        """
        if self.alpha == -1:
            normalising = 1. / xp.log(self.maximum / self.minimum)
        else:
            normalising = (1 + self.alpha) / (self.maximum ** (1 + self.alpha) -
                                              self.minimum ** (1 + self.alpha))

        with np.errstate(divide='ignore', invalid='ignore'):
            ln_in_range = xp.log(1. * self.is_in_prior_range(val))
            ln_p = self.alpha * xp.nan_to_num(xp.log(val)) + xp.log(normalising)

        return ln_p + ln_in_range

    @xp_wrap
    def cdf(self, val, *, xp=np):
        if self.alpha == -1:
            _cdf = xp.log(val / self.minimum) / xp.log(self.maximum / self.minimum)
        else:
            _cdf = (
                val ** (self.alpha + 1) - self.minimum ** (self.alpha + 1)
                / (self.maximum ** (self.alpha + 1) - self.minimum ** (self.alpha + 1))
            )
        _cdf = xp.clip(_cdf, 0, 1)
        return _cdf


class Uniform(Prior):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Uniform prior with bounds

        Parameters
        ==========
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Uniform, self).__init__(name=name, latex_label=latex_label,
                                      minimum=minimum, maximum=maximum, unit=unit,
                                      boundary=boundary)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ==========
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        =======
        Union[float, array_like]: Rescaled probability
        """
        return self.minimum + val * (self.maximum - self.minimum)

    def prob(self, val):
        """Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        return ((val >= self.minimum) & (val <= self.maximum)) / (self.maximum - self.minimum)

    @xp_wrap
    def ln_prob(self, val, *, xp=np):
        """Return the log prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: log probability of val
        """
        with np.errstate(divide="ignore"):
            return xp.log(self.prob(val))

    @xp_wrap
    def cdf(self, val, *, xp=np):
        _cdf = (val - self.minimum) / (self.maximum - self.minimum)
        return xp.clip(_cdf, 0, 1)


class LogUniform(PowerLaw):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Log-Uniform prior with bounds

        Parameters
        ==========
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(LogUniform, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                         minimum=minimum, maximum=maximum, alpha=-1, boundary=boundary)
        if self.minimum <= 0:
            logger.warning('You specified a uniform-in-log prior with minimum={}'.format(self.minimum))


class SymmetricLogUniform(Prior):

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Symmetric Log-Uniform distributions with bounds

        This is identical to a Log-Uniform distribution, but mirrored about
        the zero-axis and subsequently normalized. As such, the distribution
        has support on the two regions [-maximum, -minimum] and [minimum,
        maximum].

        Parameters
        ==========
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(SymmetricLogUniform, self).__init__(name=name, latex_label=latex_label,
                                                  minimum=minimum, maximum=maximum, unit=unit,
                                                  boundary=boundary)

    @xp_wrap
    def rescale(self, val, *, xp=np):
        """
        'Rescale' a sample from the unit line element to the power-law prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ==========
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        =======
        Union[float, array_like]: Rescaled probability
        """
        if isinstance(val, (float, int)):
            if val < 0.5:
                return -self.maximum * xp.exp(-2 * val * xp.log(self.maximum / self.minimum))
            else:
                return self.minimum * xp.exp(xp.log(self.maximum / self.minimum) * (2 * val - 1))
        else:
            vals_less_than_5 = val < 0.5
            rescaled = xp.empty_like(val)
            rescaled[vals_less_than_5] = -self.maximum * xp.exp(-2 * val[vals_less_than_5] *
                                                                xp.log(self.maximum / self.minimum))
            rescaled[~vals_less_than_5] = self.minimum * xp.exp(xp.log(self.maximum / self.minimum) *
                                                                (2 * val[~vals_less_than_5] - 1))
            return rescaled

    @xp_wrap
    def prob(self, val, *, xp=np):
        """Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        val = xp.abs(val)
        return (xp.nan_to_num(0.5 / val / xp.log(self.maximum / self.minimum)) *
                self.is_in_prior_range(val))

    @xp_wrap
    def ln_prob(self, val, *, xp=np):
        """Return the logarithmic prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float:

        """
        return np.nan_to_num(- xp.log(2 * xp.abs(val)) - xp.log(xp.log(self.maximum / self.minimum)))

    @xp_wrap
    def cdf(self, val, *, xp=np):
        asymmetric = LogUniform.cdf(self, abs(val), xp)
        return 0.5 * (1 + xp.sign(val) * asymmetric)


class Cosine(Prior):

    def __init__(self, minimum=-np.pi / 2, maximum=np.pi / 2, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Cosine prior with bounds

        Parameters
        ==========
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Cosine, self).__init__(minimum=minimum, maximum=maximum, name=name,
                                     latex_label=latex_label, unit=unit, boundary=boundary)

    @xp_wrap
    def rescale(self, val, *, xp=np):
        """
        'Rescale' a sample from the unit line element to a uniform in cosine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        norm = 1 / (xp.sin(self.maximum) - xp.sin(self.minimum))
        return xp.arcsin(val / norm + xp.sin(self.minimum))

    @xp_wrap
    def prob(self, val, *, xp=np):
        """Return the prior probability of val. Defined over [-pi/2, pi/2].

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        return xp.cos(val) / 2 * self.is_in_prior_range(val)

    @xp_wrap
    def cdf(self, val, *, xp=np):
        _cdf = (
            (xp.sin(val) - xp.sin(self.minimum)) /
            (xp.sin(self.maximum) - xp.sin(self.minimum))
        )
        _cdf *= val >= self.minimum
        _cdf *= val <= self.maximum
        _cdf += val > self.maximum
        return _cdf


class Sine(Prior):

    def __init__(self, minimum=0, maximum=np.pi, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Sine prior with bounds

        Parameters
        ==========
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Sine, self).__init__(minimum=minimum, maximum=maximum, name=name,
                                   latex_label=latex_label, unit=unit, boundary=boundary)

    @xp_wrap
    def rescale(self, val, *, xp=np):
        """
        'Rescale' a sample from the unit line element to a uniform in sine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        norm = 1 / (xp.cos(self.minimum) - xp.cos(self.maximum))
        return xp.arccos(xp.cos(self.minimum) - val / norm)

    @xp_wrap
    def prob(self, val, *, xp=np):
        """Return the prior probability of val. Defined over [0, pi].

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return xp.sin(val) / 2 * self.is_in_prior_range(val)

    @xp_wrap
    def cdf(self, val, *, xp=np):
        _cdf = (
            (xp.cos(val) - xp.cos(self.minimum))
            / (xp.cos(self.maximum) - xp.cos(self.minimum))
        )
        _cdf *= val >= self.minimum
        _cdf *= val <= self.maximum
        _cdf += val > self.maximum
        return _cdf


class Gaussian(Prior):

    def __init__(self, mu, sigma, name=None, latex_label=None, unit=None, boundary=None):
        """Gaussian prior with mean mu and width sigma

        Parameters
        ==========
        mu: float
            Mean of the Gaussian prior
        sigma:
            Width/Standard deviation of the Gaussian prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Gaussian, self).__init__(name=name, latex_label=latex_label, unit=unit, boundary=boundary)
        self.mu = mu
        self.sigma = sigma

    @xp_wrap
    def rescale(self, val, *, xp=np):
        """
        'Rescale' a sample from the unit line element to the appropriate Gaussian prior.

        Parameters
        ==========
        val: Union[float, int, array_like]

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        if "jax" in xp.__name__:
            from jax.scipy.special import erfinv
        else:
            from scipy.special import erfinv
        return self.mu + erfinv(2 * val - 1) * 2 ** 0.5 * self.sigma

    @xp_wrap
    def prob(self, val, *, xp=np):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return xp.exp(-(self.mu - val) ** 2 / (2 * self.sigma ** 2)) / (2 * np.pi) ** 0.5 / self.sigma

    @xp_wrap
    def ln_prob(self, val, *, xp=np):
        """Return the Log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return -0.5 * ((self.mu - val) ** 2 / self.sigma ** 2 + xp.log(2 * np.pi * self.sigma ** 2))

    def cdf(self, val):
        return (1 - erf((self.mu - val) / 2 ** 0.5 / self.sigma)) / 2


class Normal(Gaussian):
    """A synonym for the  Gaussian distribution. """


class TruncatedGaussian(Prior):

    def __init__(self, mu, sigma, minimum, maximum, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Truncated Gaussian prior with mean mu and width sigma

        https://en.wikipedia.org/wiki/Truncated_normal_distribution

        Parameters
        ==========
        mu: float
            Mean of the Gaussian prior
        sigma:
            Width/Standard deviation of the Gaussian prior
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(TruncatedGaussian, self).__init__(name=name, latex_label=latex_label, unit=unit,
                                                minimum=minimum, maximum=maximum, boundary=boundary)
        self.mu = mu
        self.sigma = sigma

    @property
    def normalisation(self):
        """ Calculates the proper normalisation of the truncated Gaussian

        Returns
        =======
        float: Proper normalisation of the truncated Gaussian
        """
        return (erf((self.maximum - self.mu) / 2 ** 0.5 / self.sigma) - erf(
            (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) / 2

    @xp_wrap
    def rescale(self, val, *, xp=np):
        """
        'Rescale' a sample from the unit line element to the appropriate truncated Gaussian prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        if "jax" in xp.__name__:
            from jax.scipy.special import erfinv
        else:
            from scipy.special import erfinv
        return erfinv(2 * val * self.normalisation + erf(
            (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) * 2 ** 0.5 * self.sigma + self.mu

    @xp_wrap
    def prob(self, val, *, xp=np):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        return xp.exp(-(self.mu - val) ** 2 / (2 * self.sigma ** 2)) / (2 * np.pi) ** 0.5 \
            / self.sigma / self.normalisation * self.is_in_prior_range(val)

    def cdf(self, val):
        _cdf = (erf((val - self.mu) / 2 ** 0.5 / self.sigma) - erf(
            (self.minimum - self.mu) / 2 ** 0.5 / self.sigma)) / 2 / self.normalisation
        _cdf *= val >= self.minimum
        _cdf *= val <= self.maximum
        _cdf += val > self.maximum
        return _cdf


class TruncatedNormal(TruncatedGaussian):
    """A synonym for the TruncatedGaussian distribution."""


class HalfGaussian(TruncatedGaussian):
    def __init__(self, sigma, name=None, latex_label=None, unit=None, boundary=None):
        """A Gaussian with its mode at zero, and truncated to only be positive.

        Parameters
        ==========
        sigma: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(HalfGaussian, self).__init__(mu=0., sigma=sigma, minimum=0., maximum=np.inf,
                                           name=name, latex_label=latex_label,
                                           unit=unit, boundary=boundary)


class HalfNormal(HalfGaussian):
    """A synonym for the HalfGaussian distribution."""


class LogNormal(Prior):
    def __init__(self, mu, sigma, name=None, latex_label=None, unit=None, boundary=None):
        """Log-normal prior with mean mu and width sigma

        https://en.wikipedia.org/wiki/Log-normal_distribution

        Parameters
        ==========
        mu: float
            Mean of the Gaussian prior
        sigma:
            Width/Standard deviation of the Gaussian prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(LogNormal, self).__init__(name=name, minimum=0., latex_label=latex_label,
                                        unit=unit, boundary=boundary)

        if sigma <= 0.:
            raise ValueError("For the LogGaussian prior the standard deviation must be positive")

        self.mu = mu
        self.sigma = sigma

    @xp_wrap
    def rescale(self, val, *, xp=np):
        """
        'Rescale' a sample from the unit line element to the appropriate LogNormal prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        if "jax" in xp.__name__:
            from jax.scipy.special import erfinv
        else:
            from scipy.special import erfinv
        return xp.exp(self.mu + (2 * self.sigma ** 2)**0.5 * erfinv(2 * val - 1))

    @xp_wrap
    def prob(self, val, *, xp=np):
        """Returns the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        if isinstance(val, (float, int)):
            if val <= self.minimum:
                _prob = 0.
            else:
                _prob = xp.exp(-(xp.log(val) - self.mu) ** 2 / self.sigma ** 2 / 2)\
                    / xp.sqrt(2 * np.pi) / val / self.sigma
        else:
            _prob = xp.zeros(val.size)
            idx = (val > self.minimum)
            _prob[idx] = xp.exp(-(xp.log(val[idx]) - self.mu) ** 2 / self.sigma ** 2 / 2)\
                / xp.sqrt(2 * np.pi) / val[idx] / self.sigma
        return _prob

    @xp_wrap
    def ln_prob(self, val, *, xp=np):
        """Returns the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        if isinstance(val, (float, int)):
            if val <= self.minimum:
                _ln_prob = -xp.inf
            else:
                _ln_prob = -(xp.log(val) - self.mu) ** 2 / self.sigma ** 2 / 2\
                    - xp.log(xp.sqrt(2 * np.pi) * val * self.sigma)
        else:
            _ln_prob = -xp.inf * xp.ones(val.size)
            idx = (val > self.minimum)
            _ln_prob[idx] = -(xp.log(val[idx]) - self.mu) ** 2\
                / self.sigma ** 2 / 2 - xp.log(xp.sqrt(2 * np.pi) * val[idx] * self.sigma)
        return _ln_prob

    @xp_wrap
    def cdf(self, val, *, xp=np):
        if isinstance(val, (float, int)):
            if val <= self.minimum:
                _cdf = 0.
            else:
                _cdf = 0.5 + erf((np.log(val) - self.mu) / self.sigma / np.sqrt(2)) / 2
        else:
            _cdf = xp.zeros(val.size)
            _cdf[val > self.minimum] = 0.5 + erf((
                xp.log(val[val > self.minimum]) - self.mu) / self.sigma / np.sqrt(2)) / 2
        return _cdf


class LogGaussian(LogNormal):
    """Synonym of LogNormal prior."""


class Exponential(Prior):
    def __init__(self, mu, name=None, latex_label=None, unit=None, boundary=None):
        """Exponential prior with mean mu

        Parameters
        ==========
        mu: float
            Mean of the Exponential prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Exponential, self).__init__(name=name, minimum=0., latex_label=latex_label,
                                          unit=unit, boundary=boundary)
        self.mu = mu

    @xp_wrap
    def rescale(self, val, *, xp=np):
        """
        'Rescale' a sample from the unit line element to the appropriate Exponential prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        return -self.mu * xp.log1p(-val)

    @xp_wrap
    def prob(self, val, *, xp=np):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        if isinstance(val, (float, int)):
            if val < self.minimum:
                _prob = 0.
            else:
                _prob = xp.exp(-val / self.mu) / self.mu
        else:
            _prob = xp.zeros(val.size)
            _prob[val >= self.minimum] = xp.exp(-val[val >= self.minimum] / self.mu) / self.mu
        return _prob

    @xp_wrap
    def ln_prob(self, val, *, xp=np):
        """Returns the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        if isinstance(val, (float, int)):
            if val < self.minimum:
                _ln_prob = -xp.inf
            else:
                _ln_prob = -val / self.mu - xp.log(self.mu)
        else:
            _ln_prob = -xp.inf * xp.ones(val.size)
            _ln_prob[val >= self.minimum] = -val[val >= self.minimum] / self.mu - xp.log(self.mu)
        return _ln_prob

    @xp_wrap
    def cdf(self, val, *, xp=np):
        if isinstance(val, (float, int)):
            if val < self.minimum:
                _cdf = 0.
            else:
                _cdf = 1. - xp.exp(-val / self.mu)
        else:
            _cdf = xp.zeros(val.size)
            _cdf[val >= self.minimum] = 1. - xp.exp(-val[val >= self.minimum] / self.mu)
        return _cdf


class StudentT(Prior):
    def __init__(self, df, mu=0., scale=1., name=None, latex_label=None,
                 unit=None, boundary=None):
        """Student's t-distribution prior with number of degrees of freedom df,
        mean mu and scale

        https://en.wikipedia.org/wiki/Student%27s_t-distribution#Generalized_Student's_t-distribution

        Parameters
        ==========
        df: float
            Number of degrees of freedom for distribution
        mu: float
            Mean of the Student's t-prior
        scale:
            Width of the Student's t-prior
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(StudentT, self).__init__(name=name, latex_label=latex_label, unit=unit, boundary=boundary)

        if df <= 0. or scale <= 0.:
            raise ValueError("For the StudentT prior the number of degrees of freedom and scale must be positive")

        self.df = df
        self.mu = mu
        self.scale = scale

    @xp_wrap
    def rescale(self, val, *, xp=np):
        """
        'Rescale' a sample from the unit line element to the appropriate Student's t-prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        if isinstance(val, (float, int)):
            if val == 0:
                rescaled = -np.inf
            elif val == 1:
                rescaled = np.inf
            else:
                rescaled = stdtrit(self.df, val) * self.scale + self.mu
        else:
            rescaled = stdtrit(self.df, val) * self.scale + self.mu
            rescaled[val == 0] = -xp.inf
            rescaled[val == 1] = xp.inf
        return rescaled

    @xp_wrap
    def prob(self, val, *, xp=np):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return xp.exp(self.ln_prob(val))

    @xp_wrap
    def ln_prob(self, val, *, xp=np):
        """Returns the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return gammaln(0.5 * (self.df + 1)) - gammaln(0.5 * self.df)\
            - xp.log((np.pi * self.df)**0.5 * self.scale) - (self.df + 1) / 2 *\
            xp.log(1 + ((val - self.mu) / self.scale) ** 2 / self.df)

    def cdf(self, val):
        return stdtr(self.df, (val - self.mu) / self.scale)


class Beta(Prior):
    def __init__(self, alpha, beta, minimum=0, maximum=1, name=None,
                 latex_label=None, unit=None, boundary=None):
        """Beta distribution

        https://en.wikipedia.org/wiki/Beta_distribution

        This wraps around
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

        Parameters
        ==========
        alpha: float
            first shape parameter
        beta: float
            second shape parameter
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Beta, self).__init__(minimum=minimum, maximum=maximum, name=name,
                                   latex_label=latex_label, unit=unit, boundary=boundary)

        if alpha <= 0. or beta <= 0.:
            raise ValueError("alpha and beta must both be positive values")

        self.alpha = alpha
        self.beta = beta

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Beta prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        return betaincinv(self.alpha, self.beta, val) * (self.maximum - self.minimum) + self.minimum

    @xp_wrap
    def prob(self, val, *, xp=np):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return xp.exp(self.ln_prob(val))

    @xp_wrap
    def ln_prob(self, val, *, xp=np):
        """Returns the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        _ln_prob = (
            xlogy(self.alpha - 1, val - self.minimum)
            + xlogy(self.beta - 1, self.maximum - val)
            - betaln(self.alpha, self.beta)
            - xlogy(self.alpha + self.beta - 1, self.maximum - self.minimum)
        )

        # deal with the fact that if alpha or beta are < 1 you get infinities at 0 and 1
        if isinstance(val, (float, int)):
            if xp.isfinite(_ln_prob) and self.minimum <= val <= self.maximum:
                return _ln_prob
            return -xp.inf
        else:
            _ln_prob_sub = xp.full_like(val, -xp.inf)
            idx = xp.isfinite(_ln_prob) & (val >= self.minimum) & (val <= self.maximum)
            _ln_prob_sub[idx] = _ln_prob[idx]
            return _ln_prob_sub

    @xp_wrap
    def cdf(self, val, *, xp=np):
        if isinstance(val, (float, int)):
            if val > self.maximum:
                return 1.
            elif val < self.minimum:
                return 0.
            else:
                return betainc(
                    self.alpha, self.beta,
                    (val - self.minimum) / (self.maximum - self.minimum)
                )
        else:
            _cdf = np.nan_to_num(betainc(self.alpha, self.beta,
                                 (val - self.minimum) / (self.maximum - self.minimum)))
            _cdf *= val >= self.minimum
            _cdf *= val <= self.maximum
            _cdf += val > self.maximum
            return _cdf


class Logistic(Prior):
    def __init__(self, mu, scale, name=None, latex_label=None, unit=None, boundary=None):
        """Logistic distribution

        https://en.wikipedia.org/wiki/Logistic_distribution

        Parameters
        ==========
        mu: float
            Mean of the distribution
        scale: float
            Width of the distribution
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Logistic, self).__init__(name=name, latex_label=latex_label, unit=unit, boundary=boundary)

        if scale <= 0.:
            raise ValueError("For the Logistic prior the scale must be positive")

        self.mu = mu
        self.scale = scale

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Logistic prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        if isinstance(val, (float, int)):
            if val == 0:
                rescaled = -np.inf
            elif val == 1:
                rescaled = np.inf
            else:
                rescaled = self.mu + self.scale * np.log(val / (1. - val))
        else:
            rescaled = np.inf * np.ones(val.size)
            rescaled[val == 0] = -np.inf
            rescaled[(val > 0) & (val < 1)] = self.mu + self.scale\
                * np.log(val[(val > 0) & (val < 1)] / (1. - val[(val > 0) & (val < 1)]))
        return rescaled

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return np.exp(self.ln_prob(val))

    def ln_prob(self, val):
        """Returns the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return -(val - self.mu) / self.scale -\
            2. * np.log(1. + np.exp(-(val - self.mu) / self.scale)) - np.log(self.scale)

    def cdf(self, val):
        return 1. / (1. + np.exp(-(val - self.mu) / self.scale))


class Cauchy(Prior):
    def __init__(self, alpha, beta, name=None, latex_label=None, unit=None, boundary=None):
        """Cauchy distribution

        https://en.wikipedia.org/wiki/Cauchy_distribution

        Parameters
        ==========
        alpha: float
            Location parameter
        beta: float
            Scale parameter
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Cauchy, self).__init__(name=name, latex_label=latex_label, unit=unit, boundary=boundary)

        if beta <= 0.:
            raise ValueError("For the Cauchy prior the scale must be positive")

        self.alpha = alpha
        self.beta = beta

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Cauchy prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        rescaled = self.alpha + self.beta * np.tan(np.pi * (val - 0.5))
        if isinstance(val, (float, int)):
            if val == 1:
                rescaled = np.inf
            elif val == 0:
                rescaled = -np.inf
        else:
            rescaled[val == 1] = np.inf
            rescaled[val == 0] = -np.inf
        return rescaled

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        return 1. / self.beta / np.pi / (1. + ((val - self.alpha) / self.beta) ** 2)

    def ln_prob(self, val):
        """Return the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Log prior probability of val
        """
        return - np.log(self.beta * np.pi) - np.log(1. + ((val - self.alpha) / self.beta) ** 2)

    def cdf(self, val):
        return 0.5 + np.arctan((val - self.alpha) / self.beta) / np.pi


class Lorentzian(Cauchy):
    """Synonym for the Cauchy distribution"""


class Gamma(Prior):
    def __init__(self, k, theta=1., name=None, latex_label=None, unit=None, boundary=None):
        """Gamma distribution

        https://en.wikipedia.org/wiki/Gamma_distribution

        Parameters
        ==========
        k: float
            The shape parameter
        theta: float
            The scale parameter
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(Gamma, self).__init__(name=name, minimum=0., latex_label=latex_label,
                                    unit=unit, boundary=boundary)

        if k <= 0 or theta <= 0:
            raise ValueError("For the Gamma prior the shape and scale must be positive")

        self.k = k
        self.theta = theta

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Gamma prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        return gammaincinv(self.k, val) * self.theta

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ==========
        val:  Union[float, int, array_like]

        Returns
        =======
         Union[float, array_like]: Prior probability of val
        """
        return np.exp(self.ln_prob(val))

    def ln_prob(self, val):
        """Returns the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        if isinstance(val, (float, int)):
            if val < self.minimum:
                _ln_prob = -np.inf
            else:
                _ln_prob = xlogy(self.k - 1, val) - val / self.theta - xlogy(self.k, self.theta) - gammaln(self.k)
        else:
            _ln_prob = -np.inf * np.ones(val.size)
            idx = (val >= self.minimum)
            _ln_prob[idx] = xlogy(self.k - 1, val[idx]) - val[idx] / self.theta\
                - xlogy(self.k, self.theta) - gammaln(self.k)
        return _ln_prob

    def cdf(self, val):
        if isinstance(val, (float, int)):
            if val < self.minimum:
                _cdf = 0.
            else:
                _cdf = gammainc(self.k, val / self.theta)
        else:
            _cdf = np.zeros(val.size)
            _cdf[val >= self.minimum] = gammainc(self.k, val[val >= self.minimum] / self.theta)
        return _cdf


class ChiSquared(Gamma):
    def __init__(self, nu, name=None, latex_label=None, unit=None, boundary=None):
        """Chi-squared distribution

        https://en.wikipedia.org/wiki/Chi-squared_distribution

        Parameters
        ==========
        nu: int
            Number of degrees of freedom
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """

        if nu <= 0 or not isinstance(nu, int):
            raise ValueError("For the ChiSquared prior the number of degrees of freedom must be a positive integer")

        super(ChiSquared, self).__init__(name=name, k=nu / 2., theta=2.,
                                         latex_label=latex_label, unit=unit, boundary=boundary)

    @property
    def nu(self):
        return int(self.k * 2)

    @nu.setter
    def nu(self, nu):
        self.k = nu / 2.


class FermiDirac(Prior):
    def __init__(self, sigma, mu=None, r=None, name=None, latex_label=None,
                 unit=None):
        """A Fermi-Dirac type prior, with a fixed lower boundary at zero
        (see, e.g. Section 2.3.5 of [1]_). The probability distribution
        is defined by Equation 22 of [1]_.

        Parameters
        ==========
        sigma: float (required)
            The range over which the attenuation of the distribution happens
        mu: float
            The point at which the distribution falls to 50% of its maximum
            value
        r: float
            A value giving mu/sigma. This can be used instead of specifying
            mu.
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass

        References
        ==========

        .. [1] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
           <https:arxiv.org/abs/1705.08978v1>`_, 2017.
        """
        super(FermiDirac, self).__init__(name=name, latex_label=latex_label, unit=unit, minimum=0.)

        self.sigma = sigma

        if mu is None and r is None:
            raise ValueError("For the Fermi-Dirac prior either a 'mu' value or 'r' "
                             "value must be given.")

        if r is None and mu is not None:
            self.mu = mu
            self.r = self.mu / self.sigma
        else:
            self.r = r
            self.mu = self.sigma * self.r

        if self.r <= 0. or self.sigma <= 0.:
            raise ValueError("For the Fermi-Dirac prior the values of sigma and r "
                             "must be positive.")

        self.expr = np.exp(self.r)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Fermi-Dirac prior.

        Parameters
        ==========
        val: Union[float, int, array_like]

        This maps to the inverse CDF. This has been analytically solved for this case,
        see Equation 24 of [1]_.

        References
        ==========

        .. [1] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
           <https:arxiv.org/abs/1705.08978v1>`_, 2017.
        """
        inv = -1 / self.expr + (1 + self.expr)**-val + (1 + self.expr)**-val / self.expr
        return -self.sigma * np.log(np.maximum(inv, 0))

    @xp_wrap
    def prob(self, val, *, xp=np):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        return (
            (xp.exp((val - self.mu) / self.sigma) + 1)**-1
            / (self.sigma * xp.log1p(self.expr))
            * (val >= self.minimum)
        )

    def ln_prob(self, val):
        """Return the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Log prior probability of val
        """
        return np.log(self.prob(val))

    def cdf(self, val):
        """
        Evaluate the CDF of the Fermi-Dirac distribution using a slightly
        modified form of Equation 23 of [1]_.

        Parameters
        ==========
        val: Union[float, int, array_like]
            The value(s) to evaluate the CDF at

        Returns
        =======
        Union[float, array_like]:
            The CDF value(s)

        References
        ==========

        .. [1] M. Pitkin, M. Isi, J. Veitch & G. Woan, `arXiv:1705.08978v1
           <https:arxiv.org/abs/1705.08978v1>`_, 2017.
        """
        result = (
            (np.logaddexp(0, -self.r) - np.logaddexp(-val / self.sigma, -self.r))
            / np.logaddexp(0, self.r)
        )
        return np.clip(result, 0, 1)


class DiscreteValues(Prior):
    def __init__(self, values, name=None, latex_label=None,
                 unit=None, boundary="periodic"):
        """ An equal-weighted discrete-valued prior

        Parameters
        ==========
        values: array
            The discrete values of the prior.
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        """
        nvalues = len(values)
        minimum = np.min(values)
        # Small delta added to help with MCMC walking
        maximum = np.max(values) * (1 + 1e-15)
        super(DiscreteValues, self).__init__(
            name=name, latex_label=latex_label, minimum=minimum,
            maximum=maximum, unit=unit, boundary=boundary)
        self.nvalues = nvalues
        self.values = np.sort(np.array(values))
        self.p = 1 / self.nvalues
        self.lnp = -np.log(self.nvalues)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the discrete-value prior.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ==========
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        =======
        Union[float, array_like]: Rescaled probability
        """
        idx = np.asarray(np.floor(val * self.nvalues), dtype=int)
        return self.values[idx]

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val
        """
        if isinstance(val, (float, int)):
            if val in self.values:
                return self.p
            else:
                return 0
        else:
            val = np.atleast_1d(val)
            probs = np.zeros_like(val, dtype=np.float64)
            idxs = np.isin(val, self.values)
            probs[idxs] = self.p
            return probs

    def ln_prob(self, val):
        """Return the logarithmic prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float:

        """
        if isinstance(val, (float, int)):
            if val in self.values:
                return self.lnp
            else:
                return -np.inf
        else:
            val = np.atleast_1d(val)
            probs = -np.inf * np.ones_like(val, dtype=np.float64)
            idxs = np.isin(val, self.values)
            probs[idxs] = self.lnp
            return probs


class Categorical(DiscreteValues):
    def __init__(self, ncategories, name=None, latex_label=None,
                 unit=None, boundary="periodic"):
        """ An equal-weighted Categorical prior

        Parameters
        ==========
        ncategories: int
            The number of available categories. The prior mass support is then
            integers [0, ncategories - 1].
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        """
        values = np.arange(0, ncategories)
        DiscreteValues.__init__(self, values=values, name=name, latex_label=latex_label,
                                unit=unit, boundary=boundary)


class Triangular(Prior):
    """
    Define a new prior class which draws from a triangular distribution.

    For distribution details see: wikipedia.org/wiki/Triangular_distribution

    Here, minimum <= mode <= maximum,
    where the mode has the highest pdf value.

    """
    def __init__(self, mode, minimum, maximum, name=None, latex_label=None, unit=None):
        super(Triangular, self).__init__(
            name=name,
            latex_label=latex_label,
            unit=unit,
            minimum=minimum,
            maximum=maximum,
        )
        self.mode = mode
        self.fractional_mode = (self.mode - self.minimum) / (
            self.maximum - self.minimum
        )
        self.scale = self.maximum - self.minimum
        self.rescaled_minimum = self.minimum - (self.minimum == self.mode) * self.scale
        self.rescaled_maximum = self.maximum + (self.maximum == self.mode) * self.scale

    def rescale(self, val):
        """
        'Rescale' a sample from standard uniform to a triangular distribution.

        This maps to the inverse CDF. This has been analytically solved for this case.

        Parameters
        ==========
        val: Union[float, int, array_like]
            Uniform probability

        Returns
        =======
        Union[float, array_like]: Rescaled probability

        """
        below_mode = (val * self.scale * (self.mode - self.minimum)) ** 0.5
        above_mode = ((1 - val) * self.scale * (self.maximum - self.mode)) ** 0.5
        return (self.minimum + below_mode) * (val < self.fractional_mode) + (
            self.maximum - above_mode
        ) * (val >= self.fractional_mode)

    def prob(self, val):
        """
        Return the prior probability of val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: Prior probability of val

        """
        between_minimum_and_mode = (
            (val < self.mode)
            * (self.minimum <= val)
            * (val - self.rescaled_minimum)
            / (self.mode - self.rescaled_minimum)
        )
        between_mode_and_maximum = (
            (self.mode <= val)
            * (val <= self.maximum)
            * (self.rescaled_maximum - val)
            / (self.rescaled_maximum - self.mode)
        )
        return 2.0 * (between_minimum_and_mode + between_mode_and_maximum) / self.scale

    def cdf(self, val):
        """
        Return the prior cumulative probability at val

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        float: prior cumulative probability at val

        """
        return (
            (val > self.mode)
            + (val > self.minimum)
            * (val <= self.maximum)
            / (self.scale)
            * (
                (val > self.mode)
                * (self.rescaled_maximum - val) ** 2.0
                / (self.mode - self.rescaled_maximum)
                + (val <= self.mode)
                * (val - self.rescaled_minimum) ** 2.0
                / (self.mode - self.rescaled_minimum)
            )
        )

    betaln,
