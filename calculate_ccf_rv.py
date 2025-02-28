#!/usr/bin/python

HRMOS_bandsName = ['UVB', 'G', 'R1', 'R2']


#imports:

from typing import List, Optional

from bisect import bisect_left
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt

from numpy import exp, log, sqrt
from scipy import optimize


# My functions:

def espdr_compute_CCF_fast(ll, dll, flux, error, blaze, quality, RV_table,
                           mask, berv, bervmax, mask_width=0.5):
    """
    Compute the CCF of a spectrum using a mask
    taken from iCCF: https://github.com/j-faria/iCCF
    """

    c = 299792.458

    nx_s2d = flux.size
    # ny_s2d = 1  #! since this function computes only one order
    n_mask = mask.size
    nx_ccf = len(RV_table)

    ccf_flux = np.zeros_like(RV_table)
    ccf_error = np.zeros_like(RV_table)
    ccf_quality = np.zeros_like(RV_table)

    dll2 = dll / 2.0  # cpl_image_divide_scalar_create(dll,2.);
    ll2 = ll - dll2  # cpl_image_subtract_create(ll,dll2);

    #? this mimics the pipeline (note that cpl_image_get indices start at 1)
    imin, imax = 1, nx_s2d
    while(imin < nx_s2d and quality[imin-1] != 0):
        imin += 1
    while(imax > 1 and quality[imax-1] != 0):
        imax -= 1

    if imin >= imax:
        return
    #? note that cpl_image_get indices start at 1, hence the "-1"s
    llmin = ll[imin + 1 - 1] / (1. + berv / c) * (1. + bervmax / c) / (1. + RV_table[0] / c)
    llmax = ll[imax - 1 - 1] / (1. + berv / c) * (1. - bervmax / c) / (1. + RV_table[nx_ccf - 1] / c)

    imin, imax = 0, n_mask - 1

    #? turns out cpl_table_get indices start at 0...
    while (imin < n_mask and mask['lambda'][imin] < (llmin + 0.5 * mask_width / c * llmin)):
        imin += 1
    while (imax >= 0     and mask['lambda'][imax] > (llmax - 0.5 * mask_width / c * llmax)):
        imax -= 1

    for i in range(imin, imax + 1):
        #? cpl_array_get indices also start at 0
        llcenter = mask['lambda'][i] * (1. + RV_table[nx_ccf // 2] / c)

        # index_center = 1
        # while(ll[index_center-1] < llcenter): index_center += 1
        # my attempt to speed it up
        # index_center = np.where(ll < llcenter)[0][-1] + 1
        index_center = bisect_left(ll, llcenter) + 1

        contrast = mask['contrast'][i]
        w = contrast * contrast

        for j in range(0, nx_ccf):
            llcenter = mask['lambda'][i] * (1. + RV_table[j] / c)
            llstart = llcenter - 0.5 * mask_width / c * llcenter
            llstop = llcenter + 0.5 * mask_width / c * llcenter

            # index1 = 1
            # while(ll2[index1-1] < llstart): index1 += 1
            index1 = bisect_left(ll2, llstart) + 1

            # index2 = index1
            # while (ll2[index2-1] < llcenter): index2 += 1
            index2 = bisect_left(ll2, llcenter) + 1

            # index3 = index2
            # while (ll2[index3-1] < llstop): index3 += 1;
            index3 = bisect_left(ll2, llstop) + 1

            k = j

            for index in range(index1, index3):
                ccf_flux[k] += w * flux[index-1] / blaze[index-1] * blaze[index_center-1]  # noqa: E501

            ccf_flux[k] += w * flux[index1-1-1] * (ll2[index1-1]-llstart) / dll[index1-1-1] / blaze[index1-1-1] * blaze[index_center-1]
            ccf_flux[k] -= w * flux[index3-1-1] * (ll2[index3-1]-llstop) / dll[index3-1-1] / blaze[index3-1-1] * blaze[index_center-1]

            ccf_error[k] += w * w * error[index2 - 1 - 1] * error[index2 - 1 - 1]

            ccf_quality[k] += quality[index2 - 1 - 1]

    # my_error = cpl_image_power(*CCF_error_RE,0.5);
    ccf_error = np.sqrt(ccf_error)

    return ccf_flux, ccf_error, ccf_quality


def calculate_hrmos_ccf(hrmosfile, rvarray, bands='all',
                      mask_file='ESPRESSO_G2.fits', mask=None, mask_width=0.5,
                      debug=False):

    with fits.open(hrmosfile) as hdu:

        if bands == 'all':
            if debug:
                print('can only debug one order at a time...')
                return
            bandsNames = HRMOS_bandsName
            return_sum = True
        else:
            assert isinstance(bands in HRMOS_bandsName, bool), 'band should be in the list'+ HRMOS_bandsName
            bandsNames = [bands]
            return_sum = False

        BERV = hdu[0].header['HIERARCH ESO QC BERV']
        BERVMAX = hdu[0].header['HIERARCH ESO QC BERVMAX']

        # CCF mask
        if mask is None:
            with fits.open(mask_file) as mask_hdu:
                mask = mask_hdu[1].data
        else:
            assert 'lambda' in mask, 'mask must contain the "lambda" key'
            assert 'contrast' in mask, 'mask must contain the "contrast" key'

#        # get the flux correction stored in the S2D file
#        keyword = 'HIERARCH ESO QC ORDER%d FLUX CORR'
#        flux_corr = [hdu[0].header[keyword % (o + 1)] for o in range(170)]

        ccfs, ccfes = [], []

#        with fits.open(dllfile) as hdu_dll:
#            dll_array = hdu_dll[1].data

#        with fits.open(blazefile) as hdu_blaze:
#            blaze_array = hdu_blaze[1].data

        for band in bandsNames:

#        dllfile = hdu[0].header['HIERARCH ESO PRO REC1 CAL7 NAME']
#        blazefile = hdu[0].header['HIERARCH ESO PRO REC1 CAL13 NAME']
#        print('need', dllfile)
#        print('need', blazefile)

            # WAVEDATA_AIR_BARY
            ll = hdu[band].data['wavelength']
            # mean w
            #llc = np.mean(hdu[5].data, axis=1)

            dll = hdu[band].data['dll']
#            # fit an 8th degree polynomial to the flux correction
#            corr_model = np.polyval(np.polyfit(llc, flux_corr, 7), llc)

            flux = hdu[band].data['flux']
            error = hdu[band].data['error']
            quality = hdu[band].data['quality']

            blaze = hdu[band].data['blaze']

            y = flux * blaze #/ corr_model[order]
            # y = np.loadtxt('flux_in_pipeline_order0.txt')
            ye = error * blaze# / corr_model[order]

            if debug:
                return ll, dll, y, ye, blaze, quality, rvarray, mask, BERV, BERVMAX

            print('calculating ccf (band %s)...' % band)
            ccf, ccfe, _ = espdr_compute_CCF_fast(ll, dll, y, ye, blaze, quality,
                                                  rvarray, mask, BERV, BERVMAX,
                                                  mask_width=mask_width)
            ccfs.append(ccf)
            ccfes.append(ccfe)

    if return_sum:
#        ccf = np.concatenate([ccfs, np.array(ccfs).sum(axis=0, keepdims=True)])
#        ccfe = np.concatenate([ccfes, np.zeros(len(rvarray)).reshape(1, -1)])
        # what to do with the errors?
        return np.array(ccfs), np.array(ccfes)
    else:
        return np.array(ccfs[0]), np.array(ccfes[0])


def _gauss_initial_guess(x, y):
    """ Educated guess (from the data) for Gaussian parameters. """
    # these guesses tend to work better for narrow-ish gaussians
    p0 = []

    # guess the amplitude (force it to be negative)
    p0.append(-abs(np.ptp(y)))

    # guess the center, but maybe the CCF is upside down?
    m = y.mean()
    ups_down = np.sign(np.percentile(y, 50) - m) != np.sign(y.max() - m)
    if ups_down:  # seems like it
        # warnings.warn('It seems the CCF might be upside-down?')
        p0.append(x[y.argmax()])
    else:
        p0[0] *= -1
        p0.append(x[y.argmin()])
    # guess the width
    p0.append(1)
    # guess the offset
    p0.append(0.5 * (y[0] + y[-1]))

    return p0

def gauss(x, p):
    """ A Gaussian function with parameters p = [A, x0, σ, offset]. """
    return p[0] * exp(-(x - p[1])**2 / (2 * p[2]**2)) + p[3]


def _gauss_partial_deriv(x, p):
    """ Partial derivatives of a Gaussian with respect to each parameter. """
    A, x0, sig, _ = p
    g = gauss(x, [A, x0, sig, 0.0])
    dgdA = gauss(x, [1.0, x0, sig, 0.0])
    dgdx0 = g * ((x - x0) / sig**2)
    dgdsig = g * ((x - x0)**2 / sig**3)
    dgdoffset = np.ones_like(x)
    return np.c_[dgdA, dgdx0, dgdsig, dgdoffset]

def gaussfit(x: np.ndarray,
             y: np.ndarray,
             p0: Optional[List] = None,
             yerr: Optional[np.ndarray] = None,
             return_errors: bool = False,
             use_deriv: bool = True,
             guess_rv: Optional[float] = None,
             **kwargs) -> List:
    """
    Fit a Gaussian function to `x`,`y` (and, if provided, `yerr`) using
    least-squares, with initial guess `p0` = [A, x0, σ, offset]. If p0 is not
    provided, the function tries an educated guess, which might lead to bad
    results.

    Args:
        x (array):
            The independent variable where the data is measured
        y (array):
            The dependent data.
        p0 (list or array):
            Initial guess for the parameters. If None, try to guess them from x,y.
        return_errors (bool):
            Whether to return estimated errors on the parameters.
        use_deriv (bool):
            Whether to use partial derivatives of the Gaussian (wrt the parameters)
            as Jacobian in the fit. If False, the Jacobian will be estimated.
        guess_rv (float):
            Initial guess for the RV (x0)
        **kwargs:
            Keyword arguments passed to `scipy.optimize.curve_fit`

    Returns:
        p (array):
            Best-fit values of the four parameters [A, x0, σ, offset]
        err (array):
            Estimated uncertainties on the four parameters
            (only if `return_errors=True`)
    """
    if (y == 0).all():
        return np.full(4, np.nan)

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    if p0 is None:
        p0 = _gauss_initial_guess(x, y)
    if guess_rv is not None:
        p0[1] = guess_rv

    f = lambda x, *p: gauss(x, p)
    if use_deriv:
        df = lambda x, *p: _gauss_partial_deriv(x, p)
    else:
        df = None

    pfit, pcov, *_ = optimize.curve_fit(f, x, y, p0=p0, sigma=yerr, jac=df,
                                        xtol=1e-12, ftol=1e-14, check_finite=True, **kwargs)

    if 'full_output' in kwargs:
            infodict = _

    if return_errors:
        errors = np.sqrt(np.diag(pcov))
        if 'full_output' in kwargs:
            return pfit, errors, infodict
        return pfit, errors

    if 'full_output' in kwargs:
        return pfit, infodict

    return pfit



def getRV(rv, ccf, eccf=None, **kwargs):
    """
    Calculate the radial velocity as the center of a Gaussian fit the CCF.
    
    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    kwargs : dict
        Keyword arguments passed directly to gaussfit
    """
    _, rv, _, _ = gaussfit(rv, ccf, yerr=eccf, **kwargs)
    return rv


def numerical_gradient(rv, ccf):
    """
    Return the gradient of the CCF.

    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.

    Notes
    -----
    The gradient is computed using the np.gradient routine, which uses second
    order accurate central differences in the interior points and either first
    or second order accurate one-sides (forward or backwards) differences at
    the boundaries. The gradient has the same shape as the input array.
    """
    return np.gradient(ccf, rv)

def getRVerror(rv, ccf, eccf):
    """
    Calculate the uncertainty on the radial velocity, following the same steps
    as the ESPRESSO DRS pipeline.

    Parameters
    ----------
    rv : array
        The velocity values where the CCF is defined.
    ccf : array
        The values of the CCF profile.
    eccf : array
        The errors on each value of the CCF profile.
    """
    ccf_slope = numerical_gradient(rv, ccf)
    ccf_sum = np.sum((ccf_slope / eccf)**2)
    return 1.0 / sqrt(ccf_sum)


### Main program:
def main():
    hrmosfile = 'output_spectra/r.HRMOS.2023-10-23T01:55:05.930.fits'
    vstart = -50
    vstep = 0.1
    size = 800
    rvarray = np.arange(vstart,vstart+vstep*size,vstep)

    ccf, ccfe = calculate_hrmos_ccf(hrmosfile, rvarray, bands='all',
                      mask_file='data/ESPRESSO_G2.fits', mask=None, mask_width=0.5,
                      debug=False)
#    print(rvarray, ccf)
    print(ccf.shape)
    for i, c in enumerate(ccf):
        ec = ccfe[i]
        plt.plot(rvarray, c)
        RV = getRV(rvarray, c)
        eRV = getRVerror(rvarray, c, ec)
        print(RV, eRV)
    plt.show()




if __name__ == "__main__":
    main()
