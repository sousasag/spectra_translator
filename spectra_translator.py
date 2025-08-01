#!/usr/bin/python


#imports:

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from PyAstronomy import pyasl
import glob


# My functions:

HRMOS_R = 80000
HRMOS_pixel_sampling = 2.7

HRMOS_bands = [ (383,417), 
                (498,542),
                (556,604),
                (623,677),
                ]

HRMOS_bandsName = ['B', 'G', 'R1', 'R2']
HRMOS_bandsWaveCentral = [400, 520, 575, 650]

# SNR relation from the ETC, with ADC
HRMOS_SNR_3600_V9 = [680,1080,1200,1328]

NO_ADC = True
airmass = 2.0

def get_flux_loss(wavei, airmass=1.5):
    """
    Get the flux loss for a given wavelength and airmass when no ADC is used.
    """
    wave1, floss1 = np.loadtxt("data/HRMOS_ETC/flux_loss_no_ADC_airmass1.txt", unpack=True, skiprows=1)
    wave2, floss2 = np.loadtxt("data/HRMOS_ETC/flux_loss_no_ADC_airmass1.5.txt", unpack=True, skiprows=1)

    fl1 = np.interp(wavei, wave1, floss1)
    fl2 = np.interp(wavei, wave2, floss2)
    flux_loss = (fl2-fl1)/0.5*(airmass-1)+ fl1  # linear interpolation between the two values
    return flux_loss


def adjust_SNR_no_ADC(SNR_bands, wave_central, airmass=1.5):
    """
    Adjust the SNR for the HRMOS bands without ADC
    """
    SNR_bands_no_ADC = []
    for i, snr in enumerate(SNR_bands):
            wavei = wave_central[i]
            flux_loss = get_flux_loss(wavei, airmass)
            print(i, wavei,flux_loss, snr)
            SNR_bands_no_ADC.append(snr * np.sqrt(1 -flux_loss/100))
    return SNR_bands_no_ADC

if NO_ADC:
    HRMOS_SNR_3600_V9 = adjust_SNR_no_ADC(HRMOS_SNR_3600_V9, HRMOS_bandsWaveCentral, airmass=airmass)


def get_ESPRESSO_spectra(filename):
    # Get the ESPRESSO spectra
    data = fits.getdata(filename)
    header = fits.getheader(filename)
    print(data.columns)
    wave = data['wavelength_air']
    flux = data['flux']
    error = data['error']
    wavev = data['wavelength']
    quality = data['quality']
    return wave, flux, error, header, wavev, quality


def get_hrmos_filename(filename_espresso):
    name = filename_espresso.split('/')[-1]
    name = name.replace('ESPRE', 'HRMOS')
    name = name.replace('_S1D_A', '')
    return name

def convolve_data(wavelength, flux,  R_ori, R_out):
    """
    Convolve the spectrum with a given resolution and transform it to a new lower resolution
    """
    print(R_ori, R_out)
    if R_out >= R_ori:
        print("Nothing to do!")
        return flux
    resolution = R_out
    c = 299792
    DU_espresso = c / R_ori
    R_new = resolution
    DU_new = c / R_new
    DU_conv = ((DU_new) ** 2 - (DU_espresso) ** 2) ** (0.5)
    R_conv = c / (DU_conv)
    print(DU_new, DU_conv, R_conv)
    newflux = pyasl.instrBroadGaussFast(
        wavelength, flux, R_conv, edgeHandling="firstlast", fullout=False, maxsig=None, equid=True,
    )
    return newflux


def blaze_model(wave, model='curved', peak=0.5,border=0.25):
    """
    Introduce the blaze model for a given spectral band
    """
    if model == 'flat':
        return np.ones(len(wave))
    else:
        n = len(wave)
        x = np.arange(n)
        c = (peak-border)*4
        y = -c*x**2/n**2 + c/n*x + border
        return y


def get_deltalambdaR_sampling2(wavei, wavef, pixel_sampling=2.7, delta_sampling=0.2, R=80000):
    """
    Get the wavelenght array for an averaged fixed delta sampling assuming a decreasing dll for a given order (similar to ESPRESSO)
    """
    npoints = 1.2* (wavef-wavei) / ((wavef+wavei)/2/R) * pixel_sampling
    sampling=np.arange(npoints)/npoints*delta_sampling*2+(pixel_sampling-delta_sampling)
    wave = [wavei]
    i=0
    while wave[-1] < wavef:
        wave.append(wave[-1] + wave[-1]/R/sampling[i])
        i+=1
    wave = np.array(wave)
    sampling = sampling[:len(wave)]
    dll = (wave+(wave/R/sampling)/2)-(wave-(wave/R/sampling)/2)
    return wave, dll


def simple_norm(wave, flux, iterations=4):
    """
    Simple normalization of the spectrum
    """
    pfit = np.polyfit(wave, flux, 1)
    for i in range(iterations):
        wi = []
        fi = []
        for j in range(len(wave)):
            if flux[j] > pfit[0]*wave[j]+pfit[1]:
                wi.append(wave[j])
                fi.append(flux[j])
        pfit = np.polyfit(wi, fi, 1)
    norm = flux/(pfit[0]*wave+pfit[1])    
    return norm

def add_noise(wave, flux, error, blaze, snr_center_out=100):
    """
    Add noise to the spectrum
    """
    snr_orig = flux / error
    if snr_center_out >= np.max(snr_orig):
        print ("WARNING: Problem with the SNR: Original data has lower SNR")
        return flux, error, snr_orig
    snr_outb = snr_center_out * blaze / np.max(blaze)
    snr_origf = np.sqrt(flux)
    snr_out = snr_origf / np.max(snr_origf) * snr_center_out
    snr_norm = simple_norm(wave,snr_out, iterations=8)
    snr_out2 = snr_norm * snr_outb

    sigma_out=flux/snr_out2
    sigma_orig=flux/snr_orig
    sigma2 = np.sqrt(sigma_out**2 - sigma_orig**2)
    noise_add = np.array([np.random.normal(0,s,1)[0] for s in sigma2])
    flux_out = flux + noise_add
    error_out = np.sqrt(error**2 + sigma2**2)

    return flux_out, error_out, snr_out2



def get_HRMOS_bands(spectral_data, R=HRMOS_R, pixel_samplint=HRMOS_pixel_sampling, peakSNR=100, plots=False):
    """
    Get the HRMOS bands from the ESPRESSO spectrum
    """
    wave, flux, flux80000, error, quality, header = spectral_data
    bands_spec = []
    snr_bands = compute_snrs_bands(peakSNR)
    for i, (bi,bf) in enumerate(HRMOS_bands):
        bi *= 10
        bf *= 10
        ib1 = np.where((wave > bi) & (wave < bf))
        blaze = blaze_model(wave[ib1], model='curved')
        w, f, f8, e, b = wave[ib1], flux[ib1], flux80000[ib1], error[ib1], blaze

        waves_int, dll = get_deltalambdaR_sampling2(w[0], w[-1], pixel_sampling=HRMOS_pixel_sampling, R=R) 
        f8_int = np.interp(waves_int, w, f8)
        e_int  = np.interp(waves_int, w, e)
        b_int  = np.interp(waves_int, w, b)
        quality_int = np.zeros(len(waves_int))  # quality zero for all pixels at the momment
        if peakSNR > 0:
            fluxn, errorn, snr_out = add_noise(waves_int, f8_int, e_int, b_int, snr_center_out=snr_bands[i])
            fluxo = fluxn
            erroro = errorn
        else:
            fluxo = f8_int
            erroro = e_int        
        if plots:
            plt.plot(w, f, 'b-', label="Input")
            plt.plot(w, f8, 'r--p', label="Broadened curve (full)") 
            plt.show()
        bands_spec.append((waves_int, fluxo, erroro, b_int, dll, quality_int))
    return bands_spec

def espresso2HRMOS(filein, fileout, peakSNR=100):
    """
    Read an S1D ESPRESSO FILE and generate a simulated HRMOS spectrum
    """
    wave, flux, error, header, wavev, quality = get_ESPRESSO_spectra(filein)
    res_ori = 140000
    if header["HIERARCH ESO INS MODE"] == "SINGLEUHR":
        res_ori = 190000
    fluxR       = convolve_data(wave, flux, res_ori, HRMOS_R)
    #fluxR = flux.copy()


    spectral_data = (wave, flux, fluxR, error, quality, header)
    bands_spec = get_HRMOS_bands(spectral_data, R=HRMOS_R, peakSNR=peakSNR)

    hduold = fits.open(filein)
    #hduold[-1].header['EXTNAME'] = "ESPRESSO"
    hduold.pop(1)
    snr_bands = compute_snrs_bands(peakSNR)
    for i, (waves_int,flux,error,b_int, dll, quality_int) in enumerate(bands_spec):
        col1 = fits.Column(name='wavelength', format = '1D', unit = 'angstrom', array=waves_int)
        col2 = fits.Column(name='flux'      , format = '1D', unit = 'e-'      , array=flux)
        col3 = fits.Column(name='error'     , format = '1D', unit = 'e-'      , array=error)
        col4 = fits.Column(name='blaze'     , format = '1D', unit = 'REL'     , array=b_int)
        col5 = fits.Column(name='dll'       , format = '1D', unit = 'angstrom', array=dll)
        col6 = fits.Column(name='quality'   , format = '1D', unit = ''        , array=quality_int)
        coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6])
        hdu = fits.BinTableHDU.from_columns(coldefs)
        hdu.header['EXTNAME'] = HRMOS_bandsName[i]
        hdu.header['SNR'] = snr_bands[i]
        hdu.header['RES'] = HRMOS_R
        hdu.header['PixSamp'] = HRMOS_pixel_sampling
        hduold.append(hdu)
    hduold.writeto(fileout, overwrite=True)

def compute_snrs_bands(SNR_R2):
    """
    Compute the SNR for the HRMOS bands
    """
    snrs = []
    for i in range(4):
        snrs.append(SNR_R2 * HRMOS_SNR_3600_V9[i] / HRMOS_SNR_3600_V9[-1])
    return snrs

### Main program:
def main():



    filein = "spectra/ESPRESSO/TauCeti/r.ESPRE.2023-01-08T01:30:19.668_S1D_A.fits"
    fileout = "output_spectra/TauCeti50NOADC/" + get_hrmos_filename(filein)
    SNR_R2 = 50
    espresso2HRMOS(filein, fileout, peakSNR=SNR_R2)
    return


    files = glob.glob("spectra/ESPRESSO/TauCeti/*.fits")

    SNR_R2 = 50
    print(compute_snrs_bands(SNR_R2)) 
    for filein in files:
        fileout = "output_spectra/TauCeti50/" + get_hrmos_filename(filein)
        espresso2HRMOS(filein, fileout, peakSNR=SNR_R2)


if __name__ == "__main__":
    main()

