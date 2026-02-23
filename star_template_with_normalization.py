# ==== DOWNLOAD HD 209458 DATA FROM ESO ARCHIVE AND PLOT ====

from astroquery.eso import Eso
from astropy.io import fits
from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt
import os

from specutils import Spectrum, SpectralRegion
from specutils.manipulation import extract_region
import astropy.units as u


eso = Eso()
query = 'HD 209458'
instrument = 'HARPS'

results = eso.query_surveys(target=query, surveys=instrument)
print(results.colnames)
print(results[:10])

if "Product category" in results.colnames:
    spec_rows = [("SPECTRUM" in str(x)) for x in results["Product category"]]
    spec = results[spec_rows]
else:
    spec = results
print("Total rows:", len(results))
print("Spectrum-like rows:", len(spec))
print(spec[:10])

use = spec if len(spec) > 0 else results
arc = use["ARCFILE"][0]
print("Downloading ARCFILE:", arc)
downloaded = eso.retrieve_data(arc)
print("Downloaded:", downloaded)

path = downloaded[0] if isinstance(downloaded, (list, tuple)) else downloaded
hdul = fits.open(path)
hdul.info()

star = query
star_data = fits.open(path)
data = star_data[1].data
print(data.columns)

xvals = data['WAVE'][0]
yvals = data['FLUX'][0]

xmin = xvals[0]
xmax = xvals[-1]
print("Wavelength range:", xmin, "to", xmax)

plt.plot(xvals, yvals, color='tab:pink')
plt.xlabel('Wavelength (Å)')
plt.ylabel('Flux')
plt.title(f'Full Spectrum: {star}, Instrument: {instrument}\nWavelengths from {xmin:.2f} to {xmax:.2f} Å')
plt.autoscale(enable=True, axis='both', tight=True)
filename = f"Spectrum_{star}_{instrument}_{xmin:.2f}-{xmax:.2f}.png"
plt.savefig(filename)
plt.show()


# ==== LOCAL CONTINUUM NORMALIZATION ====

spectrum = Spectrum(
    spectral_axis=xvals * u.AA,
    flux=yvals * u.Unit('erg / (cm^2 s AA)')
)

def local_normalize(spectrum, line_center, window=20, shoulder=5):
    """
    Locally normalize a spectrum around a single spectral line.

    Parameters:
        spectrum    : Spectrum object
        line_center : float, central wavelength in Angstroms
        window      : float, half-width of region to extract (default ±20 Å)
        shoulder    : float, width of continuum shoulder on each side (default ±5 Å)

    Returns:
        wave_norm   : wavelength array (Å)
        flux_norm   : normalized flux (continuum ~ 1.0, absorption lines dip below)
    """
    lo = line_center - window
    hi = line_center + window

    region = SpectralRegion(lo * u.AA, hi * u.AA)
    try:
        sub = extract_region(spectrum, region)
    except Exception:
        print(f"Warning: {lo}-{hi} Å not covered for this spectrum, skipping.")
        return None, None

    wave = sub.spectral_axis.value
    flux = sub.flux.value

    # fit continuum on shoulder regions only, masking out the line core
    shoulder_mask = (wave < line_center - shoulder) | (wave > line_center + shoulder)

    try:
        p_init = models.Chebyshev1D(1)
        fitter = fitting.LinearLSQFitter()
        p = fitter(p_init, wave[shoulder_mask], flux[shoulder_mask])
        continuum = p(wave)
        flux_norm = flux / continuum
    except Exception as e:
        print(f"Warning: continuum fit failed around {line_center} Å: {e}")
        return None, None

    return wave, flux_norm


# normalize around each feature
wave_ha,   flux_ha   = local_normalize(spectrum, line_center=6562.801, window=40, shoulder=8)   # H-alpha
wave_li,   flux_li   = local_normalize(spectrum, line_center=6707.76,  window=15, shoulder=4)   # Lithium — Fe line at 6707.4 nearby
wave_cahk, flux_cahk = local_normalize(spectrum, line_center=3950.0,   window=30, shoulder=8)   # Ca II H&K


# ==== 4-PANEL PLOT ====
figures, graphs = plt.subplots(2, 2, figsize=(10, 8))
plt.suptitle(f'\nLocally Normalized Key Spectral Features for {star}', fontsize=16)
plt.subplots_adjust(hspace=0.3)

# H-alpha
if wave_ha is not None:
    graphs[0,0].plot(wave_ha, flux_ha, 'tab:pink')
    graphs[0,0].axvline(x=6562.801, color='black', ls='--')
    graphs[0,0].axhline(y=1.0, color='gray', ls=':', linewidth=0.8)
    pad = 0.1 * (flux_ha.max() - flux_ha.min())
    graphs[0,0].set_ylim(flux_ha.min() - pad, flux_ha.max() + pad)
else:
    graphs[0,0].text(0.5, 0.5, 'Not covered', ha='center', va='center', transform=graphs[0,0].transAxes)
graphs[0,0].set_title(r'H-alpha 6562.801 $\AA$')
graphs[0,0].set_xlabel('Wavelength (Å)')
graphs[0,0].set_ylabel('Normalized Flux')

# Lithium + Fe line marker
if wave_li is not None:
    graphs[0,1].plot(wave_li, flux_li, 'tab:purple')
    graphs[0,1].axvline(x=6707.76, color='black', ls='--', label='Li 6707.76')
    graphs[0,1].axvline(x=6707.4,  color='red',   ls=':',  label='Fe 6707.4')
    graphs[0,1].axhline(y=1.0, color='gray', ls=':', linewidth=0.8)
    pad = 0.1 * (flux_li.max() - flux_li.min())
    graphs[0,1].set_ylim(flux_li.min() - pad, flux_li.max() + pad)
    graphs[0,1].legend(fontsize=7)
else:
    graphs[0,1].text(0.5, 0.5, 'Not covered', ha='center', va='center', transform=graphs[0,1].transAxes)
graphs[0,1].set_title(r'Lithium 6707.76 $\AA$')
graphs[0,1].set_xlabel('Wavelength (Å)')
graphs[0,1].set_ylabel('Normalized Flux')

# Ca II H
if wave_cahk is not None:
    graphs[1,0].plot(wave_cahk, flux_cahk, '#71C7A0')
    graphs[1,0].axvline(x=3933, color='black', ls='--')
    graphs[1,0].axhline(y=1.0, color='gray', ls=':', linewidth=0.8)
    graphs[1,0].set_xlim(3920, 3945)
    pad = 0.1 * (flux_cahk.max() - flux_cahk.min())
    graphs[1,0].set_ylim(flux_cahk.min() - pad, flux_cahk.max() + pad)
else:
    graphs[1,0].text(0.5, 0.5, 'Not covered', ha='center', va='center', transform=graphs[1,0].transAxes)
graphs[1,0].set_title(r'Ca II H 3933 $\AA$')
graphs[1,0].set_xlabel('Wavelength (Å)')
graphs[1,0].set_ylabel('Normalized Flux')

# Ca II K
if wave_cahk is not None:
    graphs[1,1].plot(wave_cahk, flux_cahk, '#4B9CD3')
    graphs[1,1].axvline(x=3968, color='black', ls='--')
    graphs[1,1].axhline(y=1.0, color='gray', ls=':', linewidth=0.8)
    graphs[1,1].set_xlim(3955, 3980)
    pad = 0.1 * (flux_cahk.max() - flux_cahk.min())
    graphs[1,1].set_ylim(flux_cahk.min() - pad, flux_cahk.max() + pad)
else:
    graphs[1,1].text(0.5, 0.5, 'Not covered', ha='center', va='center', transform=graphs[1,1].transAxes)
graphs[1,1].set_title(r'Ca II K 3968 $\AA$')
graphs[1,1].set_xlabel('Wavelength (Å)')
graphs[1,1].set_ylabel('Normalized Flux')

filename2 = f"Features_{star}_{instrument}_localnorm.png"
plt.savefig(filename2)
plt.show()
