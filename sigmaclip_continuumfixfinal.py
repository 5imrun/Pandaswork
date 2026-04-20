# base template: 
# prints EW values

import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from astroquery.eso import Eso
from astropy.io import fits
from astropy import constants as const
import time



# 1. QUERY
eso = Eso()
# eso.login('YOUR_USERNAME') 

query = 'HD 209458' 
instrument = 'HARPS' 

results = eso.query_surveys(target=query, surveys=instrument)

if results is None or len(results) == 0:
    raise ValueError(f"No results found for {query}.")



# 2. FILTER FOR SPECTRA 
if "Product category" in results.colnames:
    mask = [("SPECTRUM" in str(x)) for x in results["Product category"]]
    spec = results[mask]
else:
    spec = results

use = spec if len(spec) > 0 else results
arc = str(use["ARCFILE"][0]).strip()



# 3. DOWNLOAD FITS FILES
timestamp = int(time.time())
filename = f"{query}_{instrument}_{timestamp}.fits"
path = os.path.join(os.path.expanduser("~"), f"{query}_{instrument}_{timestamp}.fits")

try:
    print(f"Attempting manual download to: {path}")
    url = f"https://dataportal.eso.org/dataPortal/file/{arc}"
    response = requests.get(url, stream=True, timeout=30)
    
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download successful.")
    else:
        print(f"Download failed. Status: {response.status_code}")
        print("Note: If status is 401/403, you must use eso.login().")
        path = None
except Exception as e:
    print(f"Error during download: {e}")
    path = None

# open the downloaded FITS file and extract wavelength + flux
with fits.open(path) as hdul:
    hdul.info()
    data = hdul[1].data
    xvals = data['WAVE'][0]
    yvals = data['FLUX'][0]



# 4. PLOTTING
# full spectrum overview
plt.figure(figsize=(10, 4))
plt.plot(xvals, yvals, color='tab:pink', lw=0.5)
plt.title(f"Full Spectrum: {query} ({instrument})")
plt.xlabel("Wavelength (Å)")
plt.ylabel("Flux")
plt.show()

# zoom in on key spectral features
fig, graphs = plt.subplots(2, 2, figsize=(12, 10))
plt.suptitle(f'Key Spectral Features: {query}', fontsize=16)

def plot_feature(ax, wave, flux, center, window, title, color):
    mask = (wave >= center - window) & (wave <= center + window)
    ax.plot(wave[mask], flux[mask], color=color)
    ax.axvline(x=center, color='black', ls='--')
    ax.set_title(title)
    y_seg = flux[mask]
    ymin, ymax = np.nanmin(y_seg), np.nanmax(y_seg)
    pad = 0.1 * (ymax - ymin) if ymax != ymin else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)

plot_feature(graphs[0,0], xvals, yvals, 6562.8, 50, 'H-alpha (6562.8 Å)', 'tab:pink')
plot_feature(graphs[0,1], xvals, yvals, 6707.8, 10, 'Lithium (6707.8 Å)', 'grey')
plot_feature(graphs[1,0], xvals, yvals, 3933.7, 30, 'Ca II H (3933.7 Å)', 'orange')
plot_feature(graphs[1,1], xvals, yvals, 3968.5, 30, 'Ca II K (3968.5 Å)', 'tab:red')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print(instrument)



# 5. LOCAL CONTINUUM NORMALIZATION (from Simran)

from specutils import Spectrum, SpectralRegion
from specutils.manipulation import extract_region
import astropy.units as u
from astropy.modeling import models, fitting

spectrum = Spectrum(
    spectral_axis=xvals * u.AA,
    flux=yvals * u.Unit('erg / (cm^2 s AA)')
)

def local_normalize(spectrum, line_center, window=20, shoulder=5):
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

    shoulder_mask = (wave < line_center - shoulder) | (wave > line_center + shoulder)

    # one-sided sigma clip within shoulder to exclude deep absorption lines
    shoulder_flux = flux[shoulder_mask]
    shoulder_wave = wave[shoulder_mask]
    median_shoulder = np.median(shoulder_flux)
    std_shoulder = np.std(shoulder_flux)
    clip_mask = shoulder_flux > (median_shoulder - 2.0 * std_shoulder)

    try:
        p_init = models.Chebyshev1D(1)
        fitter = fitting.LinearLSQFitter()
        p = fitter(p_init, shoulder_wave[clip_mask], shoulder_flux[clip_mask])
        continuum = p(wave)
        flux_norm = flux / continuum
    except Exception as e:
        print(f"Warning: continuum fit failed around {line_center} Å: {e}")
        return None, None

    return wave, flux_norm

# normalize around each feature
wave_ha,   flux_ha   = local_normalize(spectrum, line_center=6562.801, window=40, shoulder=8)
wave_li,   flux_li   = local_normalize(spectrum, line_center=6707.76,  window=15, shoulder=6)
wave_cahk, flux_cahk = local_normalize(spectrum, line_center=3950.0,   window=30, shoulder=8)

# 4-panel normalized plot (pre-RV correction)
figures, graphs = plt.subplots(2, 2, figsize=(10, 8))
plt.suptitle(f'\nLocally Normalized Key Spectral Features for {query}', fontsize=16)
plt.subplots_adjust(hspace=0.3)

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

plt.show()



# 6. RV CORRECTION
from astroquery.simbad import Simbad

Simbad.reset_votable_fields()
Simbad.add_votable_fields('rvz_radvel')
Simbad.add_votable_fields('mesfe_h')

simbad_result = Simbad.query_object(query)
print(simbad_result.colnames)

rv   = simbad_result['rvz_radvel'][0]
Teff_mean = np.mean(simbad_result['mesfe_h.teff'])
print(f'Mean Teff: {Teff_mean}')

c_kms = const.c.to('km/s').value
xvals_rest = xvals / (1 + rv / c_kms)

spectrum2 = Spectrum(
    spectral_axis=xvals_rest * u.AA,
    flux=yvals * u.Unit('erg / (cm^2 s AA)')
)

# re-normalize on RV-corrected spectrum
wave_ha2,   flux_ha2   = local_normalize(spectrum2, line_center=6562.801, window=40, shoulder=8)
wave_li2,   flux_li2   = local_normalize(spectrum2, line_center=6707.76,  window=15, shoulder=6)
wave_cahk2, flux_cahk2 = local_normalize(spectrum2, line_center=3950.0,   window=30, shoulder=8)

# 4-panel normalized plot (post-RV correction)
figures, graphs = plt.subplots(2, 2, figsize=(10, 8))
plt.suptitle(f'\nLocally Normalized + RV Corrected Key Spectral Features for {query}', fontsize=16)
plt.subplots_adjust(hspace=0.3)

if wave_ha is not None:
    graphs[0,0].plot(wave_ha2, flux_ha2, 'tab:pink')
    graphs[0,0].axvline(x=6562.801, color='black', ls='--')
    graphs[0,0].axhline(y=1.0, color='gray', ls=':', linewidth=0.8)
    pad = 0.1 * (flux_ha.max() - flux_ha.min())
    graphs[0,0].set_ylim(flux_ha.min() - pad, flux_ha.max() + pad)
else:
    graphs[0,0].text(0.5, 0.5, 'Not covered', ha='center', va='center', transform=graphs[0,0].transAxes)
graphs[0,0].set_title(r'H-alpha 6562.801 $\AA$')
graphs[0,0].set_xlabel('Wavelength (Å)')
graphs[0,0].set_ylabel('Normalized Flux')

if wave_li is not None:
    graphs[0,1].plot(wave_li2, flux_li2, 'tab:purple')
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

if wave_cahk is not None:
    graphs[1,0].plot(wave_cahk2, flux_cahk2, '#71C7A0')
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

if wave_cahk is not None:
    graphs[1,1].plot(wave_cahk2, flux_cahk2, '#4B9CD3')
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

plt.show()



# 7. EQUIVALENT WIDTH CALCULATIONS
# manual EW using trapezoidal integration (Jasleen, Team 3)
# EW = integral(1 - normalized_flux) d(lambda)
# absorption -> positive EW
# emission   -> negative EW

def compute_ew(wave, flux_norm, line_min, line_max):
    if wave is None or flux_norm is None:
        return np.nan
    wave = np.asarray(wave, dtype=float)
    flux_norm = np.asarray(flux_norm, dtype=float)
    mask = (wave >= line_min) & (wave <= line_max)
    if np.count_nonzero(mask) < 2:
        return np.nan
    ew = np.trapz(1.0 - flux_norm[mask], wave[mask])
    return ew

# H-alpha EW
if wave_ha2 is not None:
    ew_ha = compute_ew(wave_ha2, flux_ha2, 6555.0, 6570.0)
    print(f"H-alpha EW: {ew_ha:.3f} Å")

# Lithium EW — median offset correction applied before integration
if wave_li2 is not None:
    li_offset = np.median(flux_li2)
    flux_li2_corrected = flux_li2 / li_offset
    ew_li = compute_ew(wave_li2, flux_li2_corrected, 6707.4, 6709.0)
    print(f"Lithium EW: {ew_li:.3f} Å")

# Ca II H EW
if wave_cahk2 is not None:
    ew_cah = compute_ew(wave_cahk2, flux_cahk2, 3929.0, 3937.0)
    print(f"Ca II H EW: {ew_cah:.3f} Å")

# Ca II K EW
if wave_cahk2 is not None:
    ew_cak = compute_ew(wave_cahk2, flux_cahk2, 3964.0, 3972.0)
    print(f"Ca II K EW: {ew_cak:.3f} Å")



# 8. PULL Teff FROM GITHUB
import pandas as pd

csv_url = "https://raw.githubusercontent.com/awmann/Astro502_Sp26/main/ASTR502_Master_Parameters_List.csv"
df = pd.read_csv(csv_url)

df_query = df.loc[df['hostname'] == query]
if len(df_query) == 0:
    print(f"No match found for {query} in master table")
    t_eff = None
elif df_query['tic_teff'].dropna().empty:
    print(f"No TIC Teff available for {query}, using literature value")
    t_eff = 3700
else:
    t_eff = df_query['tic_teff'].dropna().iloc[0]
print('T_eff = ', t_eff)



# 9. LI CONTINUUM FIT DIAGNOSTIC
if wave_li2 is not None:
    shoulder = 6
    line_center = 6707.76
    shoulder_mask = (wave_li2 < line_center - shoulder) | (wave_li2 > line_center + shoulder)
    p_init = models.Chebyshev1D(1)
    fitter = fitting.LinearLSQFitter()
    p = fitter(p_init, wave_li2[shoulder_mask], flux_li2[shoulder_mask] * np.median(flux_li2))
    continuum_fit = p(wave_li2)
    plt.figure(figsize=(10, 4))
    plt.plot(wave_li2, flux_li2 * np.median(flux_li2), 'tab:purple', label='normalized flux (rescaled)')
    plt.plot(wave_li2, continuum_fit, 'r--', label='Chebyshev continuum fit')
    plt.axvline(x=6707.76, color='black', ls='--', label='Li 6707.76')
    plt.axvline(x=6707.4,  color='red',   ls=':',  label='Fe 6707.4')
    plt.axhline(y=np.median(flux_li2 * np.median(flux_li2)), color='gray', ls=':', linewidth=0.8)
    plt.legend()
    plt.title(f'Li continuum fit diagnostic - {query}')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')
    plt.show()
