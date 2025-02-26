import numpy as np
import torch
import csv
from typing import Tuple

def std2sfr(wv: np.ndarray, std: np.ndarray, wv_sfr=None, len_sfr=5000):
    if wv_sfr is None:
        wv_sfr = np.linspace(wv[0] - 2 * std[0], wv[-1] + 2 * std[1], len_sfr)
    else:
        len_sfr = len(wv_sfr)

    sfr = np.zeros((len(std), len_sfr))
    for i in range(len(std)):
        sfr[i] = 1 / (std[i] * (2 * np.pi)**0.5) * np.exp(-0.5 * ((wv_sfr - wv[i]) / std[i])**2)
    return wv_sfr, sfr

def fwhm2std(fwhm: np.ndarray):
    std = fwhm / 2.355
    return std

def read_fwhm_from_csv(file, bands=None) -> np.ndarray:
    wv = []
    fwhm = []
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i > 0:
                wv.append(float(row[0]))
                fwhm.append(float(row[1]))
    wv = np.array(wv)
    fwhm = np.array(fwhm)
    if bands is not None:
        bands = np.array(bands)
        wv = wv[bands]
        fwhm = fwhm[bands]
    # if wv.max() < 100:
    #         wv = 1000 * wv # convert in nm
    return wv, fwhm

def make_bins(wavs):
    """
    Given a series of wavelength points, find the edges and widths
    of corresponding wavelength bins.
    Source: https://github.com/ACCarnall/SpectRes/blob/master/spectres/spectral_resampling.py
    """
    edges = np.zeros(wavs.shape[0]+1)
    widths = np.zeros(wavs.shape[0])
    edges[0] = wavs[0] - (wavs[1] - wavs[0])/2
    widths[-1] = (wavs[-1] - wavs[-2])
    edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2])/2
    edges[1:-1] = (wavs[1:] + wavs[:-1])/2
    widths[:-1] = edges[1:-1] - edges[:-2]

    return edges, widths

def resample_from_wv(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=0,
             verbose=False):

    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.
    Source: https://github.com/ACCarnall/SpectRes/blob/master/spectres/spectral_resampling.py
    Paper: https://arxiv.org/abs/1705.05165

    Parameters
    ----------

    new_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.

    spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.

    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.

    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.

    fill : float (optional)
        Where new_wavs extends outside the wavelength range in spec_wavs
        this value will be used as a filler in new_fluxes and new_errs.

    verbose : bool (optional)
        Setting verbose to False will suppress the default warning about
        new_wavs extending outside spec_wavs and "fill" being used.

    Returns
    -------

    new_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.

    new_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """

    # Rename the input variables for clarity within the function.
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Make arrays of edge positions and widths for the old and new bins

    old_edges, old_widths = make_bins(old_wavs)
    new_edges, new_widths = make_bins(new_wavs)

    # Generate output arrays to be populated
    new_fluxes = np.zeros(old_fluxes[..., 0].shape + new_wavs.shape)

    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape "
                             "as spec_fluxes.")
        else:
            new_errs = np.copy(new_fluxes)

    start = 0
    stop = 0
    warned = False

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_wavs.shape[0]):

        # Add filler values if new_wavs extends outside of spec_wavs
        if (new_edges[j] < old_edges[0]) or (new_edges[j+1] > old_edges[-1]):
            new_fluxes[..., j] = fill

            if spec_errs is not None:
                new_errs[..., j] = fill

            if (j == 0 or j == new_wavs.shape[0]-1) and verbose and not warned:
                warned = True
                print("\nSpectres: new_wavs contains values outside the range "
                      "in spec_wavs, new_fluxes and new_errs will be filled "
                      "with the value set in the 'fill' keyword argument. \n")
            continue

        # Find first old bin which is partially covered by the new bin
        while old_edges[start+1] <= new_edges[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while old_edges[stop+1] < new_edges[j+1]:
            stop += 1

        # If new bin is fully inside an old bin start and stop are equal
        if stop == start:
            new_fluxes[..., j] = old_fluxes[..., start]
            if old_errs is not None:
                new_errs[..., j] = old_errs[..., start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = ((old_edges[start+1] - new_edges[j])
                            / (old_edges[start+1] - old_edges[start]))

            end_factor = ((new_edges[j+1] - old_edges[stop])
                          / (old_edges[stop+1] - old_edges[stop]))

            old_widths[start] *= start_factor
            old_widths[stop] *= end_factor

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = old_widths[start:stop+1]*old_fluxes[..., start:stop+1]
            new_fluxes[..., j] = np.sum(f_widths, axis=-1)
            new_fluxes[..., j] /= np.sum(old_widths[start:stop+1])

            if old_errs is not None:
                e_wid = old_widths[start:stop+1]*old_errs[..., start:stop+1]

                new_errs[..., j] = np.sqrt(np.sum(e_wid**2, axis=-1))
                new_errs[..., j] /= np.sum(old_widths[start:stop+1])

            # Put back the old bin widths to their initial values
            old_widths[start] /= start_factor
            old_widths[stop] /= end_factor

    # If errors were supplied return both new_fluxes and new_errs.
    if old_errs is not None:
        return new_fluxes, new_errs

    # Otherwise just return the new_fluxes spectrum array
    else:
        return new_fluxes


class SpectralResampling:
    def __init__(self, embed_dim=1024):
        """
        A generic class for spectral resampling
        """
        self.embed_dim = embed_dim

    def resample(self, data: torch.Tensor, sfr: torch.Tensor,
                 wv: torch.Tensor, std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            - (batch_size x in_bands) data tensor
            - (out_bands x in_bands) spectral function response
        :returns: (batch_size x out_bands) resampled data
                & (batch_size x out_bands) spectral positional embedding
        """
        # import pdb; pdb.set_trace()

        batch_size, B = data.shape
        out_bands, in_bands = sfr.shape
        data = data.reshape(batch_size, 1, B)
        if isinstance(sfr, np.ndarray):
            sfr = torch.from_numpy(sfr).float()
        sum_sfr = torch.sum(sfr, dim=1).view(1, 1, out_bands).repeat(batch_size, 1, 1)
        resampled_data = data @ sfr.transpose(-1, -2) / sum_sfr
        resampled_data = resampled_data.reshape(batch_size, out_bands)
        wv = wv.unsqueeze(0).repeat(batch_size, 1)
        std = std.unsqueeze(0).repeat(batch_size, 1)
        # pos_embed = spectral_postional_embedding(wv, std, self.embed_dim, data.device)
        pos_embed = (wv, std)
        return resampled_data, pos_embed

    def __call__(self):
        raise NotImplementedError