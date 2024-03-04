#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file spec_preprocess.py
# @author: wujiangu
# @date: 2024-02-28 19:18
# @description: spectrum preprocess

import numpy as np
from astropy.io import fits


def normalization_zscore(x):
    """Normalizate spec with zscore"""
    return (x - np.mean(x)) / np.std(x)


def read_lamost_fits(fits_file_path: str):
    """read lamost fits

    :param fits_file_path: str, fits file path
    :return: dict {"flux": flux, "wavelength": wavelength}
    """

    try:
        fits_file = fits.open(fits_file_path)
        # get flux
        flux = fits_file[1].data["flux"][0]
        # get wavelength
        wavelength = fits_file[1].data["wavelength"][0]

    except Exception as e:
        print(e)
        return None

    ret = {
        "flux": flux,
        "wavelength": wavelength,
    }

    return ret


def read_sdss_fits(fits_file_path: str):
    """read sdss fits

    :param fits_file_path: str, fits file path
    :return: dict {"flux": flux, "wavelength": wavelength}
    """
    try:
        fits_file = fits.open(fits_file_path)
        # get flux
        flux = np.array(fits_file[1].data["FLUX"])
        wavelength = np.array(fits_file[1].data["LOGLAM"])
        wavelength = 10**wavelength

    except Exception as e:
        print(e)
        return None

    ret = {
        "flux": flux,
        "wavelength": wavelength,
    }

    return ret


def read_other_fits(fits_file_path: str):
    """read other fits

    :param fits_file_path: str, fits file path
    :return: dict {"flux": flux, "wavelength": wavelength}
    """

    flux = []
    wavelength = []

    # TODO: read other survey fits

    ret = {
        "flux": flux,
        "wavelength": wavelength,
    }

    return ret


def main():
    # file path: SDSS
    fits_file_path = "./fits/spec-0324-51616-0159.fits"

    # read fits
    ret = read_sdss_fits(fits_file_path)

    # normalization
    flux = normalization_zscore(ret["flux"])

    # clip by wavelength
    wavelength_range = (4000, 9000)
    wavelength = ret["wavelength"][
        np.where(
            (ret["wavelength"] > wavelength_range[0])
            & (ret["wavelength"] < wavelength_range[1])
        )
    ]

    flux = flux[
        np.where(
            (ret["wavelength"] > wavelength_range[0])
            & (ret["wavelength"] < wavelength_range[1])
        )
    ]

    # TODO: change spectrum length
    # spectrum_length = len(flux)
    # change spectrum_length in cfg/cfg.py

    # save to csv (wavelength, flux)
    spectrum = np.column_stack((wavelength, flux))
    np.savetxt("spectrum.csv", spectrum, delimiter=",")


if __name__ == "__main__":
    main()
