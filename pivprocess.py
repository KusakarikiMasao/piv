#!/usr/bin/env python3
import numpy as np
import sys
from scipy.ndimage import maximum_filter


def _correlation_list(
        img0,
        img1,
        x_grid,
        y_grid,
        interrogation_window,
        search_window,
        xoffset,
        yoffset,
        cal_area):

    correlation_list = np.zeros((
        int(search_window[1] - search_window[0] + 1), int(search_window[3] - search_window[2] + 1), int(x_grid.shape[0] * x_grid.shape[1])))

    for j in range(x_grid.shape[0]):
        for i in range(x_grid.shape[1]):
            if cal_area[j, i] == 1:
                index_j = int(y_grid[j, i])
                index_i = int(x_grid[j, i])
                window_img0 = img0[int(index_j - interrogation_window[0] // 2):int(index_j + interrogation_window[0] // 2),
                                   int(index_i - interrogation_window[1] // 2):int(index_i + interrogation_window[1] // 2)]

                correlation_list[:, :, j * x_grid.shape[1] + i] = _correlation_map(
                    window_img0, img1, interrogation_window, search_window, index_j + int(yoffset[j, i]), index_i + int(xoffset[j, i]))
    return correlation_list


def _correlation_map(window_img0, img1, interrogation_window, search_window, index_j, index_i):
    correlation_map = np.zeros((search_window[1] - search_window[0] + 1, search_window[3] - search_window[2] + 1))

    for dj in range(search_window[0], search_window[1] + 1):
        for di in range(search_window[2], search_window[3] + 1):
            index_j2 = index_j + dj
            index_i2 = index_i + di
            if (0 <= index_j2 - interrogation_window[0] // 2 and img1.shape[0] > index_j2 + interrogation_window[0] // 2 and
               0 <= index_i2 - interrogation_window[1] // 2 and img1.shape[1] > index_i2 + interrogation_window[1] // 2):
                window_img1 = img1[int(index_j2 - interrogation_window[0] // 2):int(index_j2 + interrogation_window[0] // 2),
                                   int(index_i2 - interrogation_window[1] // 2):int(index_i2 + interrogation_window[1] // 2)]
                if window_img0.shape == window_img1.shape:
                    correlation_map[abs(search_window[0]) + dj, abs(search_window[2]) + di
                                    ] = np.corrcoef(window_img0.flatten(), window_img1.flatten())[0, 1]

    return correlation_map


def _detect_peak(correlation_map):
    def _sub_pixel_interpolation():
        try:
            VALUE0 = max(correlation_map[peak_j, peak_i], 0.01)
            VALUE1 = max(correlation_map[peak_j - 1, peak_i], 0.01)
            VALUE2 = max(correlation_map[peak_j + 1, peak_i], 0.01)
            VALUE3 = max(correlation_map[peak_j, peak_i - 1], 0.01)
            VALUE4 = max(correlation_map[peak_j, peak_i + 1], 0.01)
        except IndexError:
            return peak_j, peak_i

        delta_j = peak_j + 0.5 * (np.log(VALUE1) - np.log(VALUE2)) / (np.log(VALUE1) + np.log(VALUE2) - 2*np.log(VALUE0))
        delta_i = peak_i + 0.5 * (np.log(VALUE3) - np.log(VALUE4)) / (np.log(VALUE3) + np.log(VALUE4) - 2*np.log(VALUE0))
        if np.isnan(delta_j):
            delta_j = peak_j
        if np.isnan(delta_i):
            delta_i = peak_i
        return delta_j, delta_i

    correlation_map[np.isnan(correlation_map)] = 0

    local_max = maximum_filter(correlation_map, footprint=np.ones((5, 5)), mode='constant')
    detected_peaks = np.ma.array(correlation_map, mask=~(correlation_map == local_max))
    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * 0.3))
    peaks_index = np.where((temp.mask == 0))

    if len((peaks_index[0])) == 1:
        peak_j, peak_i = np.unravel_index(correlation_map.argmax(), correlation_map.shape)
        return _sub_pixel_interpolation()
    elif len((peaks_index[0])) > 1:
        peak_j, peak_i = np.unravel_index(correlation_map.argmax(), correlation_map.shape)
        value_lis = []
        for num in range(len((peaks_index[0]))):
            value_lis.append(correlation_map[peaks_index[0][num], peaks_index[1][num]])
        rs1 = sorted(value_lis)[-1]
        rs2 = sorted(value_lis)[-2]
        if rs1 / rs2 >= 1.75:
            return _sub_pixel_interpolation()
        else:
            return np.nan, np.nan
    else:
        return np.nan, np.nan


def normal_piv(
        img0,
        img1,
        x_grid,
        y_grid,
        interrogation_window,
        search_window,
        xoffset,
        yoffset):

    vector_u = np.zeros(x_grid.shape)
    vector_v = np.zeros(x_grid.shape)

    correlation_list = _correlation_list(img0, img1, x_grid, y_grid, interrogation_window, search_window, xoffset, yoffset)
    for j in range(x_grid.shape[0]):
        for i in range(x_grid.shape[1]):
            peak_j, peak_i = _detect_peak(correlation_list[:, :, j * x_grid.shape[1] + i])
            vector_v[j, i] = peak_j + search_window[0] + int(yoffset[j, i])
            vector_u[j, i] = peak_i + search_window[2] + int(xoffset[j, i])

    return vector_u, vector_v


def ensemble_piv(
        img0,
        img1,
        x_grid,
        y_grid,
        interrogation_window,
        search_window,
        xoffset,
        yoffset,
        cal_area):
    """Particle image velocimetry with ensemble correlation

    Velocity field estimation from particle images (Particle image velocimetry, PIV)
    Calculate direct correlation coefficient and take average in correlation field

    Parameters
    ----------
    img0 : 3d_ndarray
        particle images stacked in component3 direction. particle images must be gray scale.
    img1 : 3d_ndarray
        particle images photographed after dt[s] stacked in component3 direction. particle images must be gray scale.
    x_grid :
    y_grid :
    interrogation_window :
    search_window :
    xoffset :
    yoffset :
    cal_area :

    Returns
    -------
    type
        2d_ndarray
    describe : type
        brbrbrbr
    out : (2d_ndarray, 2d_ndarray)
        velocity field (u, v)

    Examples
    --------
    >>> import cv2
    >>> import numpy as np
    >>> from pivpy import pivprocess

    >>> img_all = cv2.imread('./img%04d.bmp' % 0, 0)
    >>> for i in range(1, 10):
    >>>     img = cv2.imread('./img%04d.bmp' % i, 0)
    >>>     img_all = np.dstack([img_all, img])
    >>> velocity_u, velocity_v = pivprocess.ensemble_piv(img_all[:, :, :-2], img_all[:, :, 1:-1])

    """
    if img0.shape != img1.shape:
        sys.exit(1)

    vector_u = np.zeros(x_grid.shape)
    vector_v = np.zeros(x_grid.shape)

    correlation_list = np.zeros((
        int(search_window[1] - search_window[0] + 1), int(search_window[3] - search_window[2] + 1), int(x_grid.shape[0] * x_grid.shape[1])))
    for pair_num in range(img0.shape[2]):
        correlation_list = correlation_list + _correlation_list(img0[:, :, pair_num], img1[:, :, pair_num],
                                                                x_grid, y_grid, interrogation_window, search_window, xoffset, yoffset, cal_area)
    for j in range(x_grid.shape[0]):
        for i in range(x_grid.shape[1]):
            if cal_area[j, i] == 1:
                peak_j, peak_i = _detect_peak(correlation_list[:, :, j * x_grid.shape[1] + i])
                vector_v[j, i] = peak_j + search_window[0] + int(yoffset[j, i])
                vector_u[j, i] = peak_i + search_window[2] + int(xoffset[j, i])

    return vector_u, vector_v
