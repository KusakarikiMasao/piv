import os
import sys
import glob
import warnings
import csv
import time
import cv2
import psutil
import numpy as np
from scipy.ndimage import maximum_filter

import pivprocess


def initial_check(file_extention):
    img0 = glob.glob("./img/img0/*." + file_extention)
    img1 = glob.glob("./img/img1/*." + file_extention)
    if len(img0) == len(img1):
        print("particle image pairs = %d pairs" % len(img0))

    else:
        warnings.warn("check number of images")
        sys.exit(0)

    cal_area = cv2.imread("./img/cal_area.bmp", 0)
    if type(cal_area) != np.ndarray:
        warnings.warn("define calculation area")
        sys.exit(0)

    return len(img0)


def set_grid(param_xgrid, param_ygrid):
    os.makedirs("./grid_info", exist_ok=True)
    x = np.genfromtxt("./coordinate/xgrid.csv", delimiter=",")
    y = np.genfromtxt("./coordinate/ygrid.csv", delimiter=",")
    x = np.round(x, 0)
    y = np.round(y, 0)
    """
    num_xgrid = int((param_xgrid[1] - param_xgrid[0]) / param_xgrid[2]) + 1
    num_ygrid = int((param_ygrid[1] - param_ygrid[0]) / param_ygrid[2]) + 1
    xgrid = np.arange(param_xgrid[0], param_xgrid[0]+param_xgrid[2]*num_xgrid, param_xgrid[2])
    ygrid = np.arange(param_ygrid[0], param_ygrid[0]+param_ygrid[2]*num_ygrid, param_ygrid[2])

    x, y = np.meshgrid(xgrid, ygrid)

    with open('./grid_info/xgrid.csv', 'w') as f1:
        writer = csv.writer(f1, lineterminator='\n')
        writer.writerows(x)
    with open('./grid_info/ygrid.csv', 'w') as f2:
        writer = csv.writer(f2, lineterminator='\n')
        writer.writerows(y)
    """

    cal_point = check_cal_point(x, y)

    return x, y, cal_point


# 真っ黒い部分は計算量を減らすため省いている
def check_cal_point(x, y):
    cal_point = np.zeros(x.shape)
    cal_area = cv2.imread("./img/cal_area.bmp", 0)
    cal_area_out = cv2.imread("./img/cal_area.bmp", 0)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_coord = int(x[i, j])
            y_coord = int(y[i, j])
            if 0 <= x_coord < cal_area.shape[1] and 0 <= y_coord < cal_area.shape[0]:
                if cal_area[y_coord, x_coord] == 255:
                    cal_point[i, j] = 1
                    cv2.circle(cal_area_out, (x_coord, y_coord), 5, (130), thickness=2)
    cv2.imwrite("./grid_info/cal_point.png", cal_area_out)  # 白い部分のみ計算をするよ
    with open("./grid_info/cal_point.csv", "w") as f1:
        writer = csv.writer(f1, lineterminator="\n")
        writer.writerows(cal_point)
    return cal_point


def load_all_images(num_of_images, file_extention):
    mem = psutil.virtual_memory()  # パソコンのメモリの表示
    print("initial memory usage : %f %%" % mem.percent)
    print("initial memory available : %f GB" % (mem.available / 1000000000))
    print("load all images")

    img0 = cv2.imread("./img/img0/img%04d." % 0 + file_extention, 0)
    for i in range(1, num_of_images):
        img_stack = cv2.imread("./img/img0/img%04d." % i + file_extention, 0)
        img0 = np.dstack([img0, img_stack])

    img1 = cv2.imread("./img/img1/img%04d." % 0 + file_extention, 0)
    for i in range(1, num_of_images):
        img_stack = cv2.imread("./img/img1/img%04d." % i + file_extention, 0)
        img1 = np.dstack([img1, img_stack])

    mem = psutil.virtual_memory()
    print("")
    print("now memory usage : %f %%" % mem.percent)
    print("now memory available : %f GB" % (mem.available / 1000000000))

    return img0, img1


def correlation(x, y, cal_point, img0, img1, param_xsearch, param_ysearch):
    os.makedirs("./vector_info", exist_ok=True)
    print("")
    print("piv process")

    mem = psutil.virtual_memory()
    print("now memory usage : %f %%" % mem.percent)
    print("now memory available : %f GB" % (mem.available / 1000000000))

    print(img0.shape)
    print(img1.shape)

    vector_u, vector_v = pivprocess.ensemble_piv(
        img0=img0,
        img1=img1,
        x_grid=x,
        y_grid=y,
        interrogation_window=(36, 36),
        search_window=(
            param_ysearch[0],
            param_ysearch[1],
            param_xsearch[0],
            param_ysearch[1],
        ),
        xoffset=np.zeros(x.shape),
        yoffset=np.zeros(y.shape),
        cal_point=cal_point,
    )

    # for i in range(x.shape[0]):  # single-pixelの計算
    #     print("%d / %d" % (i, x.shape[0]))
    #     for j in range(x.shape[1]):
    #         if cal_point[i, j]:
    #             x_coord = x[i, j]
    #             y_coord = y[i, j]
    #             try:
    #                 data0 = img0[int(y_coord), int(x_coord), :]
    #                 corr_map = np.zeros(
    #                     (
    #                         param_ysearch[1] - param_ysearch[0] + 1,
    #                         param_xsearch[1] - param_xsearch[0] + 1,
    #                     )
    #                 )
    #                 for i1 in range(param_xsearch[0], param_xsearch[1] + 1):
    #                     for j1 in range(
    #                         param_ysearch[0], param_ysearch[1] + 1
    #                     ):  # search windowでの繰り返し
    #                         if (
    #                             0 <= x_coord + i1 < img1.shape[1]
    #                             and 0 <= y_coord + j1 < img1.shape[0]
    #                         ):
    #                             data1 = img1[int(y_coord + j1), int(x_coord + i1), :]
    #                             corr_map[
    #                                 j1 - param_ysearch[0], i1 - param_xsearch[0]
    #                             ] = np.corrcoef(data0, data1)[
    #                                 0, 1
    #                             ]  # 相互相関係数の計算
    #                 corr_map[np.isnan(corr_map)] = 0
    #                 if np.sum(corr_map) == 0:
    #                     vector_u[i, j] = 0
    #                     vector_v[i, j] = 0
    #                 else:
    #                     deltaj, deltai = _detect_peak(corr_map)  # 一番関連度の高いところを出している
    #                     vector_u[i, j] = float(deltai) + param_xsearch[0]
    #                     vector_v[i, j] = float(deltaj) + param_ysearch[0]
    #             except IndexError:
    #                 print(x_coord, y_coord)

    with open("./vector_info/dj.csv", "w") as f1:
        writer = csv.writer(f1, lineterminator="\n")
        writer.writerows(vector_v)
    with open("./vector_info/di.csv", "w") as f2:
        writer = csv.writer(f2, lineterminator="\n")
        writer.writerows(vector_u)


def _detect_peak(correlation_map):  # ピークの検出(ネットで見てみる)
    def _sub_pixel_interpolation():  # サブピクセル補間
        try:
            VALUE0 = max(correlation_map[peak_j, peak_i], 0.01)
            VALUE1 = max(correlation_map[peak_j - 1, peak_i], 0.01)
            VALUE2 = max(correlation_map[peak_j + 1, peak_i], 0.01)
            VALUE3 = max(correlation_map[peak_j, peak_i - 1], 0.01)
            VALUE4 = max(correlation_map[peak_j, peak_i + 1], 0.01)
        except IndexError:
            return peak_j, peak_i

        delta_j = peak_j + 0.5 * (np.log(VALUE1) - np.log(VALUE2)) / (
            np.log(VALUE1) + np.log(VALUE2) - 2 * np.log(VALUE0)
        )
        delta_i = peak_i + 0.5 * (np.log(VALUE3) - np.log(VALUE4)) / (
            np.log(VALUE3) + np.log(VALUE4) - 2 * np.log(VALUE0)
        )
        if np.isnan(delta_j):
            delta_j = peak_j
        if np.isnan(delta_i):
            delta_i = peak_i
        return delta_j, delta_i

    correlation_map[np.isnan(correlation_map)] = 0

    local_max = maximum_filter(
        correlation_map, footprint=np.ones((5, 5)), mode="constant"
    )
    detected_peaks = np.ma.array(correlation_map, mask=~(correlation_map == local_max))
    temp = np.ma.array(
        detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * 0.3)
    )
    peaks_index = np.where((temp.mask == 0))

    if len((peaks_index[0])) == 1:
        peak_j, peak_i = np.unravel_index(
            correlation_map.argmax(), correlation_map.shape
        )
        return _sub_pixel_interpolation()
    elif len((peaks_index[0])) > 1:
        peak_j, peak_i = np.unravel_index(
            correlation_map.argmax(), correlation_map.shape
        )
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


def single_pixel_piv():
    file_extention = "bmp"
    param_xgrid = (15, 1005, 1)  # 昔使ってたから影響なし
    param_ygrid = (15, 1005, 1)  # 昔使ってたから影響なし
    param_xsearch = (-12, 3)  # motlab_PIVと同じ使い方
    param_ysearch = (-3, 3)  # motlab_PIVと同じ使い方

    num_of_images = initial_check(file_extention)
    x, y, cal_point = set_grid(param_xgrid, param_ygrid)
    img0, img1 = load_all_images(num_of_images, file_extention)

    correlation(x, y, cal_point, img0, img1, param_xsearch, param_ysearch)


if __name__ == "__main__":
    time1 = time.time()
    single_pixel_piv()
    time2 = time.time()
    print("calculation time is %f s" % (time2 - time1))
