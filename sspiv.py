import os
import csv
import copy
import glob
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import median_filter
from pivpy import postprocess


class Stereoscopic:
    def __init__(self):
        self.plane_num = len(glob.glob('./img/case*'))

    def load_calib(self):
        self.calib = np.zeros((10, 4, 3, 2))
        for cam in (0, 1):
            for num, plane in enumerate((1, 3, 5)):
                calib = pd.read_csv('./calib/coe%d-%d.d' % (cam, plane), delim_whitespace=True, names=('0', '1', '2', '3'))
                data = calib.to_numpy()
                self.calib[:, :, num, cam] = data[7:, :]

    def make_grid(self, param_xgrid, param_ygrid):
        os.makedirs('./grid_info', exist_ok=True)

        num_xgrid = int((param_xgrid[1] - param_xgrid[0]) / param_xgrid[2]) + 1
        num_ygrid = int((param_ygrid[1] - param_ygrid[0]) / param_ygrid[2]) + 1
        xgrid = np.arange(param_xgrid[0], param_xgrid[0]+param_xgrid[2]*num_xgrid, param_xgrid[2])
        ygrid = np.arange(param_ygrid[0], param_ygrid[0]+param_ygrid[2]*num_ygrid, param_ygrid[2])

        x_cam0, y_cam0 = np.meshgrid(xgrid, ygrid)

        def change_coordinate():
            lis = np.zeros(10)
            x_world = np.zeros(x_cam0.shape)
            y_world = np.zeros(y_cam0.shape)

            for j in range(x_cam0.shape[0]):
                for i in range(x_cam0.shape[1]):
                    lis[0] = 1.0
                    lis[1] = x_cam0[j, i]
                    lis[2] = x_cam0[j, i]**2
                    lis[3] = x_cam0[j, i]**3
                    lis[4] = y_cam0[j, i]
                    lis[5] = y_cam0[j, i]**2
                    lis[6] = y_cam0[j, i]**3
                    lis[7] = x_cam0[j, i]*y_cam0[j, i]
                    lis[8] = x_cam0[j, i]**2*y_cam0[j, i]
                    lis[9] = x_cam0[j, i]*y_cam0[j, i]**2

                    for num in range(10):
                        x_world[j, i] += self.calib[num, 0, 1, 0]*lis[num]
                        y_world[j, i] += self.calib[num, 1, 1, 0]*lis[num]

            with open('./grid_info/xgrid_world.csv', 'w') as f2:
                writer = csv.writer(f2, lineterminator='\n')
                writer.writerows(x_world)
            with open('./grid_info/ygrid_world.csv', 'w') as f2:
                writer = csv.writer(f2, lineterminator='\n')
                writer.writerows(y_world)

            x_cam1 = np.zeros(x_cam0.shape)
            y_cam1 = np.zeros(y_cam0.shape)

            for j in range(x_cam0.shape[0]):
                for i in range(x_cam0.shape[1]):
                    lis[0] = 1.0
                    lis[1] = x_world[j, i]
                    lis[2] = x_world[j, i]**2
                    lis[3] = x_world[j, i]**3
                    lis[4] = y_world[j, i]
                    lis[5] = y_world[j, i]**2
                    lis[6] = y_world[j, i]**3
                    lis[7] = x_world[j, i]*y_world[j, i]
                    lis[8] = x_world[j, i]**2*y_world[j, i]
                    lis[9] = x_world[j, i]*y_world[j, i]**2

                    for num in range(10):
                        x_cam1[j, i] += self.calib[num, 2, 1, 1]*lis[num]
                        y_cam1[j, i] += self.calib[num, 3, 1, 1]*lis[num]
            return x_cam1, y_cam1

        x_cam1, y_cam1 = change_coordinate()

        with open('./grid_info/xgrid_cam0.csv', 'w') as f1:
            writer = csv.writer(f1, lineterminator='\n')
            writer.writerows(x_cam0)
        with open('./grid_info/ygrid_cam0.csv', 'w') as f2:
            writer = csv.writer(f2, lineterminator='\n')
            writer.writerows(y_cam0)
        with open('./grid_info/xgrid_cam1.csv', 'w') as f1:
            writer = csv.writer(f1, lineterminator='\n')
            writer.writerows(x_cam1)
        with open('./grid_info/ygrid_cam1.csv', 'w') as f2:
            writer = csv.writer(f2, lineterminator='\n')
            writer.writerows(y_cam1)

        def check_coordinate():
            for case in range(self.plane_num):
                cal_area = cv2.imread('./cross_section/plane%02d_0.bmp' % case, 0)
                img_cam0 = cv2.imread('./cross_section/plane%02d_0.bmp' % case, 0)
                img_cam1 = cv2.imread('./cross_section/plane%02d_1.bmp' % case, 0)
                for j in range(x_cam0.shape[0]):
                    for i in range(x_cam0.shape[1]):
                        if cal_area[int(y_cam0[j, i]), int(x_cam0[j, i])] == 255:
                            cv2.circle(img_cam0, (int(x_cam0[j, i]), int(y_cam0[j, i])), 5, (125), thickness=2)
                            if 0 < int(x_cam1[j, i]) < cal_area.shape[1] and 0 < int(y_cam1[j, i]) < cal_area.shape[0]:
                                cv2.circle(img_cam1, (int(x_cam1[j, i]), int(y_cam1[j, i])), 5, (125), thickness=2)
                img_res = cv2.hconcat([img_cam0, img_cam1])
                cv2.imwrite('./grid_info/grid%02d.png' % case, img_res)


        check_coordinate()

    def stereoscopic_piv(self, dt):
        for case in range(self.plane_num):
            xgrid, ygrid = self.load_grid_data()
            xgrid, ygrid = self.set_vector(case, copy.deepcopy(xgrid), copy.deepcopy(ygrid))
            x_pln0, y_pln0, x_pln2, y_pln2 = self.trans(xgrid, ygrid)
            xd, yd, zd = self.intersection(x_pln0, y_pln0, x_pln2, y_pln2)
            self.output_3d_data(dt, xd, yd, zd, case)

        return 0

    def load_grid_data(self):
        df1 = np.loadtxt('./grid_info/xgrid_cam0.csv', delimiter=',')
        df2 = np.loadtxt('./grid_info/xgrid_cam1.csv', delimiter=',')
        df3 = np.loadtxt('./grid_info/ygrid_cam0.csv', delimiter=',')
        df4 = np.loadtxt('./grid_info/ygrid_cam1.csv', delimiter=',')

        xgrid = np.zeros((df1.shape[0], df1.shape[1], 4))
        ygrid = np.zeros((df1.shape[0], df1.shape[1], 4))
        xgrid[:, :, 0] = df1
        xgrid[:, :, 1] = df2
        ygrid[:, :, 0] = df3
        ygrid[:, :, 1] = df4

        return xgrid, ygrid

    def set_vector(self, case, xgrid, ygrid):
        df1 = np.loadtxt('./2dpiv/case%02d/cam%d/di.csv' % (case, 0), delimiter=',')
        df2 = np.loadtxt('./2dpiv/case%02d/cam%d/dj.csv' % (case, 0), delimiter=',')
        df1, df2, _ = postprocess.error_vector_interp_2d(df1, df2)
        df3 = np.loadtxt('./2dpiv/case%02d/cam%d/di.csv' % (case, 1), delimiter=',')
        df4 = np.loadtxt('./2dpiv/case%02d/cam%d/dj.csv' % (case, 1), delimiter=',')
        df3, df4, _ = postprocess.error_vector_interp_2d(df3, df4)

        xgrid[:, :, 2] = df1[:, :] + xgrid[:, :, 0]
        ygrid[:, :, 2] = df2[:, :] + ygrid[:, :, 0]
        xgrid[:, :, 3] = df3[:, :] + xgrid[:, :, 1]
        ygrid[:, :, 3] = df4[:, :] + ygrid[:, :, 1]
        return xgrid, ygrid

    def trans(self, xgrid, ygrid):
        lis = np.zeros(10)
        x_pln0 = np.zeros(xgrid.shape)
        y_pln0 = np.zeros(xgrid.shape)
        x_pln2 = np.zeros(xgrid.shape)
        y_pln2 = np.zeros(xgrid.shape)

        for cam in (0, 1, 2, 3):
            cam2 = cam
            if cam2 >= 2:
                cam2 -= 2
            for j in range(xgrid.shape[0]):
                for i in range(xgrid.shape[1]):
                    lis[0] = 1.0
                    lis[1] = xgrid[j, i, cam]
                    lis[2] = xgrid[j, i, cam]**2
                    lis[3] = xgrid[j, i, cam]**3
                    lis[4] = ygrid[j, i, cam]
                    lis[5] = ygrid[j, i, cam]**2
                    lis[6] = ygrid[j, i, cam]**3
                    lis[7] = xgrid[j, i, cam]*ygrid[j, i, cam]
                    lis[8] = xgrid[j, i, cam]*xgrid[j, i, cam]*ygrid[j, i, cam]
                    lis[9] = xgrid[j, i, cam]*ygrid[j, i, cam]*ygrid[j, i, cam]

                    for num in range(10):
                        x_pln0[j, i, cam] += self.calib[num, 0, 0, cam2]*lis[num]
                        y_pln0[j, i, cam] += self.calib[num, 1, 0, cam2]*lis[num]
                        x_pln2[j, i, cam] += self.calib[num, 0, 2, cam2]*lis[num]
                        y_pln2[j, i, cam] += self.calib[num, 1, 2, cam2]*lis[num]

        return x_pln0, y_pln0, x_pln2, y_pln2

    def intersection(self, x_pln0, y_pln0, x_pln2, y_pln2):
        """detect intersection"""
        ss = np.zeros((x_pln0.shape[0], x_pln0.shape[1]))
        tt = np.zeros((x_pln0.shape[0], x_pln0.shape[1]))
        xd = np.zeros(x_pln0.shape)
        yd = np.zeros(x_pln0.shape)
        zd = np.zeros(x_pln0.shape)
        ZD3 = -0.0002
        ZD1 = 0.0002
        ZGAP = ZD3 - ZD1

        for plane in (0, 2):
            for j in range(x_pln0.shape[0]):
                for i in range(x_pln0.shape[1]):
                    da1 = x_pln0[j, i, plane] - x_pln0[j, i, plane+1]
                    db1 = x_pln2[j, i, plane] - x_pln0[j, i, plane]
                    dc1 = x_pln2[j, i, plane+1] - x_pln0[j, i, plane+1]
                    da2 = y_pln0[j, i, plane] - y_pln0[j, i, plane+1]
                    db2 = y_pln2[j, i, plane] - y_pln0[j, i, plane]
                    dc2 = y_pln2[j, i, plane+1] - y_pln0[j, i, plane+1]
                    bb = db1**2 + db2**2 + ZGAP**2
                    cc = dc1**2 + dc2**2 + ZGAP**2
                    dd = da1 * db1 + da2*db2
                    ee = dc1 * da1 + dc2*da2
                    ff = db1*dc1 + db2*dc2 + ZGAP**2

                    det = bb*cc - ff**2
                    if abs(det) >= 1.0e-30:
                        ss[j, i] = (-cc*dd + ee*ff)/det
                        tt[j, i] = (-ff*dd + bb*ee)/det
                    else:
                        ss[j, i] = 0.0
                        tt[j, i] = 0.0

            for j in range(x_pln0.shape[0]):
                for i in range(x_pln0.shape[1]):
                    dumx1 = ss[j, i]*x_pln2[j, i, plane] + (1.0 - ss[j, i])*x_pln0[j, i, plane]
                    dumy1 = ss[j, i]*y_pln2[j, i, plane] + (1.0 - ss[j, i])*y_pln0[j, i, plane]
                    dumz1 = ss[j, i]*ZD3 + (1.0 - ss[j, i])*ZD1
                    dumx2 = tt[j, i]*x_pln2[j, i, plane+1] + (1.0 - tt[j, i])*x_pln0[j, i, plane+1]
                    dumy2 = tt[j, i]*y_pln2[j, i, plane+1] + (1.0 - tt[j, i])*y_pln0[j, i, plane+1]
                    dumz2 = tt[j, i]*ZD3 + (1.0 - tt[j, i])*ZD1

                    xd[j, i, plane] = 0.5*(dumx1 + dumx2)
                    yd[j, i, plane] = 0.5*(dumy1 + dumy2)
                    zd[j, i, plane] = 0.5*(dumz1 + dumz2)

        return xd, yd, zd

    def output_3d_data(self, dt, xd, yd, zd, case):
        os.makedirs('./3dpiv', exist_ok=True)
        uu = np.zeros((xd.shape[0], xd.shape[1]))
        vv = np.zeros((xd.shape[0], xd.shape[1]))
        ww = np.zeros((xd.shape[0], xd.shape[1]))

        for j in range(xd.shape[0]):
            for i in range(xd.shape[1]):
                uu[j, i] = (xd[j, i, 2] - xd[j, i, 0]) / dt*1000
                vv[j, i] = (yd[j, i, 2] - yd[j, i, 0]) / dt*1000
                ww[j, i] = (zd[j, i, 2] - zd[j, i, 0]) / dt*1000

        output_2dcsv_data('./3dpiv/raw_u%02d.csv' % (case), uu)
        output_2dcsv_data('./3dpiv/raw_v%02d.csv' % (case), vv)
        output_2dcsv_data('./3dpiv/raw_w%02d.csv' % (case), ww)

    def post_process(self):
        for case in range(self.plane_num):
            u0 = np.genfromtxt('./3dpiv/raw_u%02d.csv' % case, delimiter=',')
            v0 = np.genfromtxt('./3dpiv/raw_v%02d.csv' % case, delimiter=',')
            w0 = np.genfromtxt('./3dpiv/raw_w%02d.csv' % case, delimiter=',')
            u1, v1, w1, _ = postprocess.error_vector_interp_3d(u0, v0, w0)
            u1 = median_filter(u1, size=3)
            v1 = median_filter(v1, size=3)
            w1 = median_filter(w1, size=3)
            output_2dcsv_data('./3dpiv/u%02d.csv' % (case), u1)
            output_2dcsv_data('./3dpiv/v%02d.csv' % (case), v1)
            output_2dcsv_data('./3dpiv/w%02d.csv' % (case), w1)
        return 0


def output_2dcsv_data(fname, value):
    with open(fname, 'w') as f1:
        writer = csv.writer(f1, lineterminator='\n')
        writer.writerows(value)


def main():
    do_grid = False
    do_piv = False
    do_sspiv = True
    do_post_process = True

    stereoscopic = Stereoscopic()
    stereoscopic.load_calib()

    if do_grid:
        param_xgrid = (100, 1100, 1)
        param_ygrid = (15, 1600, 1)
        stereoscopic.make_grid(param_xgrid, param_ygrid)
    if do_piv:
        pass
    if do_sspiv:
        dt = 100E-6
        stereoscopic.stereoscopic_piv(dt)
    if do_post_process:
        stereoscopic.post_process()


if __name__ == '__main__':
    main()
