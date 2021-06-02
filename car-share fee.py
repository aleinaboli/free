import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d

'''
author : baiy
date : 2021-4-17
version ：0.1
'''

'''
！！！最终费用请根据实际情况添加调度费！！！
'''

car_type = ['polo']
car = 'polo'
km = 20
tm = 120

if car == 'polo':
    km_perm = 0.53
    baoxian_perh = 6
    baoxian_max = 36
    tm_perm = 0.39
    tm_6h = 69
    tm_3h = 39

km_list = np.linspace(0, 300, 301)  # 0~9间产生500个元素的均匀列表
tm_list = np.linspace(0, 720, 721)  # 0~3.5间产生500个元素的均匀列表

KM, TM = np.meshgrid(km_list, tm_list)  # 产生二维矩阵

#flat_KM, flat_TM = KM.ravel(), TM.ravel()  # 二维矩阵扁平化

'''gofun'''
#if 0<km<301:
gofun_km = KM * km_perm
##tm
tm_grid = np.ones_like(TM)
#tm_6h_grid = np.ones_like(TM)
baoxian_num_grid = np.ones_like(TM)
baoxian_num_grid[100:121] = 2
baoxian_num_grid[121:181] = 3
baoxian_num_grid[181:241] = 4
baoxian_num_grid[241:301] = 5
baoxian_num_grid[301:721] = 6

#if 0 < tm < 60:
gofun_tm_0_60 = TM[0:60]* tm_perm + baoxian_perh
#elif 60 <= tm < 100:
gofun_tm_60_100 = TM[60:100] * tm_perm + baoxian_perh * 2
#elif 100 <= tm < 180:
#baoxian_num = 2 if TM[100:180] - 120 <= 0 else 3
#TM[100:180] = tm_3h
#tm_3h_grid_1 = np.ones(80,300)*tm_3h
#baoxian_num_100_120 = \
#baoxian_num_grid[100:120] = 2
#baoxian_num_120_180 = \
#baoxian_num_grid[120:180] = 3
baoxian_num_100_180 = np.vstack((baoxian_num_grid[100:120],baoxian_num_grid[120:180]))
print(type(tm_3h),type(baoxian_perh))
gofun_tm_100_180 = tm_grid[100:180]*tm_3h + baoxian_perh * baoxian_num_100_180
#elif 180<= tm <257:
tm_more_180 = TM[180:257] - 180
# baoxian_num_180_240 = baoxian_num_grid[180:240] = 4
# baoxian_num_240_257 = baoxian_num_grid[240:257] = 5
baoxian_num_180_257 = np.vstack((baoxian_num_grid[180:240],baoxian_num_grid[240:257] ))
gofun_tm_180_257 = tm_grid[180:257]*tm_3h +tm_more_180*tm_perm+ baoxian_perh * baoxian_num_180_257
#elif 257<=tm <360:
# baoxian_num_257_300 = baoxian_num_grid[257:300] = 5
# baoxian_num_300_360 = baoxian_num_grid[240:257] = 6
baoxian_num_257_360 = np.vstack((baoxian_num_grid[257:300],baoxian_num_grid[300:360]))
gofun_tm_257_360 = tm_grid[257:360]*tm_6h + baoxian_perh * baoxian_num_257_360
#elif 360<=tm<1201:
tm_more_360 = TM[360:721] - 360
gofun_tm_360_1201 = tm_grid[360:721]*tm_6h+tm_more_360*tm_perm+baoxian_max

gofun_tm = np.vstack((gofun_tm_0_60,gofun_tm_60_100,gofun_tm_100_180,gofun_tm_180_257,gofun_tm_257_360,gofun_tm_360_1201))

# loss_metrix = train_y.reshape(-1, 1)  # 生成误差矩阵（-1,1）表示自动计算维度
# outer = np.outer(train_x, flat_w1)  # 求外积（train_x和flat_w1元素两两相乘的新矩阵）
# 计算损失：((w0 + w1*x - y)**2)/2
# flat_loss = (((flat_w0 + outer - loss_metrix) ** 2).sum(axis=0)) / 2
# grid_loss = flat_loss.reshape(grid_w0.shape)
gofun_z = gofun_km + gofun_tm

point_km = KM[0][km]
point_tm = TM[tm][0]
point_z = gofun_z[tm][km]

#z = 0.53 * KM + 0.39 * TM

mp.figure('price_comare')
ax = mp.gca(projection='3d')
mp.title('price', fontsize=14)
ax.set_xlabel('km', fontsize=14)
ax.set_ylabel('time', fontsize=14)
ax.set_zlabel('money', fontsize=14)
ax.plot_surface(KM, TM, gofun_z, rstride=5, cstride=5, linewidth=0, cmap='summer')# color='green',
# ax.contour(KM, TM, z, zdir = 'z', offset = 0)
ax.plot(point_km, point_tm, point_z, '.', c='orangered',  zorder=3,markersize = 3)#label='BGD',
ax.text(point_km,point_tm,point_z,'{}km,{}min,{}RMB'.format(point_km,point_tm,point_z))#fontsize,style,color
# mp.legend(loc='lower left')


mp.show()
