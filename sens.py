# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 17:26:58 2023
step：1)data 2)figure
@author: sudonghn_14513887214
"""

import matplotlib.pyplot as plt
import numpy as np

# In[1]:
# ## data 
# px: 坡向 
px = np.arange(0,2*np.pi,0.04)
# pd：坡度
pd = np.arange(0,0.5*np.pi,0.01)
PX,PD = np.meshgrid(px,pd) 

def sensitivity(h,r):
    # hxj:航向角 角度转弧度方便计算 '*'后为角度
    hxj = np.deg2rad(h)
    # rsj:入射角 角度转弧度方便计算 '*'后为角度
    rsj = np.deg2rad(r)
    # C1 C2 C3: 公式常数
    C1 = np.cos(hxj)*np.sin(rsj)
    C2 = np.sin(hxj)*np.sin(rsj)
    C3 = np.cos(rsj)
    
    
    # sens: 灵敏度
    sens = C1*np.sin(PX)*np.cos(PD) + C2*np.cos(PX)*np.cos(PD) + C3*np.sin(PD)
    # sens[0][0]=-1
    return sens

def merge_min(s1,s2):
    sens_min = []
    for i in range(0, len(s1)):
        row =[]
        for j in range(0, len(s1[i])):
            row.append(min(s1[i][j],s2[i][j]))
        sens_min.append(row)
    return sens_min
      
def merge_max(s1,s2):
    sens_max = []
    for i in range(0, len(s1)):
        row =[]
        for j in range(0, len(s1[i])):
            row.append(max(s1[i][j],s2[i][j]))
        sens_max.append(row)
    return sens_max

sens_12_25 = sensitivity(12,25)
sens_12_46 = sensitivity(12,46)
sens_168_25 = sensitivity(168,25)
sens_168_46 = sensitivity(168,46)
print(len(sens_168_25),len(sens_168_46),len(sens_12_25),len(sens_12_46))

# In[2]:
# ## figure 

# 极坐标系
ax = plt.subplot(111, projection='polar')
# 等值图
# heatmap = ax.contourf(px,pd,sens_12_25,levels=50,cmap='jet', vmin=-1, vmax=1)
# heatmap = ax.contourf(px,pd,sens_12_46,levels=50,cmap='jet', vmin=-1, vmax=1)
# heatmap = ax.contourf(px,pd,merge_min(sens_12_25,sens_12_46),levels=50,cmap='jet', vmin=-1, vmax=1)
# heatmap = ax.contourf(px,pd,sens_168_25,levels=50,cmap='jet', vmin=-1, vmax=1)
# heatmap = ax.contourf(px,pd,sens_168_46,levels=50,cmap='jet', vmin=-1, vmax=1)
# heatmap = ax.contourf(px,pd,merge_min(sens_168_25,sens_168_46),levels=50,cmap='jet', vmin=-1, vmax=1)
heatmap = ax.contourf(px,pd,merge_max(merge_min(sens_12_25,sens_12_46),merge_min(sens_168_25,sens_168_46)),levels=50,cmap='jet', vmin=-1, vmax=1)

# 坐标设置
ax.grid(True, linestyle="--", color="w", linewidth=0.8, alpha=0.8, which="major")
ax.set_theta_zero_location('N')
ax.set_theta_direction("clockwise")

# 刻度标注
ax.get_yaxis().set_ticks([])
# xticks = [0,78,90,180,270,348]
# xticks = [0,90,180,192,270,282]
xticks = [0,90,180,270]
ax.set_xticks(np.deg2rad(xticks))
# ax.set_xticklabels(["N","ALD","E","S","W","r"], fontsize=10)
# ax.set_xticklabels(["N","E","S","r","W","ALD"], fontsize=10)
ax.set_xticklabels(["N","E","S","W"], fontsize=10)
yticks = [0,30,60,90]
ax.set_yticks(np.deg2rad(yticks))
ax.set_yticklabels(["0°","30°","60°","90°"], fontsize=8)

# 颜色条
cbar = plt.colorbar(heatmap,location='right')
cbar.ax.set_xlabel('sensitivity')    

plt.show()
