import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True) 
# 作用是取消numpy默认的科学计数法，测试表明open3d点云读取函数没法读取科学计数法的表示
data1 = np.load('./semantic_3d_pointcloud/point.npy')
data2 = np.load('./semantic_3d_pointcloud/color01.npy')
data = np.hstack([data1, data2])
print(data.shape)

new_data = np.empty((0,6), float)

# 檢查每一列的z是否要為天花板
for i in range(data.shape[0]):
    if (data[i][1] < -0.01 and data[i][1] > -0.035):  # 用axis_pcd測試出來的值 大於0.02的刪除
        current_row = np.array([[data[i][0]*10000/255, data[i][1]*10000/255, data[i][2]*10000/255, data[i][3], data[i][4], data[i][5]]])
        new_data = np.append(new_data, values = current_row, axis = 0)
        new_data = np.array(new_data)
        print(new_data.shape)
plt.scatter(new_data[:, 2], new_data[:, 0], s = 5, color = new_data[:, 3:], alpha=1) # slice
plt.xlim(-6, 11) #設定x軸顯示範圍
plt.ylim(-4, 8) #設定y軸顯示範圍
plt.axis("off")

plt.savefig("map.png", bbox_inches='tight', pad_inches = 0)
plt.show()
# print(new_data.shape)
txt_data = np.savetxt('pointRGB.txt', new_data)

# 此处因为npy里面正好是 x y z r g b的数据排列形式，所以format='xyzrgb'
pcd = o3d.io.read_point_cloud('pointRGB.txt', format='xyzrgb')

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])  # 用此印到點雲上可知0, 0, 0和sim場景的0, 0, 0相同
# o3d.visualization.draw_geometries([pcd], width=1200, height=600, zoom=1.1, 
#                                                     front=[0, -0.1, 0],
#                                                     lookat=[0, -0.1, 0],
#                                                     up=[0.1, 0, 0])
