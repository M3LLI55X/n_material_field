import numpy as np
import matplotlib.pyplot as plt
   # 加载 SDF 场
sdf_field = np.load('/dtu/blackhole/11/180913/test/ViewFusion/syncdreamer_3drec/test_imgnew2/chair_5/sdf_field.npy')

   # 打印基本信息
print("Shape:", sdf_field.shape)
print("Data type:", sdf_field.dtype)
print("Min value:", np.min(sdf_field))
print("Max value:", np.max(sdf_field))
print("Mean value:", np.mean(sdf_field))

# 显示直方图
plt.figure(figsize=(10, 6))
plt.hist(sdf_field.flatten(), bins=100)
plt.title('SDF Values Distribution')
plt.xlabel('SDF Value')
plt.ylabel('Frequency')
plt.show()

# 显示中心切片
center_slice = sdf_field[sdf_field.shape[0]//2, :, :]
plt.figure(figsize=(10, 10))
plt.imshow(center_slice, cmap='viridis')
plt.colorbar(label='SDF Value')
plt.title('Center Slice of SDF Field')
plt.show()

from mayavi import mlab

# 创建 3D 体积渲染
mlab.figure(size=(800, 800))
vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(sdf_field))
mlab.colorbar(title='SDF Value', orientation='vertical')
mlab.show()