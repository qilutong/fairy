读取图像


```python
import fairy

# 每次使用时读取，节省内存
images = fairy.data.load_images(
    "../data/datasets/images", mode="RGB", batch_size=2)

# # 将数据全部加载到内存中
# images = fairy.data.load_images_all(
#     "../data/datasets/images", mode="RGB", batch_size=2)

for img in images:
    print(img.shape)
```