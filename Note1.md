# Note1

# 1、zip读取图片

通过路径，实例化一个压缩文件的对象

```python
from zipfile import ZipFile
zip_files = ZipFile(root)
```

通过对象的`namelist()`方法获取文件列表，

```python
for file_name in zip_files.namelist()
```

通过判断文件的结尾，来过滤文件

```python
if file_name.endswith('.jpg'):
```



打开图片：

```python
import numpy
import cv2
"""通过文件名，读取文件，读取出来为二进制文件"""
image = zip_files.read(file_name)
"""二进制文件转为array类型"""
image = numpy.asarray(bytearray(image), dtype='uint8')
"""由cv2的imdecode方法读取，即可"""
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
cv2.imshow("JK",image)
cv2.waitKey()
```

