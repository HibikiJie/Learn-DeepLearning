import mxnet as mx
import mxnet.ndarray as nd
from skimage import io
import numpy
from PIL import Image
import os
path_prefix = 'D:/BaiduNetdiskDownload/faces_emore/faces_emore/train'
rec_path = path_prefix + ".rec"
idx_path = path_prefix + ".idx"
print(rec_path)
save_path = 'D:/data/object2/dataset'
train_iter = mx.image.ImageIter(
    batch_size=1,
    data_shape=(3, 112, 112),
    path_imgrec=rec_path,
    path_imgidx=idx_path,
    shuffle=False,)

train_iter.reset()
i = 0
for batch in train_iter:
    x = batch.data[0]
    y = batch.label[0]
    target = int(y[0].asnumpy()[0])
    if target == 10000:
        break
    img = nd.transpose(x, (0, 2, 3, 1)).asnumpy().astype(numpy.uint8).squeeze(0)
    image = Image.fromarray(img)
    if not os.path.exists(f'{save_path}/{target}'):
        i = 0
        os.makedirs(f'{save_path}/{target}')
    image.save(f'{save_path}/{target}/{i}.jpg',quality=95)
    print(target)

    i+=1
img = nd.transpose(x, (0, 2, 3, 1))
io.imshow(img[0].asnumpy().astype(numpy.uint8))
io.show()