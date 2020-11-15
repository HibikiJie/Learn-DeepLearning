import os
import cv2
import numpy
path = 'C:/Users/lieweiai/Desktop/yolov5-master/images'
def save_mp4(path, images, fps):
    h, w, c = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_movie = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for image in images:
        output_movie.write(image)


if __name__ == '__main__':
    images = []
    for image_name in range(1000):
        try:
            image = cv2.imread(f'{path}/{image_name}.jpg')
            print(type(image))
            if isinstance(image,numpy.ndarray):
                images.append(image)
        except:
            pass

    save_mp4(f'1.mp4', images, 25)