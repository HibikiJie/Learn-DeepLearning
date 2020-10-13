import os
import cv2


def save_mp4(path, images, fps):
    h, w, c = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_movie = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for image in images:
        output_movie.write(image)


if __name__ == '__main__':
    images = []
    for i in range(236):
        image = cv2.imread(f'D:/data/object3/1/{i}.jpg')
        images.append(image)

    save_mp4(f'D:/data/object3/1.mp4', images, 5)
