import imageio
import os
def create_gif(image_list, gif_name, duration=1.0):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    :return:
    '''
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def main(images):
    # 这里放上自己所需要合成的图片
    image_list = images
    gif_name = 'new.gif'
    duration = 0.1
    create_gif(image_list, gif_name, duration)


if __name__ == '__main__':
    images = []
    for i in range(0,100):
        file_name = f'D:/data/chapter3/image/epoch{i}.png'
        images.append(file_name)
    for i in range(100, 600,4):
        file_name = f'D:/data/chapter3/image/epoch{i}.png'
        images.append(file_name)
    main(images)
