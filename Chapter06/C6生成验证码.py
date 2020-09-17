from PIL import Image,ImageDraw,ImageFont,ImageFilter
import random
import os


class GenerateVerifyCode:

    def __get_code(self):
        char1 = random.randint(48,57)
        char2 = random.randint(65,90)
        char3 = random.randint(97,122)
        char = random.choice([char1,char2,char3])
        return chr(char)

    def __bg_color(self):
        color_r = random.randint(120, 255)
        color_g = random.randint(120, 255)
        color_b = random.randint(120, 255)
        color = (color_r, color_g, color_b)
        return color

    def __ft_color(self):
        color_r = random.randint(0, 160)
        color_g = random.randint(0, 160)
        color_b = random.randint(0, 160)
        color = (color_r, color_g, color_b)
        return color

    def genervericode(self):
        w, h = 240, 60
        image = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("BRADHITC.TTF", 50)
        image_name = ''
        for i in range(w):
            for j in range(h):
                draw.point((i, j),self.__bg_color())
        for i in range(4):
            txt = self.__get_code()
            image_name+=txt
            draw.text((30+i*50+random.randint(-10,10), random.randint(-10,5)), txt, self.__ft_color(), font,stroke_width=random.randint(0,3))
        image = image.filter(ImageFilter.BoxBlur(1))
        image = image.filter(ImageFilter.DETAIL)
        return image,image_name

    def save_image(self):
        imam = self.genervericode()
        path = os.getcwd()
        imam.save(path + "/sss.jpg")


if __name__ == '__main__':
    save_path = 'D:/data/chapter6/test'
    a = GenerateVerifyCode()

    for i in range(2000):
        image, name = a.genervericode()
        image_name = f'{save_path}/{name}.{i}.jpg'
        image.save(image_name)
        print(i)
