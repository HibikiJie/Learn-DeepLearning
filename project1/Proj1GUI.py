from PIL import Image, ImageDraw
from project1.Proj1Explorer import Explorer
import tkinter as tk
from tkinter.filedialog import askopenfile


class GUI:

    def __init__(self):
        self.explorer = Explorer()
        self.windows = tk.Tk()
        self.windows.geometry('320x30')
        self.windows.title('多目标人脸侦测')
        self.windows.iconbitmap('face.ico')
        self.path = tk.StringVar()
        self.path_txt = ''
        self.label = tk.Label(self.windows, text="图片路径:").grid(row=0, column=0)
        self.entry = tk.Entry(self.windows, textvariable=self.path).grid(row=0, column=1)
        self.button1 = tk.Button(self.windows, text="路径选择", command=self.select_path).grid(row=0, column=2)
        self.button2 = tk.Button(self.windows, text="开始侦测", command=self.catch_face).grid(row=0, column=3)

    def select_path(self):
        self.path_txt = askopenfile().name
        self.path.set(self.path_txt)

    def catch_face(self):
        image = Image.open(self.path_txt)
        image = image.convert('RGB')
        boxes = self.explorer.explore(image)
        if boxes.shape[0] == 0:
            pass
        else:
            cors = boxes[:, 0:4]
            draw = ImageDraw.Draw(image)
            for cor in cors:
                x1 = cor[0]
                y1 = cor[1]
                x2 = cor[2]
                y2 = cor[3]
                w = x2 - x1
                h = y2 - y1
                c_x = x1 + w // 2
                c_y = y1 + h // 2
                sid_length = max(0.4 * w, 0.3 * h)
                x1 = c_x - sid_length
                y1 = c_y - sid_length
                x2 = c_x + sid_length
                y2 = c_y + sid_length
                x1, y1, x2, y2 = (int(x1), int(y1), int(x2), int(y2))
                cor = (x1,y1,x2,y2)
                draw.rectangle(tuple(cor), outline='red', width=2)
            image.show()

    def main_loop(self):
        self.windows.mainloop()
