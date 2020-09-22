import os

path = 'D:/data/chapter6/test'


def finde_dir(path):
    for file in os.listdir(path):
        file_path = f'{path}/{file}'
        if os.path.isdir(file_path):
            finde_dir(file_path)
        else:
            print(file_path)
            os.remove(file_path)
    print(path)
    os.removedirs(path)

finde_dir(path)