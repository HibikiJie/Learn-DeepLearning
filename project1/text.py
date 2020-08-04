
import os

root = 'D:/data/face'

files = os.listdir(root)

for i,file in enumerate(files):
    if file.endswith('.jpg'):
        # print(file)
        file_name = '2' + str(i).rjust(5,'0')
        new_name = f"{root}/{file_name}.jpg"
        old_name = f'{root}/{file}'
        os.rename(old_name,new_name)
        print(file+">>>>>"+file_name)
    else:
        os.remove(f'{root}/{file}')

    # print(i,file)