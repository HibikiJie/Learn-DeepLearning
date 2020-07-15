from sklearn import preprocessing
import numpy

file = open('D:/sun/test_para_input.txt')
file_data = file.read().split()
a = len(file_data)//11
print(a)
file_name = []
x = []
for i in range(a):
    file_name.append(file_data[11*i])
    x.append(file_data[11*i+1:11*i+11])
file.close()
x = numpy.array(x,numpy.float)
# print(x)

scaler = preprocessing.MaxAbsScaler()
x_scale = scaler.fit_transform(x)
file = open("test_input_para.txt",'w')
print(x_scale)
for name,datas in zip(file_name,x_scale):
    file.write(name)
    for data in datas:
        file.write("  ")
        file.write(numpy.str(data))
    file.write("\n")