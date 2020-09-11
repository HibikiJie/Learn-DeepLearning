import numpy

a = numpy.array(
    [[33., 96.],
     [120., 80.],
     [69., 87.],
     [67., 114.],
     [88., 149.],
     [41., 88.]]
)

print(numpy.mean(a,axis=0))
