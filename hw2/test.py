import numpy as np

a = np.array([[62 , 60],[143,112]])
print(a.T-[0,2])
a = a@[2,1]#/(2**2+1)*[1,2]+[0,3]

print(a)