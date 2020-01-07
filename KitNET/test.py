import numpy as np 
a = np.random.randint(0,10,size = (10,))
print(a)
detected_l  = [index for index in range(len(a)) if a[index]>1 ]
print(detected_l)