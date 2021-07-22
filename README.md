# machine-learning

#NUMPY
#PYTHON LIBRARY FOR CREATING N DIMENSIONAL ARRAY
#ABILITY TO QICKLY BROADCAST FUNCTION,BUILT IN LINEAR ALGEBRA,STATISTICAL DISTRIBUTION,TRIGINOMETRIC AND RANDOM NUMBER CAPABILITIES
#NUMPY STRUCTURES ARE MUCH MORE EFFICIENT THEN A STANDARD PYTHON LIST


#NUMPY ARRAYS CREATION
#1.Transforming standard list
#built in functions
#generating random data






#array generation

mylist=[1,2,3]
my_arr=np.array(mylist)
type(my_arr)

my_matrix=[[1,2,3],[1,3,4],[1,5,6]]
#output=[[1, 2, 3], [1, 3, 4], [1, 5, 6]]
np.array(my_matrix)
my_matrix
np.array(my_matrix)
#output=array([[1, 2, 3],
       [1, 3, 4],
       [1, 5, 6]])
       
       
 np.arange(1,10,2)      
array([1, 3, 5, 7, 9])# it goes upto but not including last point


np.zeros([5,5])
np.ones([2,5])

np.linspace(0,10,3)#how many evenly spaced number  between 0 and 10 THEY ARE EVENLY SPACED INCLUDING LAST



np.eye(5)#numpy identity matrix

np.random.rand(1)

array([0.44005132]) #it gives a number between [0,1)



np.random.rand(5,2)

array([[0.58534841, 0.57637531],
       [0.87488622, 0.64613947],
       [0.0343725 , 0.81967927],
       [0.80997996, 0.4959912 ],
       [0.61815559, 0.54628377]])


#standard normal distribution
np.random.randn(2,3)

array([[-1.22672462,  0.10722199, -0.54775252],
       [-1.65239305, -0.33521635,  0.23569301]])
       
np.random.randint(0,100,(4,5))#excluding 100


array([[66, 59, 55, 15,  7],
       [84, 61, 99, 73, 58],
       [26, 48, 67, 36, 70],
       [74, 20,  3, 24, 17]])
       
       
np.random.seed(42)
np.random.rand(4)

array([0.37454012, 0.95071431, 0.73199394, 0.59865848])#we always get the same random numbers,they have to be in same cell



arr=np.arange(0,25)

array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24])
arr.reshape(5,5)

array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])
       
       
arr.max()
24
arr.min()
0
arr.argmin()#for index




#NUMPY INDEXING AND SELECTION


#GRABBING SINGLE ELEMENT
#GRABBING SLICE OF ELEMENT
#BROADCASTING SELECTION
#INDEXING AND SELECTION IN 2-D
#CONDITIONAL SELECTION



import numpy as np
arri=np.arange(0,11)
arri[1:5]



#BROADCAST

arri[0:5]=1000
array([1000, 1000, 1000, 1000, 1000,    5,    6,    7,    8,    9,   10])

slice_of_array=arri[0:5]
array([1000, 1000, 1000, 1000, 1000])
arri
array([1000, 1000, 1000, 1000, 1000,    5,    6,    7,    8,    9,   10])
#copying
arri_copy=arri.copy()



arr_2d=np.array([[1,2,3],[2,3,4],[5,6,7]])
array([[1, 2, 3],
       [2, 3, 4],
       [5, 6, 7]])
)

arr_2d[0,2]
3

arr_2d[:2,1:]

array([[2, 3],
       [3, 4]])
​
