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
      
      
      
#conditional arrays 
arr=np.arange(1,11)
arr>4
array([False, False, False, False,  True,  True,  True,  True,  True,
        True])
        
bool_=arr>4
arr[bool_]
array([ 5,  6,  7,  8,  9, 10])
arr[arr>4]
array([ 5,  6,  7,  8,  9, 10])
       
       
#NUMPY OPERATION

arr=np.arange(0,10)
arr+5
arr-5
arr*arr
arr/arr
1/arr #in case of zero it numpy still gives answer like nan or inf
np.sqrt(arr)
np.sin(arr)
np.log(arr)

print(arr.sum())
print(arr.mean())
print(arr.var())
print(arr.std())



np.arange(0,25).reshape(5,5)

array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])


np.arange(0,25)
arr=np.arange(0,25).reshape(5,5)
arr.sum(axis=0)#for sum across the row
arr.sum(axis=1)#for sum across the column




#PANDAS
#CLEAN AND ORGANISE DATA,EXPLORATORY DATA ANALYSIS 
#EXTRMELY POWERFUL TABLE(DATA FRAME) SYSTEM BUILT OF NUMPY
#WHY PANDA INSTEAD OF OPENING SPREAC SHEET OR GOOGLESPREAD SHEET

#WHAT PANDAS CAN DO
#TOOLS FOR READING AND WRITING DATA  BETWEEN MANY FORMAT
#INTELLIGENTLY GRAB DATA BASED ON O=INDEXING AND,LOGIC,SUBSETTING,AND MORE
#HANDLE MISSING DATA
#ADJUST AND RESTRUCTURE DATA
#IT IS LIMITED ONLY BY THE RAM OF YOUR COMPUTER


#WE STUDY
#SERIES AND DATA FRAME
#CONDITIONAL FILTERING AND USEFUL METHODS
#MISSING DATA
#GROUP BY OPERATIONS
#COMBINING DATA FRAMES
#TEXT METHODS AND TIME METHODS
#INPUTS AND OUTPUTS




#SERIES IS DATA STRUCTURE IN PANDAS  THAT HOLDS AND ARRAY OF INFORMATION ALONG WITH A NAMED INDEX
BASICALLY A 1D array with axis list
pandas adds on the labelled index that can be string or names making it easier to grab data in an informative way,data is numerically organised


myindex=['USA','CANADA','MEXICO']
mydata=[1776,1862,1900]
myser=pd.Series(data=mydata,index=myindex)
myser
myser['USA']

ages={'sam':5,'frank':10,'spike':7}
pd.Series(ages)



#grabbing of data

# Imaginary Sales Data for 1st and 2nd Quarters for Global Company
q1 = {'Japan': 80, 'China': 450, 'India': 200, 'USA': 250}
q2 = {'Brazil': 100,'China': 500, 'India': 210,'USA': 260}
sales_q1=pd.Series(q1)
sales_q2=pd.Series(q2)
sales_q1['Japan']

#Broadcast

# Imaginary Sales Data for 1st and 2nd Quarters for Global Company
q1 = {'Japan': 80, 'China': 450, 'India': 200, 'USA': 250}
q2 = {'Brazil': 100,'China': 500, 'India': 210,'USA': 260}
sales_q1=pd.Series(q1)
sales_q2=pd.Series(q2)
sales_q1['Japan']
sales_q1*2  #broad casting work as pajdas is built using numpy array


sales_q1+sales_q2
Brazil      NaN
China     950.0
India     410.0
Japan       NaN
USA       510.0
dtype: float64



#for values not present in both just put 0 for the cas where they are not present
sales_q1.add(sales_q2,fill_value=0)
Brazil    100.0
China     950.0
India     410.0
Japan      80.0
USA       510.0
dtype: float64

#DATAFRAME IS A TABLE OF COLUMNS AND ROWS IN PANDAS THAT WE CAN EASILY RESTRUCTURE AND FILTER
#A GROUP OF PANDAS SERIES OBJECT THAT SHARE THE SAME INDEX

#CREATE DATA FRAME,#GRAB A COLUMN OR MULTIPLE COLUMNS
#GRAB A ROW OR MULTIPLE ROW
#INSERT NEW COLUMN OR NEW ROW

np.random.seed(101)
mydata=np.random.randint(1,101,(4,3))
myindex=['CA','NY','AZ','TX']
mycolumn=['Jan','Feb','mar']
df=pd.DataFrame(mydata,index=myindex,columns=mycolumn)
df

     Jan	Feb	mar
CA	96	12	82
NY	71	64	88
AZ	76	10	78
TX	41	5	64


#how to read in pandas data frame from an existing file such as .csv file

np.random.seed(101)
mydata=np.random.randint(1,101,(4,3))
myindex=['CA','NY','AZ','TX']
mycolumn=['Jan','Feb','mar']
df=pd.DataFrame(mydata,index=myindex,columns=mycolumn)
df.info()
df=pd.read_csv('C:\\Users\\home\\Downloads\\03-Pandas\\tips.csv')
df



mycols=['total_bill','tip']
df[mycols]


df['tip_percentage']=100*df['tip']/df['total_bill']
df['prize_per_person']=df['total_bill']/df['size']
df.head()
