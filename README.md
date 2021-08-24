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




#SERIES IS DATA STRUCTURE IN PANDAS  THAT HOLDS AN ARRAY OF INFORMATION ALONG WITH A NAMED INDEX
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
np.round(100*df['tip']/df['total_bill'],2)#to calculate round

#to drop the column

df.drop('tip_percentage',axis=1)#it doesnot change permanently#also you have to present actual string not just row number
df.shape[0]
df.shape[1]


#it set and reset the index Payment ID as an index
df.set_index('Payment ID')
df.reset_index()
#column are features
#row are instnaces of the data




#for finding the location

df.iloc[0]
df.loc['Sun2959']

df.iloc[0:4]#slicing

#conditional filtering #when data is large enugh we donot filter based on a number but we typically filter based on a condtion .it allow us to select row
#filter by single condition
#filter by multile condition
#check against multiple possible values
bool_series=df['total_bill']>14
df[bool_series]
#or
df[df['total_bill']>14]

#MULTIPLE CONDTION & (AND) | (OR) for pandas #for python it is (and, or)
df[(df['total_bill']>40)&(df['sex']=='Male')]






#METHOD IN PANDAS

.apply()method call to apply any custom python function of our own to every row in a series.so we can use one or multiple column as input

def last_four(num):
    return str(num)[-4:]
df['CC Number'].apply(last_four)
#it applies this to last four digit of every number
str(num) as we cannot indexed object thus we put the integer as s tring and then apply[-4:]
str(num)[-4:]


#it applies to single column
def yelp(price):
    if price<=10:
        return '$'
    elif price>=10 and price<=30:
        return  '$$'
    else:
        return '$$$'
        
df['total_bill'].apply(yelp)    
      
      
 #these apply function should only return a string value to be applied to every  row to be applied to every series
 
      
#APPLY METHOD 
def simple(num):
   return num*2
can be wriiten in terms of anonymous function using lambda

lambda num:num*2  #used for multiple column #first expression is an argument and can have multiple argument seperated by comma and second expression always return an object 

df['total_bill'].apply(lambda num:num*2)
def quality(total_bill,tip):
    if tip/total_bill>.25:
        return "generous"
    else:
        return "other"
 
 
 
 #NP.VECTOR
 
 df['Quality']=np.vectorize(quality)(df['total_bill'],df['tip'])
 
 
 
 #DESCRIPTION METHOD AND SORTING METHOD
 #describe method is used for determining some  statistical data like percentile ,mean,std of the numerical values
 df.descr
 
 
 
        
        
        
     
df[['total_bill','tip']].apply(lambda df:quality(df['total_bill'],df['tip']),axis=1)  


#sort values


df.sort_values('tip',ascending=False)

df.sort_values(['tip','size'])
df.sort_values('tip')

#first it will sort by tip and then if values in the tip are same then it will sort by size

#grab min and max location 

df['total_bill'].max()#return maximum values
df['total_bill'].idxmax()#return index of the maximum values

#to see how correlated are each other
df.corr()

#counts per catogory

df['sex'].value_counts()
Male      157
Female     87
Name: sex, dtype: int64

#unique

df['day'].unique() #only to get the unique values
#nunique #number of unique values
df['day'].nunique() 





#SWITCHING OUT OR REPLACING VALUES

df['sex'].replace(['Female','Male'],['F','M'])  #REPLACE IS EASIER FOR FEWER ITEMS 

mymap={'Female':'F','Male':'M'}
df['sex'].map(mymap)#MAP IS EASIER FOR LOTS OF ITEMS

#FOR DUPLICATE FUNCTION
simple_df=pd.DataFrame([1,2,2],['a','b','c'])
simple_df.duplicated()
a    False
b    False
c     True
dtype: bool

simple_df=pd.DataFrame([1,2,2,2],['a','b','c','d'])
simple_df.drop_duplicates()
a	1
b	2

df[df['total_bill'].between(10,20,inclusive=True)]


#how to grab n largest and n smallest
df.nlargest(10,'tip')
df.nsmallest(10,'tip')


#sample the data
df.sample(5)
df.sample(frac=.1)

#this wil sample 10%of my data frame



#MISING DATA REAL WOORLD DATA WILL OFTEN BE MISSING DATA FOR A WIDE VARIETY OF REASON

#MANY MACHINE LEARNINGMODELS AND STATISTICAL METHODS CANNOT WORK WITH MISSING DATA POINTS,IN WHICH WE NEED TO DECIDE WHAT TO DO WITH THE MISSING DATA
#options for dealing with missing data is
 1.keep it
 2.Remove it 
 2.Replace it
 
 
 #keeping the missing dta pros
 easiest to do 
 doesnot manipulate or change the true data
 
 
 CONS
 many method doesnot suport NaN 
 often there are resonable guessess
 
 #DROPPING OR REMOVING THE MISSING DATA 
 PROS:
 EASY TO DO 
 CAN BE BASED ON RULES
 
 CONS:
 
 POTENTIAL TO LOSE A LOT OF DATA OR USEFUL  INFORMATION
 LIMITS TRAINED MODEL FOR FUTUTE DATA
 
 
 #FILLING IN THE MISSING DATA
 
 PROS:
 POTENTIAL TO SAVE  A LOT OF DATA FOR USE IN THE TRAINING MODEL
 CONS: HARDEST TO DO  AND SOMEWHAT ARBITRARY 
 POTENTIAL TO LEAD TO FALSE  CONCLUSIONS
 
 
 #filling in the missing data 
 
 
 PROS:
 
 POTENTIAL TO SAVE LOT OF DATA FOR USE IN TRAINING MODEL 
 
 CONS
 HARDEST TO DO  AND SOMEWHAT ARBITRARY  
 POTNETIAL TO LEAD TO FALSE CONCLUSION
 
 #FILLING IN MISSING DATA 
 FILL WITH INTERPOLATED  OR ESTIMATED VALUE
 MUCH HARDER AND REQUIRE REASONABLE ASSUMPTION
 
 
 
 df.isnull() #it return true if there is null value
 
 df.notnull()# return true if there is no null values
 
 df[df['pre_movie_score'].notnull()]
 
 irst_name	last_name	age	sex	pre_movie_score	post_movie_score
0	Tom	Hanks	63.0	m	8.0	10.0
3	Oprah	Winfrey	66.0	f	6.0	8.0
4	Emma	Stone	31.0	f	7.0	9.0


df['pre_movie_score']=df['pre_movie_score'].fillna(0)  #for permanent 




df[(df['first_name'].notnull())&(df['sex'].notnull())]

first_name	last_name	age	sex	pre_movie_score	post_movie_score
0	Tom	Hanks	63.0	m	8.0	10.0
2	Hugh	Jackman	51.0	m	NaN	NaN
3	Oprah	Winfrey	66.0	f	6.0	8.0
4	Emma	Stone	31.0	f	7.0	9.0



#KEEEP THE DATA IS EASY YOU READ  THE DATA AND  KEEEP NAME OF THE MISSING VALUE


#DROP THE DATA IT WILL ASK 
HOW YOU DROP THE DATA IN  ROW OR COLUMN OR ANY CELLS OR VALUES ARE MISSING 

ALSI IF ANY NAN VALUE IS PRESENT DROP IT
OR IF ALLL THE VALUES ARE NAN DROP THEM

THRESH :REQUIRE THAT MANY NON NA VALUES i.e if I WRITE THRES=1 THEN USING df.dropna(thresh=1) means donot remove that row which contains atleast one non null value

df.dropna()

first_name	last_name	age	sex	pre_movie_score	post_movie_score
0	Tom	Hanks	63.0	m	8.0	10.0
3	Oprah	Winfrey	66.0	f	6.0	8.0
4	Emma	Stone	31.0	f	7.0	9.0


it is going to drop the rows which has any missing values


df.dropna(thresh=1)


first_name	last_name	age	sex	pre_movie_score	post_movie_score
0	Tom	Hanks	63.0	m	8.0	10.0
2	Hugh	Jackman	51.0	m	NaN	NaN
3	Oprah	Winfrey	66.0	f	6.0	8.0
4	Emma	Stone	31.0	f	7.0	9.0





df.dropna(thresh=5)#hugh jackman gets dropped cuz he donot has atleast 5 non null values


first_name	last_name	age	sex	pre_movie_score	post_movie_score
0	Tom	Hanks	63.0	m	8.0	10.0
3	Oprah	Winfrey	66.0	f	6.0	8.0
4	Emma	Stone	31.0	f	7.0	9.0



df.dropna(axis=1)#here it says that drop the column who contain any non null values 



0
1
2
3
4




#thus genrally we leave default axis =0 i.e df.dropna(axis=0)



df.dropna(subset=['last_name'])

first_name	last_name	age	sex	pre_movie_score	post_movie_score
0	Tom	Hanks	63.0	m	8.0	10.0
2	Hugh	Jackman	51.0	m	NaN	NaN
3	Oprah	Winfrey	66.0	f	6.0	8.0
4	Emma	Stone	31.0	f	7.0	9.0







#FILLING IN THE DATA 

df['pre_movie_score'].fillna(df['pre_movie_score'].mean())

#to fill in the data with the mean value
df.fillna(df.mean())



first_name	last_name	age	sex	pre_movie_score	post_movie_score
0	Tom	Hanks	63.00	m	8.0	10.0
1	NaN	NaN	52.75	NaN	7.0	9.0
2	Hugh	Jackman	51.00	m	7.0	9.0
3	Oprah	Winfrey	66.00	f	6.0	8.0
4	Emma	Stone	31.00	f	7.0	9.0



airline_tix = {'first':100,'business':np.nan,'economy-plus':50,'economy':30}
ser=pd.Series(airline_tix)
ser

first           100.0
business          NaN
economy-plus     50.0
economy          30.0
dtype: float64




we can interpolate the misssing data which in this case is the average between 50 and 100 i.e 70

ser.interpolate()
first           100.0
business         75.0
economy-plus     50.0
economy          30.0
dtype: float64


##NEVER DO COMPARISION ON THE NULL VALUE USING == INSTEAD USE IS 




##GROUP BY OPERATION
A GROUP BY OPERATION ALLOW US TO EXAMINE DATA ON  A PER CATOGORYY BASIS I.E MEAN PER CATEGORY HOW MANY ROOWS PER CATEGORY
##ONCE YOU HAVE DECIDED THE CATOGORY TO GROUP BY WE SEPERATE OUT ALL THE OTHER COLUMN BY DISTINCT CATOGORY
##THEN LETS SAY WE DECIDE ON SUM THEN ITS GONE TAKE SUM PER CATEGORY 
#SAY IF WE USE THE MEAN THEN IT USE THE AVERAGE VALUE PER CATEGORY 
#COUNT IT COUNT NUMBER OF VALUES PER CATEGORY


avg_year=df.groupby('model_year').mean()
avg_year.index


Int64Index([70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82], dtype='int64', name='model_year')



avg_year=df.groupby('model_year').mean()

mpg	cylinders	displacement	weight	acceleration	origin
model_year						
70	17.689655	6.758621	281.413793	3372.793103	12.948276	1.310345
71	21.250000	5.571429	209.750000	2995.428571	15.142857	1.428571
72	18.714286	5.821429	218.375000	3237.714286	15.125000	1.535714
73	17.100000	6.375000	256.875000	3419.025000	14.312500	1.375000
74	22.703704	5.259259	171.740741	2877.925926	16.203704	1.666667
75	20.266667	5.600000	205.533333	3176.800000	16.050000	1.466667
76	21.573529	5.647059	197.794118	3078.735294	15.941176	1.470588
77	23.375000	5.464286	191.392857	2997.357143	15.435714	1.571429
78	24.061111	5.361111	177.805556	2861.805556	15.805556	1.611111
79	25.093103	5.827586	206.689655	3055.344828	15.813793	1.275862
80	33.696552	4.137931	115.827586	2436.655172	16.934483	2.206897
81	30.334483	4.620690	135.310345	2522.931034	16.306897	1.965517
82	31.709677	4.193548	128.870968	2453.548387	16.638710	1.645161

avg_year['mpg']

model_year
70    17.689655
71    21.250000
72    18.714286
73    17.100000
74    22.703704
75    20.266667
76    21.573529
77    23.375000
78    24.061111
79    25.093103
80    33.696552
81    30.334483
82    31.709677
Name: mpg, dtype: float64



df.groupby(['model_year','cylinders']).mean()#this is the idea of multilevel index
#under the model year it shows the cylinders 

#here model_years and cylinders are not the column name they are the index name they are the two level of the index

df.groupby(['model_year','cylinders']).mean().index


year_cyl=df.groupby(['model_year','cylinders']).mean()

year_cyl.index.names

year_cyl.index.levels

FrozenList([[70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82], [3, 4, 5, 6, 8]])

year_cyl.loc[70]
             mpg	      displacement	weight	      acceleration	origin
cylinders 
4	25.285714	107.000000	2292.571429	16.000000	2.285714
6	20.500000	199.000000	2710.500000	15.500000	1.000000
8	14.111111	367.555556	3940.055556	11.194444	1.000000



year_cyl.loc[[70,82]]#in here we take two coloumn and they act like a multi level index


70    4	25.285714	107.000000	2292.571429	16.000000	2.285714
      6	     20.500000	199.000000	2710.500000	15.500000	1.000000
      8	     14.111111	367.555556	3940.055556	11.194444	1.000000
82    4	32.071429	118.571429	2402.321429	16.703571	1.714286
      6	     28.333333	225.000000	2931.666667	16.033333	1.000000
      
      
      
year_cyl.loc[(70,4)]

here inside the location we put the tuple

mpg               25.285714
displacement     107.000000
weight          2292.571429
acceleration      16.000000
origin             2.285714
Name: (70, 4), dtype: float64


##  .xs()  :- this method takes in the key argument to select data at a particular level of  a multiindex


key:- label or tuple of  a label .Label contained in the index or partially in the multiindex

axis:-'0 for index','1 for the column'

level: it can be reffered by label or position


year_cs.xs(key=70,axis=0,level='model_year')

            mpg	displacement	weight	     acceleration	       origin
cylinders					 


4	25.285714	107.000000	2292.571429	16.000000	2.285714
6	20.500000	199.000000	2710.500000	15.500000	1.000000
8	14.111111	367.555556	3940.055556	11.194444	1.000000



#can't pass for differnt values .xs


year_cyl.loc[[70,80]]


70	4	25.285714	107.000000	2292.571429	16.000000	2.285714
       6	   20.500000	199.000000	2710.500000	15.500000	1.000000
       8	    14.111111	367.555556	3940.055556	11.194444	1.000000
80	3	     23.700000	70.000000	2420.000000	12.500000	3.000000
       4	       34.612000	111.000000	2360.080000	17.144000	2.200000
       5	       36.400000	121.000000	2950.000000	19.900000	2.000000
       6	       25.900000	196.500000	3145.500000	15.050000	2.000000



##year_cyl.xs(key=4,level='cylinders')


df[df['cylinders'].isin([6,8])].groupby(['model_year','cylinders']).mean()



year_cyl.swaplevel()


##it swap the last two level in a multi index



year_cyl.sort_index(level='model_year',ascending =False)

#it sort in the descendiing order 



df.agg(['median','mean'])
#find the aggerigate method allows you to customize what aggeregate function you want per category


df.agg(['sum','mean'])[['mpg','cylinders']]


           mpg	cylinders
sum	9358.800000	2171.000000
mean	23.514573	5.454774



df.agg({'mpg':['max','mean'],'weight':['mean','std']})


           mpg	weight
max	46.600000	NaN
mean	23.514573	2970.424623
std	NaN	       846.841774

#applying specific aggeregate function to a specific column



#often data we need exists in two or more sources,fortunately,pandas make it easy to combine these togehther
3the simplest combination is if both the sources  are already in the same format then concatenation through pd.concat()



axis0 = pd.concat([one,two],axis=0)
axis0##NaN is there because you can't have same  index value  between A and C THUS WE USE NaN 

0       A0	B0	NaN	NaN
1	A1	B1	NaN	NaN
2	A2	B2	NaN	NaN
3	A3	B3	NaN	NaN
0	NaN	NaN	C0	D0
1	NaN	NaN	C1	D1
2	NaN	NaN	C2	D2
3	NaN	NaN	C3	D3
#cocat two string with rows


axis1= pd.concat([one,two],axis=1)##PUT IN THE LIST BRACKET


0       A0	B0	C0	D0
1	A1	B1	C1	D1
2	A2	B2	C2	D2
3	A3	B3	C3	D3

#concat two string with column



two.columns=one.columns
pd.concat([one,two])
#to concat two data together in row is to equal the column index
#if C and D have same features

A	B
0	A0	B0
1	A1	B1
2	A2	B2
3	A3	B3
0	C0	D0
1	C1	D1
2	C2	D2
3	C3	D3



mydf.index=range(len(mydf))
mydf

       A	B
0	A0	B0
1	A1	B1
2	A2	B2
3	A3	B3
4	C0	D0
5	C1	D1
6	C2	D2
7	C3	D3

#there can be column preseent in one data frame and not present in the other or there can be rows present in on =e dat arame but not in other thus we need merge


#the .merge() methood takes in  a key argument labeled  how 
 there are 3 main ways of merging thetables using how parameter :
 1. INNER 
 2. OUTER 
 3. LEFT or RIGHT 




##INNER 

registrations = pd.DataFrame({'reg_id':[1,2,3,4],'name':['Andrew','Bobo','Claire','David']})
logins = pd.DataFrame({'log_id':[1,2,3,4],'name':['Xavier','Andrew','Yolanda','Bobo']})
#HERE WE ASSUME THAT ONLY UNIQUE NAME ARE THERE

# HERE WE NEED TO DECIDE ON WHAT COLUMN DO WE NEED TO MERGE TOGETHER SAY HERE WE USE'NAME'


#TWO RULES 1. THE ON COLUMN SHOULD BE A PRIMARY IDENTIFIER ,MEANING UNIQUE IDENTIFIER FOR THAT ROW 
on="name".
#also how to merge tables on the name column #intitally we start with inner in this case only thosee record will show that match in the both tables.

pd.merge(registrations,logins,how='inner',on='name')

    reg_id	name	log_id
0	1	Andrew	2
1	2	Bobo	4




#LEFT MERGER AND RIGHT MERGE HERE THE ORDER DOES MATTER IN WHICH YO WRITE YOUR DATA FRAMES 


#HERE REGISTRATION ON THE LEFT HAND SIDE ARE MY LEFT TABLE AND REGISTRATION ON THE RIGHT HAND SIDE WILL BE THE RIGHT TABLE

#HERE EVRYTHING THAT IS IN THE LEFT WILL BE PRSENT AFTER WE APPLY MERGE OPERATION WITH how='left'

pd.merge(registrations,logins,how='left',on='name')#in here as you can see it includes the reg_id and excludes the log_id

    reg_id	name	log_id
0	1	Andrew	2.0
1	2	Bobo	4.0
2	3	Claire	NaN
3	4	David	NaN


pd.merge(registrations,logins,how=right',on='name')#in here as you can see it includes the login_id and excludes the registration_id

    reg_id	name	log_id
0	1.0	Andrew	2
1	2.0	Bobo	4
2	NaN	Xavier	1
3	NaN	Yolanda	3


Outer Join#setting how =outer allow us to include everything in both the table
Match up on all info found in either Left or Right Table. Show everyone that's in the Log in table and the registrations table. Fill any missing info with NaN


pd.merge(registrations,logins,how='outer')

      reg_id	name	log_id
0	1.0	Andrew	2.0
1	2.0	Bobo	4.0
2	3.0	Claire	NaN
3	4.0	David	NaN
4	NaN	Xavier	1.0
5	NaN	Yolanda	3.0



#NOW HOW TO JOIN ON A INDEX OR COLUMN

####TIME METHOD FOR DATE AND TIME

python has a datetime object containing date and time information 
#pandas allow us to easily extract information from a datetime object to use feature engineering
we have a recent timestamped sales data .
Pandas will allow us to extract information from the timestamp  such as 
1.Days of the week
2.weekend or weekdays
3.AM vs PM 





import numpy
import pandas
from datetime import datetime
my_year = 2017
my_month = 1
my_day = 2
my_hour = 13
my_minute = 30
my_second = 15
my_date=datetime(my_year,my_month,my_day)
my_date
datetime.datetime(2017, 1, 2, 0, 0)
my_datetime.year

myser=pd.Series(['Nov 3,1990','2000-01-01',None])
myser[0]
'Nov 3,1990'


pd.to_datetime(myser)
0   1990-11-03
1   2000-01-01
2          NaT
dtype: datetime64[ns]


timeser=pd.to_datetime(myser)#it transfer everything to the correct date time format 
timeser[0].year


#how to know something is european date or american date 

obvi_euro_date='31-12-2021'
pd.to_datetime(obvi_euro_date)
Timestamp('2021-12-31 00:00:00')#year downto a month to day then hours sec min


euro_date='10-12-2000'
pd.to_datetime(euro_date)
Timestamp('2000-10-12 00:00:00')#here it assumes that 10 is the month not the date


euro_date='10-12-2000'
pd.to_datetime(euro_date,dayfirst=True)
Timestamp('2000-12-10 00:00:00')#here we take that day is 10 and months and year is 2000


style_date='12--Dec--2000'
pd.to_datetime(style_date,format='%d--%b--%Y')
Timestamp('2000-12-12 00:00:00')



custom_date="12th of Dec 2000"
pd.to_datetime(custom_date)
Timestamp('2000-12-12 00:00:00')


sales=pd.read_csv('RetailSales_BeerWineLiquor.csv')
sales



       DATE	MRTSSM4453USN
0	1992-01-01	1509
1	1992-02-01	1541
2	1992-03-01	1597
3	1992-04-01	1675
4	1992-05-01	1822
...	...	...
335	2019-12-01	6630
336	2020-01-01	4388
337	2020-02-01	4533
338	2020-03-01	5562
339	2020-04-01	5207

type(sales.iloc[0]['DATE'])
str


but after
sales['DATE']=pd.to_datetime(sales['DATE'])

this becomes the date time type


sales['DATE'][0].year
1992

sales=pd.read_csv('RetailSales_BeerWineLiquor.csv',parse_dates=[0])##here it uses the first column as the date time object 

sales['DATE']
0     1992-01-01
1     1992-02-01
2     1992-03-01
3     1992-04-01
4     1992-05-01
         ...    
335   2019-12-01
336   2020-01-01
337   2020-02-01
338   2020-03-01
339   2020-04-01
Name: DATE, Length: 340, dtype: datetime64[ns]


##THUS DO PARSING THE DATE FIRST INSTEAD OF pd.datetime

sales=sales.set_index("DATE")
sales

            MRTSSM4453USN
DATE	
1992-01-01	1509
1992-02-01	1541
1992-03-01	1597
1992-04-01	1675
1992-05-01	1822
...	...
2019-12-01	6630
2020-01-01	4388
2020-02-01	4533
2020-03-01	5562
2020-04-01	5207



sales.resample(rule='A')##here it is like the grop by object##the series offset alias is given in the lecture notes
#'A' is for year end frequency

#USING .DT METHOD()
sales['DATE'].dt.year

0      1992
1      1992
2      1992
3      1992
4      1992
       ... 
335    2019
336    2020
337    2020
338    2020
339    2020
Name: DATE, Length: 340, dtype: int64


##PANDAS INPUT AND OUTPUT CSV FILES


for reading :-df=pd.read_csv('example.csv',index_col=0)
for writing:-df.to_csv('newfile.csv',index=True)
new=pd.read_csv("newfile.csv")
new


pwd
'C:\\Users\\home\\Downloads\\03-Pandas'#where the current file is 
ls
to have the list of file in the current directory


##PANDAS USE IN THE EXCEL 
##PANDAS CAN ONLY READ AND WRITE IN RAW DATA ,IT IS NOT ABLE TO READ IN MACROS ,VISUALIZATION OR FORMULAE CREATED INSIDE OF SPREADSHEETS
##PANDAS TREAT AN EXCEL WORKBOOK AS A DICTIONARY WITH THE KY BEING THE SHEET NAME AND THE VALUE BEING THE DATA FRAME REPRESENTING THE SHEET ITESELF.
NOTE:-USING PANDAS WITH EXCEL REQUIRES ADDITIONAL LIBRARIES



import pandas as pd
df=pd.read_excel('my_excel_file.xlsx',sheet_name='First_Sheet')
df
       a	b	c	d
0	0	1	2	3
1	4	5	6	7
2	8	9	10	11
3	12	13	14	15


list all the sheet name we use 
pd .ExcelFile('my_excel_file.xlsx')


wb=pd.ExcelFile('my_excel_file.xlsx')
wb.sheet_names   
['First_Sheet']

excel_sheet_dict=pd.read_excel('my_excel_file.xlsx',sheet_name=None)
excel_sheet_dict#it is a dictionary
##it is a dictionary where key is the sheet name and the value is the data frame itself


{'First_Sheet':     a   b   c   d
 0   0   1   2   3
 1   4   5   6   7
 2   8   9  10  11
 3  12  13  14  15}
 
 our_df.to_excel('example.xlsx',sheet_name='First_Sheet',index=False)##to save the data frame to the excel file



###PIVOT TABLES
##they allow you to reorganize data refactoring cells based on columns and a new index.
this is best shown visually


A data frame with the repeated values can be pivoted for a reorganization and clarity


df=pd.read_csv('Sales_Funnel_CRM.csv')
df
       Account Number	Company	Contact	Account Manager	Product	Licenses	Sale Price	Status
0	2123398	Google	Larry Pager	Edward Thorp	Analytics	150	2100000	Presented
1	2123398	Google	Larry Pager	Edward Thorp	Prediction	150	700000	Presented
2	2123398	Google	Larry Pager	Edward Thorp	Tracking	300	350000	Under Review
3	2192650	BOBO	Larry Pager	Edward Thorp	Analytics	150	2450000	Lost
4	420496	      IKEA	   Elon Tusk	Edward Thorp	Analytics	300	4550000	Won
5	636685	    Tesla Inc.	Elon Tusk	Edward Thorp	Analytics	300	2800000	Under Review
6	636685	     Tesla Inc.	Elon Tusk	Edward Thorp	Prediction	150	700000	Presented
7	1216870	Microsoft	Will Grates	Edward Thorp	Tracking	300	350000	Under Review
8	2200450	Walmart	Will Grates	Edward Thorp	Analytics	150	2450000	Lost
9	405886	     Apple	Cindy Phoner	Claude Shannon	Analytics	300	4550000	Won
10	470248  	Exxon Mobile	Cindy Phoner	Claude Shannon	Analytics	150	2100000	Presented
11	698032	     ATT	Cindy Phoner	Claude Shannon	Tracking	150	350000	Under Review
12	698032	    ATT	Cindy Phoner	Claude Shannon	Prediction	150	700000	Presented
13	902797	     CVS Health	Emma Gordian	Claude Shannon	Tracking	450	490000	Won
14	2046943	Salesforce	Emma Gordian	Claude Shannon	Analytics	750	7000000	Won
15	2169499	Cisco	Emma Gordian	Claude Shannon	Analytics	300	4550000	Lost
16	2169499	Cisco	Emma Gordian	Claude Shannon	GPS Positioning	300	350000	Presented



licenses=df[['Company','Product','Licenses']]
licenses

Company	Product	Licenses
0	Google	Analytics	150
1	Google	Prediction	150
2	Google	Tracking	300
3	BOBO	Analytics	150
4	IKEA	Analytics	300
5  Tesla Inc.	Analytics	300
6  Tesla Inc.	Prediction	150
7  Microsoft	Tracking	300
8  Walmart	Analytics	150
9  Apple	Analytics	300
10 ExxonMobileAnalytics	150
11  ATT	Tracking	150
12 ATT	       Prediction	150
13CVS Health	Tracking	450
14Salesforce	Analytics	750
15Cisco	Analytics	300
16Cisco  GPS Positioning	300



pd.pivot(data=licenses,index='Company',columns='Product',values='Licenses')##here we take the pivotted values

Product	Analytics	GPS Positioning	Prediction	Tracking
Company				
Google	         150.0	NaN	                  150.0	300.0
ATT	             NaN	NaN	                  150.0	       150.0
Apple	300.0	NaN	NaN	NaN
BOBO	150.0	NaN	NaN	NaN
CVS Health	NaN	NaN	NaN	450.0
Cisco	300.0	300.0	NaN	NaN
Exxon Mobile	150.0	NaN	NaN	NaN
IKEA	300.0	NaN	NaN	NaN
Microsoft	NaN	NaN	NaN	300.0
Salesforce	750.0	NaN	NaN	NaN
Tesla Inc.	300.0	NaN	150.0	NaN
Walmart	150.0	NaN	NaN	NaN




##How much you have sold to each company 

pd.pivot_table(df,index="Company",aggfunc='sum') or df.groupby('Company').sum()
#for summing the license number 



Account Number	Licenses	Sale Price
Company			
Google	6370194	600	3150000
ATT	1396064	300	1050000
Apple	405886	300	4550000
BOBO	2192650	150	2450000
CVS Health	902797	450	490000
Cisco	4338998	600	4900000
Exxon Mobile	470248	150	2100000
IKEA	420496	300	4550000
Microsoft	1216870	300	350000
Salesforce	2046943	750	7000000
Tesla Inc.	1273370	450	3500000
Walmart	2200450	150	2450000


pd.pivot_table(df,index="Company",aggfunc='sum',values=['Licenses','Sale Price'])#in the above examle there is no sense in adding the account no. so here we can use the values




Licenses	Sale Price
Company		
Google	600	3150000
ATT	300	1050000
Apple	300	4550000
BOBO	150	2450000
CVS Health	450	490000
Cisco	600	4900000
Exxon Mobile	150	2100000
IKEA	300	4550000
Microsoft	300	350000
Salesforce	750	7000000
Tesla Inc.	450	3500000
Walmart	150	2450000



pd.pivot_table(df,index=['Account Manager','Contact'],values=['Sale Price'],aggfunc='sum')

                                  Sale Price
Account Manager	Contact	
Claude Shannon	Cindy Phoner	7700000
                      Emma Gordian	 12390000
Edward Thorp	        Elon Tusk	8050000
                    Larry Pager	5600000
                    Will Grates	2800000
                    
                    
                    
                    
                    
                    
                    IT 
                    
                    
                    
                    
                    
 ##MATPLOTLIB
 
 
 
 #VISUALIZING DATA IS CRUCIAL TO QUICKLY UNDERSTANDING TRENDS AND RELATIONSHIP IN YOUR DATASET.
 #MATPLOTLIB IS ONE OF THE MOST POPULAR LIBRARY FOR PLOTTING WITH PYTHON
 #MATPLOTLIB IS HEAVILY INSPIRED BY THE PLOTTING FUNCTION OF THE MATLB ROGRAMMING LANGUAGE
 #IT ALLOWS FOR CREATION OF ALMOST ANY PLOT TYPE AND HEAVY CUSTOMIZATION
 #THERE ARE TWO SEPRATE APPROACHES TO CREAING PLOT FUNCTIONAL BASED METHOD AND OOP BASED METHOD
 
 
 ##FUNCTIONAL BASED METHOD
 
 
 plt.plot(x,y)
 import matplotlib.pyplot as plt ##since we are working with the pypplot region of matplot library 
 ## for older version we are using %matplotlib inline
 
 
 import matplotlib.pyplot as plt
import numpy as np
x=np.arange(0,10)
y=2*x
plt.plot(x,y)  ##it takes this array to the a particular location and rendering( the visualization
plt.show() its just rendering the visualization
## or just put ;in front of plt.plot(x,y)

plt.plot(x,y);
plt.title('String title');#title 


#next 4 lines must be in same block for python to connect to all the title ,x,y label on to a single graph
plt.plot(x,y);
plt.xlim(0,6)#to set the limit in the x axis
plt.ylim(0,15)#to set the limit in the y axis
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title('String title');

#OOP 


#Amore comprehensive Matplot lib OOP API make use of the figure object 
#we then axe to this fiigure object and then lot those on the axes
#this allows for very robust control over the entire plot

#the figure object is not tecchnically visible

##the genral procedure for the figure function in usage

#fig=plt.figure()
#axes=fig.add_axes([0,0,1,1]) ## [0,0] is the lower left corner [1,1] is the width and height of the axes it can be [.5,1] or [.5,.5 ] [.1,.1] 10% of the original height and width 
#axes.plot(x,y)

This methodology allow us to add in multiple axes as well as move and resize the axes

##in theory we could set the axes side by side using plt.figure() calls ,but typically  it is easier to use plt.subplots() function calls for this 
##we will explore multiple side by side plots in a future lecture for now lets explore the figure  object methodology for matplotlib
 
 
fig=plt.figure()#figsize is the figure size and second is the dpi which is the fidelity of dots per square inch
axes=fig.add_axes([0,0,1,1])#first two are the x and y axis and next two are the widht and the height ofthe figure
axes.plot(x,y)
plt.show()


fig=plt.figure(figsize=(12,8),dpi=100)#dots per inch =100 to make it more clear .#dont set it too high as it wil take lots of am for this higher resolution#figure size for size of the figure,to stretch it out a little bit of to make it long
#LARGE AXES
axes1=fig.add_axes([0,0,1,1])
axes1.plot(a,b)
#SMALL AXES
axes2=fig.add_axes([.2,.2,.25,.25]) #if i want to be inside that larger plot i must be 20%on the  x axes  
axes2.set_xlim(1,2)
axes2.set_ylim(0,50)
axes2.set_xlabel('A')
axes2.set_ylabel('B')
axes2.set_title('Zoomed in')
axes2.plot(x,y)
plt.show()
fig.savefig('new_figure.png',bbox_inches='tight')
#we have here plot over the previous plot  
##FIGURE PARAMETER


##MATPLOTLIB _SUBPLOTS FUNCTIONALITY

plt.subplots() call allows us to easily create Figure and Axes objects in side by side formations
##plt.subplots()command returns a tuple conntaining  the figure canvas and then numpy array holding the axes objects.
fig,axes=plt.subplots(nrows=2,ncols=2)#it return a tuple which by common convetion we labeel(fig,axes)

fig:-this is the entire figure canvas 
axes:-this is a numpy array holding each of the axes according  to position  in the overall canvas.






plt.subplots()



import numpy as np
import matplotlib.pyplot as plt
a=np.linspace(0,10,11)
b=a**4
x=np.arange(0,10)
y=x*2
fig,axes=plt.subplots(nrows=1,ncols=2)
type(axes)
numpy.ndarray
axes[0].plot(x,y)
#since axes is the numpy array thus we have the graph of x,y in the left figure

axes[1].plot(a,b)

#if in the above if we do nrows=2,ncol=2 then we get an error#here this numpy array is still 2D
#so now i have to use axes[0][0] to plot on that 
 
##when nrows=3 and ncols=1
then there is no need to do 2D kind of thing it is a list atlast i.e the axes


##fig,axes=plt.subplots(nrows=3,ncols=1)
for ax in axes:
    ax.plot(x,y)
    
axes[0].plot(x,y) plot on the first graph

plt.tight_layout()#it ensures that no set of axes overlap
fig.subplots_adjust()
#left,bottom,right,top,wspace,hspace(width space ,height space)

 
 
 left = 0.125  # the left side of the subplots of the figure
right = 0.9   # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9     # the top of the subplots of the figure
wspace = 0.2  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.2  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
              
              
              
              
fig.subplots_adjust(wspace=1,hspace=1)
#in here for setting the x label or the y label it is same as before just set the 

axes[0][1].set_xlabel('A')
#same for other thing also like for the set the limit 
fig.suptitle("Figurelevel",fontsize=16)#here we place a figure level title with font size 16

fig.savefig('new_subplots.png',bbox_inches='tight')##we have a bounding box that ist'nt going to cuttoff anything 

##Matplot lib styling

##matplot offers very robust styling fnctions that allow us to edit  colors ,legends,line width markers and mch more!


x=np.linspace(0,10,11)
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.plot(x,x)
ax.plot(x,x**2)


##here we donot knoww that it is actually what is i.e is it x vs x or x vs x**2
##thus we add in the legend


ax.plot(x,x,label='X VS X')
ax.plot(x,x**2,label='X VS X^2')
ax.legend(loc='upper left)

using this we can place  a legend

#where to place a legend there is an LOc argument i.e upper left ,upper right etc and also there is a number i.e for best there is 0
upper right 1
upper left 2



ax.legend(loc=4)#here we fix the location according to th number 
ax.legend(loc=(1.1,.5))#here we fix the location by ourself acording to the location
#also negative can also be done in order to place it at the left of the plot


##matplot lib styling based on coloring and styles


ax.plot(x,x,color='blue')

ax.plot(x,x,color='#8f4c47')##after going to hex color code  


ax=fig.add_axes([0,0,1,1])
ax.plot(x,x,color='#8f4c47',label='x vs x')#hex color code
ax.plot(x,x+1,color='#c75048',label='x vs x+1')
ax.legend()


ax.plot(x,x+1,color='#c75048',lw=10)#linewidth =10 that is linewwidth is 10 times the default values i.e 1
ax.plot(x,x+1,color='purple',linewidth=10,linestyle='--')##instead of lw or ls we can wrtie linewidth and line style

lines[0].set_dashes([1,1,1,2,3,5])#to customize the line style

i want 5 solid point 2 blank point 10 solid point and then 2 blank point


x here is not continous 
array([ 0.        ,  1.22222222,  2.44444444,  3.66666667,  4.88888889,
        6.11111111,  7.33333333,  8.55555556,  9.77777778, 11.        ])
        
lines=ax.plot(x,x+1,color='purple',marker='o',linestyle='--',markersize=20,markersize=20,markerfacecolor='red',markeredgewidth=8,markeredgecolor='orange')
#there are many thing like 
1.setting a logarithmic scale axes[1].set_yscale("log")









##SeaBorn


1.Seaborn is a statistical plottingg library that is specifically designed to interact well with pandas data frames to create common statistical plot types.
2.seaborn use sa simple one line syntax

##we tradeofff customization for ease
we can still make 
#sns.scatterplot(x='salary',y='salary',data=df)


##Seaborn takes in a pandas dataframe and then the user provides the corresponding string column names for x and y (depending on the plot type)


##how do i choose what plot to use

IT depend on what question is there you are trying to solve like google image searching chooses a "lot of visualization " will use many flowchart



##SECTION TOPIC
SCATTER PLOTS
DISTRIBUTION PLOTS
CATEGORICAL PLOTS
COMPARISION PLOTS
SEABORN PLOTS
MATRIX PLOTS

##Scatter plot wiith the seaborn 

#Scatter plot show the relationship between two continous feature  
#3continous features are numerical variables that takes any number of values between anny two values
continuos freatures examples
1 age is say for year month day  hours minutess second
2.height :- say someone is 1m tall oor in cm or in nm
3.salry 
4.temperature
5.prices 

#continous feature allows for a values to be always between two values 
#these are not to be confused with categorical feature which represent distinct and unique categories:
colors
shape 
names

#scatter plots line up a set of two continous feature and plot them out  as coordinates.
plt.figure(figsize=(12,4),dpi=200)

sns.scatterplot(x='salary',y='sales',data=df,hue='level of education,palette='Dark2',size='salary')#thus the bigger your salary is the bigger your size is .




###HUE:-the hue paraemter takes in a column and it is going to ask for what should you color these plots by that by what category should we color them 
#hre ussing hue i am able to show 3D information
#hue can be by the continous column or a categorical one.


##how to chose the colors

https://matplotlib.org/stable/tutorials/colors/colormaps.html
so we choose a pallette of colors  


#alpha wiht htiss many point we get many points right on top of another point to elviate this we can set the alpha parameteer an alpha paratemter 
alpha =0 is fully transparent(white) and alpha =1 i it is fully opaque i.e default lets say we take in between alpha =.2 then we can see the transparency i.e where there is more points over the other and find which are more occuring.



##Style Parameter

#i can have markers to be plus ,triangle etc
sns.scatterplot(x='salary',y='sales',data=df,s=200,style='level of education',hue='level of education')


##MAIN THING IS THAT YOU DEFINE YOUR X DEFINE YOUR Y AND DEFINE YOUR DATA EVERYTHING AFTER THAT IS JUST STYLING AND CUSTOMIZING



##DISTRIBUTION PLOTS


IT DISPLAYS A SINGLE CONTINOUS FEATURE AND HELP VISUALIZE PROPERTIES SUCH AS DEVIATION AND AVERAGE VALUES .
THERE ARE 3 MAIN DISTRIBTION PLOT TYPES:
1.RUG PLOT
2.HISTOGRAM
3.KDE PLOT#KERNEL DENSITY ESTIMATION

##EXPLORING THE DISTRIBUTION OF EMPLOYEEE SALARIES 
#ONE WAY IS THE RUG PLOT
IT IS A SIMPLEST DISTRIBUTION PLOT AND MERELY ADDS A DASH AND TICK LINE FOR EVERY SINGLE VALUES ALONG X AXIS
THUS Y AXIS DOESNOT HAVE ANY MEANING FOR THE RUG PLOT

IN THE RUG PLOT IF WE COUNT HOW MANY TICKS THERE ARE PER X  RANGES WE CAN CREATE A HISTOGRAM
3 STEPS ARE THERE 

1.WE SEE HOW MANY CHUNKS WE CAN CUT X AXIS INTO  (FOR EX:-FOR 30 WE HAVE THREE VALUES)
2.THE WE CREATE A BAR AS HIGH AS THE COUNT

#WE CAN ALSSO y AXIS AS THE PERCENT
#WE CAN INCREASE THE NNUMBER OF BINS TO TRY TO GET MORE DETAILS
#BUT GOING TO HIGH ON THE BIN COUNT LEAD TO SOME BIN COUNTS TO BE COMPLETELY EMPTY

##KERNEL DENSITY ESTIMIATION

WE PLOT THE GRAPH ON TOP OF THE HISTOGRAM LEADS TO TELL US WHAT PROBABILITY FUNCITON FUNCTION LOOK LIKE FO THIS PARTICULAR DATA SET


#KDE (KERNEL DENNSITY ESTIMATION IIS A METHOD OF ESTIMATING PROBABILITY DENSITY FUNCTION OF A RANDOM VARIABLE 

#IN SIMPLE TERMS IT IS A WAY OF ESTIMATING A CONTINUOS PROBABILITY CURVE FOR A FINITE DATA SAMPLE 

1. START WITH THE RUG PLOT 
2.WHAT KERNEL WE USE (I.E WHAT PORBABILITY DISTRIBUTON YOU WANT TO TAG ON PROBABILITY DISTRIBUTION ON EACH OF THESE TICK)#THE MOST COMMON IS THE GAUSSIAN DISTRIBUTION
3. WHAT WE ARE GOING TO DO IS TO STACK A NORMAL DISTRIBUTION ON TOP OF EACH OF THESE RUG TICK THUS WE HAVE A GAUSIAN CURVE THAT IS CENTERED AT THAT TICK.
4. ADD THESE ALL CURVE 
5. WE END UP WITH KERNEL DENSITY ESTIMATION PLOT THUS THIS CURVE IS CONTINOUS 
ALSO SOMETIMES IT HAPPEN THAT WE HAVE TO DO THE HARD CUTTOFF AS THERE IS  A PART OF THE GRAPH THAT DONOT HAVE ANY VALUES .


plt.figure(figsize=(5,8),dpi=200)
sns.rugplot(x='salary',data=df)



##sns.distplot##distribution plot
##sns.histplot


sns.set(style='darkgrid')
sns.displot(x='salary',data=df,linewidth=4,edgecolor='red',linestyle='--')#displot is more genralized version so throughtout the course we use the displot
sns.histplot(data=df,x='salary')

sns.set(style='darkgrid')
sns.displot(x='salary',data=df,kde=True,rug=True)#here we have been shown the rug plot ,the kde plot ,the distribution (histogram plot)

sns.kdeplot(x='salary',data=df)#the kdeplot only is shown through this


##NumPy random seed is simply a function that sets the random seed of the NumPy pseudo-random number generator. It provides an essential input that enables NumPy to generate pseudo-random numbers for random processes.



sns.rugplot(data=sample_ages,x='age')  #it gives us the rug plot it is a uniform plot
sns.displot(data=sample_ages,x='age',bins=30)#as we increase bins then we can see way less ticks on those particular bar 
sns.kdeplot(data=sample_ages,x='age',clip=[0,100],bw_adjust=.6,shade=True)#it draws a kde plot##samller bandwidht wil pick up lots of variance here so we can make this a little larger and now we pick up more and more of that actual variance #but is we increase much than we get more and more of the general distribution.
#in order to remove the the extra part coming in the kde plot we use clip
#bandwidth adjust 



##CATEGORICAL PLOTS

HERE THESE PLOTS WILL DISPLAY  ASTATISTICAL METRICS  PER A CATEGORY
FOREX:-MEAN VALUE PER CATEGORY OR COUNT OF THE NUMBER OF ROWS PER CATEGORY
#IT IS THE VISUALIZATION EQUIVALENCE OF A GROUP BY CALL

#THERE ARE TWO MAIN TYPE OF THE PLOT
1. countplot(_
IT COUNTS THE NUMBER OF ROWS PER CATEGORY
2. barplot()
GENERAL FORM OF DISPLAYING ANY CHOSEN METIRC PER CATEGORY 
we could plot out the mean value annd standard deviation per category


df['division'].value_counts()
office supplies      252
printers             250
peripherals          243
computer hardware    159
computer software     96
Name: division, dtype: int64

or 

sns.countplot(data=df,x='division')
plt.ylim(90,260)
sns.countplot(data=df,x='level of education',hue='division')#hue divide level of education to more divisions

countplot is special case of bar plot

df['level of educatioon'].value_counts()#estimator is for various mean variance etc .#but mean is the default value#ci is the confidence interval
sns.barplot(data=df,x='level of education',y='salary',estimator=np.mean,ci='sd',hue='division'



##Distribution within the categories


1.boxplot(more common)
2.violinplot
3.swarmplot
4.boxenplot(letter-value-plot)



Boxplot displays the distribution of a continuos variable .
it does thsi through the use of quartiles.
quartiles seperate out the data into 4 equal number of datapoints 
1.25%of the data points are in bottom quartile and 25 % in the tp quartile
2.50th percentiile (Q2)is the median( it is the line in the middle taht splits the data in half)
#the bos is the interquartile range IQR is the box width it is basically middle 50% of all data points
#out of 50%percent point inside the boox 25th percentile is within the Q1 and 75th percentile within Q3
#Q1 it seperate the line for the bottom 25%of the data points 
#Q3 seperate the line of the above 25%of the data points #highest 25%of all data points are above Q3
#whiskers are defined by 1.5 *IQR
anything outside of this 1.5 range are going to be said as the outiers


#boxplot can quickly give statistical distribution inforamtion in a visual format.


thus we can create box plot per category



VIOLINPLOT




plays the similar role as the boxplot .
1.it displays the probability desity across the data using a KDE 
2.we can imagine it as a mirrored KDE plot
3.WHite dot in the middle is actually going to be the median.
4.Box within this violin plot is going to be the interquartile range





#swarmplot  is a very simple and simplyshows all the data points in the distribution.
#for very large data set it won't show all the points,but will display  the general disstribution of them 

##it  is not mirrored





#the boxen plot


1.it is desinged as an expansion to normal box plot
2.it shows letter value qunatiles  to display against standard boxplot


#creating the plot
##boxplot
plt.figure(figsize=(10,4),dpi=200)
sns.boxplot(data=df,y='math score',x='parental level of education',hue='test preparation course')
plt.legend(bbox_to_anchor=(1.05,1))



sns.violinplot(data=df,y='math score',x='parental level of education',hue='test preparation course',split=True,inner =None )#split lead to one half of the violin be none and other half to be completed #inner =None will remove that bar
#these violin plot are harder to compare compared to the box plot
#inner ='quratile'#quartile for each KDE is drawn
#inner ='stick' going to draw line for every instance
#bw:- for essentially how much noise you want to pick up for stacking and summing up those gaussian distribution so if bw=.1 then KDE are going to pick up lots of noise
#thus the less bw is more noise we are going to pick up and more it is less noise we are going to pick up
#but if you ake bandwidth to large they all start to look the same





##swarmplot
#sns.swarmplot(data=df,x='math score',size=2) for size =2 as by default size is large leadin to not all points are covered

plt.figure(figsize=(8,4),dpi=200)
sns.swarmplot(data=df,x='math score',y='gender',size=2,hue='test preparation course',dodge=True)#here we seperate out the swarm plot based on the gender
#dodge seperates out the two plot.


#boxen plot
sns.boxenplot(x='math score',y='test preparation course',data=df,hue='gender')





##COMPARISION PLOTS


COMPARISION PLOTWS ARE BASICALLY 2D verision of all the plots we have learned so far


the two main plot type we have discussed 
1.jointplot()
2.pairplot()


jointplot()#essentially allow us to tell the relationship between the math score or the reading score 
#we can map histogram to each feature of the histogram to clarify the distribution within each feature
#we can also adjust  the scatter plot to be  a hex or a 2D     KDE plot
#the basic joinplot is the scatter plot of two continous feature that adds in the histogram
#hexagons are dark the more points fall into that area


#2D KDE plots show shaded distribution between the both KDE 




###pairplot()



##the pairplot is qquick way to compare all numerical columns in a DataFrame 

#it automatically creates  a histogram for each column and a scatter plot comparision between  all possible combination of column

##pairplot  can be CPU or RAM intensive for large Data Frames  with many column 
##it is a good idea to filter down only to a column that ou are interested in 

here also we can add in the hue parameter
#3also we change our diagonal to the kde 


##CODING THE PLOT

sns.jointplot(data=df,x='math score',y='reading score',kind='hex')
kind ='hex' means we have darker hexaagon where there are many points 
kind='kde' replace the hisstogram and shows us this KDE 


sns.jointplot(data=df,x='math score',y='reading score',hue='gender')

#PAIR PLOT


sns.pairplot(data=df)
#herre we have the same plot along the diagonal 
#using 'hue ' leads to diagonal to be the KDE for you
#if you want the diagonal to be the histogram use diag_kind='hist'

#it will took some more time to process as it deal with whole dataframe
sns.pairplot(data=df,hue='gender',diag_kind='hist',corner=True)
#corner =True get rid of the duplicate values 




#GRID PLOT 


Seaborn grid calls use Matplotlib subplots() to automaticaly create a grid based off a categorical column.
Instead of passing in a specific number of cols or rows for the subplots ,we can simply pass in thhe names of the column  and seaborn will automatically map the subplots grid
many of the seaborn's built in plot calls are running on the top of the grid system .
directly calling the grid system  allow user to heavily customize plots.

#FOR SEABORN GRID SYSTEM TWO TYPES OF PLOTS CAN BE USED 

1. catplot()  
sns.catplot(data=df,x='gender',y='math score',kind='box',row='lunch')
#each row represent a different lunch type 
sns.catplot(data=df,x='gender',y='math score',kind='box',col='lunch')
#here each column is based of some lunch type 



sns.catplot(data=df,x='gender',y='math score',kind='box',col='lunch',row='test preparation course')
#for grid go for 2*2 the maximunm size 



2.pairgrid()

sns.pairplot(df)##essentially what is happening as this is calling the pair grid functionality which creates the grid and just fills it in



sns.PairGrid(df)#after thisit is waiting for us to provide mapping command taht it wants us to provide

#g=g.map_upper(sns.scatterplot) # for upper diagonal 
g=g.map_lower(sns.kdeplot)
g=g.map_diag(sns.histplot)#fro diagonal it is better to use the histogram instead of scatter plot







#MATRIX PLOT 


MATRIX PLOT ARE THE CISUAL EQUIVALENT OF DISPLAYIN GA PIVOT TABLE 
THE MATRIX PLOT DISPLAYS  ALL THE DATA  PASSED IN ,VISUALIZING  ALL THE NUMERIC VALUES IN A DATA FRAME

NOTE:-NOT EVERY DATAFRAME IS A VALID CJOICE FOR  A MATRIX PLOT SUCH AS HEAT MAP


#THE TWO MAIN TYPES OF MATRIX PLOT IS THE 

1. heatmap():-it visually displays the distribution of cell values with a color mapping
2. 2.clustermap():-same visual as heat map but first conducts hierarchial clustering to reorganize data into groups 
3. #at first sea born group together a very similar data rows or data columns to see which grouping are very siimilar before visualizing with color



#HEAT MAP 


#PASSING THE DATA FRAME INTO THE HEAT MAP NOW I ONLY HAVE THE RATE COLUMN

#NOTE THAT HEAT MAP SHOULD IDEALLY HAVE ALL THE CELLS BE INTHE SAME UNIT,SO THE COLOR AMPPPING MAKES MORE SENSE  ACROOS THE ENTIRE DATAFRAME 

#IF WE ADD THE AGE THEN WE CAN SEE IT IS OF DIFFERRENT UNIT THAN THE RATE THUS IT IS OF DIFFERNET COLOR 

#THUS THIS IS THE REASON EVERY DATA FRAME CANNOT BE PASSED INTHE HEAT MAP AS ALL THE CELL ARE INN THE DIFFERETN UNIT AS AGE IS IN THE YEARS



#heatmap here have to drop life expectancy 
sns.heatmap(df.drop('Life expectancy',axis=1),linewidth=.5,annot=True,,cmap='viridis')#annot means we can ask to have the values,#viridis in this case extreme jump out a littlemore 


#to centre the colormap


#for the cluster map
plt.figure(dpi=200)
sns.clustermap(df.drop('Life expectancy',axis=1),linewidth=.5,annot=True,cmap='viridis',col_cluster=False)  #col_cluster is not there







#CAPSTONE PROJECT


RECALL IN AMACHNELEARNING PATHWAY WE HAVE A DATA ANALYSIS PROBLEM WE NEED TO
1.COLLECT & STORE DATA:-SCRAPE REVIEW OF VARIOUS WEBSITES
2.CLEAN &ORGANIZE DATA:-PANDAS TO ORGANIZE THE DATA REFORM IT SEABORN TO VISUALIZE IT .
3.EXPLORATORY DATA ANALYSIS(REPORT ,VISUALIZATION,COMMUNICATION THESE THINGS TO ANSWER KEY QUESTION):-USE SEABORN TO EXPLORE
4.DATA ANALYSES
5.MAKE DECISION AND ANSWER KEY QUESTION:-DOES FANDANGODISPLAY ARTIFICIALLY HIGHER THAN AVERAGE REVIEW?


#FANDANGO HAS TWO RATING 

STARS:-
1.RATING IN STARS 0-5 DISPLAYED ON THEIR WEBSITE  HTML
RATING:-
1.ACTUAL TRUE RATING NUMERICALLY SHOWN ON THE MOVIE PAGE



FIRST WE WILL CHECHK TWO RATINGS  AND COMPAE THESE RATING TO CHECK FOR DISCREPENCIES
SECONDLY CHECK FUNDANGO RATING TO OTHER WEBSITES AND SEE IF THERE IS A SKEW .




##MACHINE LEARNING


MACHINE LEARNING PATHWAY YOU HAVE A PROBLEM TO SOLVE  OR QUESTION TO ANSWER
#SO WE ARE trying to build data product instead of just doing data analysis
#we know how to collect and store data .
#clean and organize data 
#perform exploratory analysis
#now we can do  machine learning models
1.Supervised learning i.e predict an outcome 
2.unsupervised learning  i.e  discover pattern in the data 



#OUR MAIN GOAL IN ML OVERVIEW SECTION
1.PROBLEM SOOLVED BY THE MACHINE LEARNING
2.TYPES OF MACHINE LEARNING
 #SUPERVISED AND UNSUPERVISED
3.ML PROCESS FOR SUPERVISED LEARNING 

##WHY MACHINE LEARNING ?
MACHINE LEARNING IN GENERAL  IS THE STUDY  OF STATISTICAL COMPUTER ALGORITHMS THAT RELY ON HUMAN INPUT FOR  WHAT APPROACH TO TAKE ,ML INFERS THE BEAT APPROCAH FRO THE DATA ITSELF
MACHINE LEARNING IS THE SUBSET OF ARTIFICIAL INTELLIGENCE 
#ML ALGORITHM ARE NOT EXPLICILY PROGRAMMED ON WHICH DECISION TO MAKE ,INSTEAD THE ALGORITHM IS DESIGNED TO INFER FROM THE DATA THE MOST OPTIMAL CHOICE TO MAKE .



WHAT KIND OF PROBLEM MACHINE LEARNING CAN SOLVE 

1.CREDIT SCORING
2.INSURANCE RISK
3.PRICE FORECASTING
4.EMAIL SPAM FILTERING 
5.CUSTOMER SEGMENTATION (dATA OF THE CUSTOMER)
ETC



#STRUCTURE OF ML PROBLEM FRAMING 
1.GIVEN FEATURES FROM A DATA SET OBTAIN  A DESIRED LABEL(SO DATA HA SFETURE SO WE EITHER WANT TO PRDICT THE FEATURE LABEL OR WE WANT TO TRY TO ASSIGN A LABEL)
2.ML ALGORITHM ARE ALSO CALLED "ESTIMATOR " SINCE THEY  RE ESTIMATING THE DESIRED LABEL OR OUTPUT



#NOW HOW CAN ML BE SO ROBUST  IN SOLVING ALL SORT OF PROBLEM?
ANS. MACHINE LEARINIGN RELY ON  DATA ANS A SET OF STATISTICAL METHOD  TO LEARN WHAT FEATURE  ARE IMPORTANT IN THE DATA (IT DOESN'T REQUIRE ANY HUMAN TO TELL WHAT TO USE OR NOT .


#SIMPLE EXAPLE IS PREDICT THE PRICE OF THE HOUSE GIVEN IT CURRENT FEATURE WE SHOULD SELL I.E WE HAVE ALL THESE DATA FEATURES OF THE HOUSE SUCH AS BEDROOM ITS AREA,ITS BATHROOMS .


#HOUSE PRICE PREDICTION 


TYPICAL ALGORITHM 
#HUMAN USER DEFINES  AN ALGORITHM TO  MANUALLY SET VALUES OF IMPORTANCE FOR EACH FEATURE 

MACHINE LEARNING ALGORITHM

ALGORITHM AUTOMATICALLY DETERMNES THE IMPORTANCE OF EACH FEATURES  FROM EXISTING DATA



WHY MACHINE LEARNING ?

1.MANY COMLEX PROBLEM ARE ONLY SOLVABLE WITH MAHCINE LEARNING TECHNIQUES 
2.PROBLEMS SUCH AS SPAM EMAIL OR HANDWRITING  IDENTIFICATION  REQUIRE ML FOR AN EFFECTIVE SOLUTIION .



#WHY MACHINE LERANING IS NOT USED FOR EVERYTHING
1. MAJOR CAVEAT  TO EFFECTIVE ML IS GOOD DATA .
2. MAJORITY  OF DEVELOPEMENT TIME IS SPENT LEANING AND ORGANISNG DATA ,NOT IMPLEMENTING ML ALGORITHM I.E MOST OF THE TIME IS SPENT ON DEVELOPING A GOOD DATA SET
#THERE IS NO SUCH NEED TO DEVELOP AN ALGORITHM SINCE THESE TECHNIQUES ARE WELL DOCUMENTED AND WELL DEVELOPED THUS WE CAN IMPORT THOSE ALGORITHM FROM  SCIKIT LEARNING 


#TYPES OF MACHINE LEARNNING ALGORITHM 
1.SUPERVISED MACHINE LEARNING ALGORITHM 
2.UNSUPERVSED MACHINE LEARNING ALGORITHM 


SUPERVISED LEARNING HAS THE HISTORICAL AND LABELED DATA AND FROM THIS DATA THE MACHINE LEARNING MODEL PREDICTS A FUTURE  VALUE.
UNSUPERVISED LEARNING :-APPLIED TO UNLABELED DATA , THE MACHINE LEARNING MODEL THEN DISCOVERS THE POSSIBLE PATTERN  IN THE DATA .


SUPERVISED LEARNING REQUIRES HISTORIICAL LABELED DATA I.E FOR EX:-PRICE OF THE DATA WE HAVE SQUARE FOOTAGE SQUARE AREA NUMBER OF BEDROOMS BATHROOMS
1.HISTORICAL MEANS -:KNOWN RESULT AND DATA FROM THE PAST
2.LABELED :-MEANS YOU KNOW THE DESIRED OUTPUT FOR THE HISTORICAL  DATA .

TWO MAIN LABEL TYPES:-
1.CATEGORICAL VALUE TO PREDICT I.E LABEL HAS DISCRETE VALUE 
#CLASSIFICATION TASK
2. CONTINUOS VALUE TO PREDICT I.E LABEL HAS CONTINUOUS VALUE
#REGRESSION TASK



#CLASSIFICATION TASK 
1.PREDICT AN ASSIGNED VALUE CATEGORY 
IT HELPS DOCTOR TO PREDICT IF THE TUMOR IS 
#CANCEROUS VS BENIGN TUMOR 
ARE THE PEOPLE PAY OFF THERE LOAN OR NOT IT HELP TO PREDICT WHTHER THE CCURETN CUSTOMER IS CREDIT WORTHY 
#FULFILLMENT VS CREDIT DEFAULT 
PREVIOUS ONE ARE TTHE BINARY CALSSIFICATION BUT FOR HANDWRITIN I CAN HAVE EACH LETTER BE ITS OWN CLASS I.E WHAT LETTER A IMAG EIS CORRESPONDING TO IS IT ABCD ETC.
##ASSGINING IMAGE CATEGORY  
HANDWRITING RECOGNITION 



#REGRESSION TASK 
PREDICT CONTINOUS VALUES:-
1. FUTURE PRICES 
2.ELELCTRICITY LOAD
3.TEST SCORES





UNSUPERVISED LEARNING



#GROUP  ADN INTERPRET DATA WITHOUT  A LABEL 
EXAMPLE:-
CLUSTERING CUSTOMERS INTO THE SEPERATE GROUPS BASED ON THERE BEHAVIOUR FEATURES 


#MAJOR DOWNSIDE IS BECAUSE THERE WAS NO HISTORICAL "CORRECT LABEL" ,IT IS MUCH HARDER TO EVALUATE PERFORMANCE OF AN UNSUPERVISED LEARNING ALGORITHM.





SUPERVISED MACHINE LEARNING PROCESS 


#MACHINE LERANING PATHWAY:-


FIRSTLY FROM REAL WORLD WE AHVE TO COLECT AND STORE THE DATA  THEN CLEAN AND ORGANIZE IT  AND THEN DO EXPLORATORY DATA ANALYSIS  THEN TO GO BEYOND THE DATA ANALYSIS WE WILL USE MACHINE LEARNING MODEL TO HAVE THE DATA PRODUCT 


#FOR THE ABOVE WE HAVE ,JUPYTER ,NUMPY,MATPLOTLIB,,PANDAS,SEABORN AND FOR THE DATA MODEL WE HAVE THE SCIKIT LEARN



SUPERVISED LEARNING MODEL

ML PROCESS :FOR SUPERVISED LEARNING TASK 

EXAMPLE :- PREDICT THE PRICE WE SHOULD SELL OUR

STEPS
1. START OF BY COLLECTING AND ORGANISING DATA SETS BASED OF THE HISTORY (HISTORICAL LABELED DATA SET ON PREVIOUSLY SOLD HOUSES )


PROBLEM
1.IF A NEW HOUSE COMES COMES ON THE MARKET WITH A KNOWN AREA ,BEDROOM ,BATHROOMS THEN PRDICT WHAT PRICE SHOULD WE SELL IT AT 

DATA PRODUCT :
1.INPUT HOUSE FEATURES 
2.OUTPUT PREDICTED SELLING PRICE 
 
 
 SO WE ARE USING A HISTORICAL ,LABELED DATAPREDICT A FUTURE OUTCOME OR RESULT 



SOLUTION 

1. DATA , seperate that data into x=features(known characteristics or component in the data ) and y= labels(it is just a term for what are we trying to predict here ti si the price )

2.split this data set into TRAIN AND TEST DATA SET

3. THEN WE FIT AND TRIN THE MODEL AND WE EVALUATE ITS PERFORMANCE ABSED ON THE TEST SET 

4.IF YOU ARE UNSATISFIED WITH THE PERFORMANCE ONT HE TEST SET YOU CAN GO BACK AND ADJUCST YOUR MODEL PARAEMETER AND RETRAIN THE MODEL ON THE TRAINING DATA SET  
THEN EVALUATE THIS ADJUSTED MODEL BASED ON THE TEST DATA SET REPEAT THIS PROCESS A S NECESSARY .
5. THEN WE CAN DEPLOY THE MODEL TO THE REAL WORLD 
6. THEN WE CAN DEPLOY THE MODEL AS A SERVICE ,DASHBOARD APPLICAITON ,THEN CREATED DATA PRODUCT TO PREDICT THE OUTCOME GAIN INSIGHT ON THE DATA .  
#NOW HOW TO SPLIT AND WHY TO SPLIT 

TRAIN AND TESTT SET IS THERE TO FAIRLY EVALUATE THE PERFORMANCE OF OUR MACHINE LEARNING MODEL

1. WE TAKE OUR TRAINING SET ADN HAVE OUR ML MODEL LEARN FROM IT .
2. THEN EVLUATE THEM BASED ON THE DATA THEY HAVEN'T SEEN BEFORE .I.E BY ASKING BY GIVING THE FEATURE TO PREDICT THE LABEL I.E THE OUTPUT 
3. THEN COMPARE PRDICTION AGAINST THE TRUE PRICE (Y HAT IS FOR THE PREDICTION AND Y FOR THE OUTPUT.



WE HAVE FEATURE FOR TRAING OR LABELS FOR TRAINING AND WE ALSO HAVE XTRAIN Y TRAIN X TEST Y TEST ( SAME USED IN THE PYTHON) 





#LINEAR REGRESSION 
1. THEORY OF LINEAR REGRESSION 
2. SIMPLE IMPLEMENTATION WITH PYTHON 
3. SCIKIT-LEARN OVERVIEW 
4. LINEAR REGRESSION WITH SCIKIT LEARN 
5. PLOYNOMIAL REGRASSION 
6. REGULARIZATION 
7. OVERIEW OF PROJECT DATA SET 



LINEAR REGRESSION 


LINEAR RELATIONSHIP IMPLIES SOME CONSTANAT SPACE TIME RELATIONSHIP
#SIMPLEST POSSIBLE IS Y=X
NOW BAED ON X=[1,2,3] AND Y=[1,2,3] WE CAN BULID OUT SOME THEORITICAL RELATIONSHIP Y=X I DON'T KNOW IF THIS IS TRUE RELATIONAHSIP BETWEEN THESE THREE POINTS BUT IT ESSENTIALLY HAS A PERFECT FIT IMPLYING FOR A NEW VALUE X=1.5 I CAN DECIDE WAHT IS THE VALUE OF THE  Y 


BUT WITH REAL DATA ITS SI NOT GOING TO FIT PERFECTLY ON REAL LINE  HTUS WE HAVE TO DO IS TOO BASICALLLY DECIDE TO WHERE TO DRAW THIS LINE OF BEST FIT  

FUNDAMENTALLY WHAT WE DO NEED TO MINIMIZE IS THE DISTANCE BETWEEN THE POINTS AND THE LINE

THE ERROR IS CALLED THE RESIDUAL ERROR . #THUS WE TRY TO MINIMIZE THI RESIDUAL ERROR 

ORDINARY LEAST SQUARES WORKS BY MINIMIZING THE SUM OF THE SQAURES OF THE DIFFERENCE BETWEEN THE VALUES OF THE VARIABLE BEING OBSERVED AND THOSE PREDICTED BY THE LINEAR FUCNTION



##ORDINARY LEAST SQUARES 
 
 
 WE KNOW THAT THE EQUATION OF A LINE IS Y=mX+b here we have one X is instance of  feature  and Y is the label (or what we want to predict )
 m is the slope b is the intercept these things are going to be later on fixed 
 ##so OLS will allow us to directly solve  for the slope m and intercept b for the given input feature X and the output Y #LATER WE SEE ONLY IF WE AHVE MULTIPLE FEATURE WE USE GRADIENT DESCENT TO SCALE THIS 
 
 
 
 ##TRANSLATE THIS DATA INTO A GENERALIZED MATHEMATICAL NOTATION
 1. I WIL SAY I AHVE A MATRIX X CONTAINING MULTIPLE FEATURES AND VECTOR Y CONTAINING LABEL THAT I AM TRYING TO PREDICT 
 2. 2.X->x1,x2,x3 intheory it could go to xn nuber of features and now insted of price i am to predict my Y
 3. then for each say x1 we ahve a superscript for the row and subscript for the feature and then same for rest of the data point 
 
 


linear regression
1. basically each feature should have a beta and the final output yhat(my prediction )=beta1 x1+beta2 x2+ beta3 x3... +betan xn (i.e we have n number of feature in our data set)(this we hve a line equation )

2. n a linear regression where we try to formulate te relationship between  variables y=mx+b becoomes => y (hat)=b0+b1*x

our goal is to predict hte dependent variable(y) based on that  of an independent variable (x)

b1=phox,y * sigma y/sigma x  (phox,y is the correlation coefficient ,sigma x ,sigma y are the standard deviation )

#b1=sum(xi-x(bar)(yi-y(bar))/sum((xi-x(bar))^2)
#b0=ybar-b1*x(bar)


#BUT AS THE NUMBER OF FEATURES INCREASES  AN ANALYTICAL SOLUTON BECOME UNSCALABLE 
#SO INSTEAD OF OLS WE SHIFT FOCUS ON MINIMIZING A COST FUNCTION  WITH GRADIENT DESCENT .

#SO WE ARE GOING TO TO USE GRADIENT DESCENT TO SOLVE ACOST FUCNTION  TO CALCULATE BETA VALUES WHEN WE ARE DEALING WITH MORE THAN ONE FEATURE 
#THIS LEAD TO SCALE N NUMBER OF FEATURE GIVEN A GENRALIZED FORMULA 



##COST FUNCTION

WE HAVE DECIDED THE BEST FIT  AS MINIMIZING THE SQUARED ERROR 
y(hat)=sum(i=0 to i=n)( bi* xi)


residual error =yj(true )-y(hat)j
##AVERAGE SQUARE ERROR FOR MY PREDICTION
(sum(j=1 to j=m)(yj(true )-y(hat)j)^2)/m #summation of all the errors for m row is this

#cost function j
J(beta)=this is cost function defined by some measure of error .thus we need to minimize the cost function i.e we need to minimize the cost fucntion i.e minimize the cost function=(sum(j=1 to j=m)(yj(true )-y(hat)j)^2)/2*m

#Recall to minimize we are taking some derivative and reduce it to zero thus 2 is put to ease the clculation
y(hat)j is the function of beta and features =sum(i=0 to i=n)(betai) *xij


##THUS TAKE THE DRIVATIVE WITH RESPECT TO THE BETA VALUES #NOTEE IT IS NOT SCALABLE  TO TRY TO GET  AN ANALYTICAL SLUTION TO MINIMIZE THIS COST FUCNITON 
THUS WE USE GRADIENT DESCENT .TO MINIMIZE THE COST FUNCTION



