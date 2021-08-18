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
