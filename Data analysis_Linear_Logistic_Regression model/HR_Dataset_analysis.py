# Author: Kushagra Shukla

# HR data analysis
# Download HR.csv from https://www.kaggle.com/giripujar/hr-analytics
# We will look into how linear regression in ML help in data analysis and step by step we will illustrate that


#Description of coloumns is shown below:

#Coloumn 1: satisfaction_level- It tells about the satisfaction level of employees
#Coloumn 2: last_evaluation- It describes evaluation
#Coloumn 3: number_project- It tells about the total number of projects done by employee
#Coloumn 4: average_monthly_hours- It tells about the total number of hours spend monthly by employee
#Coloumn 5: time_spend_company- It tells about the total number of years spend by employee
#Coloumn 6: Work_accident- It tells about about accident inside company
#Coloumn 7: left- It tells about employees who left the company or still working in the company
#Coloumn 8: promotion_last_5years- Tells about promotion
#Coloumn 9: Department- Tells about Department
#Coloumn 10: salary- Tells about salary (low, medium, high)

#Total dataset of employees are equal to 14999

import time
print("")
print ('\033[31;42;1m' + 'CAUTION!! Install all Modules before you begin' + '\033[0m')
#time.sleep(2)
print ('\033[30;47;1m' + 'Author- Kushagra Shukla' + '\033[0m')
#time.sleep(4)

import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pickle
#Setting rows and coloums limits
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df=pd.read_csv("HR.csv")

#Printing 20 datasets give a look to it
print("")
print ('\033[31;42;1m' + 'Printing our First 20 datasets dataset of HR.csv' + '\033[0m')
data20 = df.iloc[0:20,0:10]
print(data20)
time.sleep(5)

#########################################Exploring our Model##################################################################

print("")
print ('\033[31;42;1m' + 'Exploring and analysing our datasets' + '\033[0m')
time.sleep(2)

print( '\033[35;46;1m' + "Note:You will see 'POPUP WINDOWS' for visual data representation, MAXIMIZE the figure-window and use tools for better analysis when you are done close the window to proceed further" + '\033[0m')
time.sleep(8)

print("")
print ('\033[31;42;1m' + 'Fig1 shows Rentention of employess according to the Satisfaction level' + '\033[0m')
time.sleep(3)

print("Total Employees with satisfaction greater than 0.5 and who left=", len(df [ (df['left'] == 1) &  (df['satisfaction_level'] > 0.5 ) ].index) )
print("Total Employees with satisfaction less than 0.5 and who left=", len(df [ (df['left'] == 1) &  (df['satisfaction_level'] < 0.5 ) ].index) )
print("Total Employees with satisfaction greater than 0.5 and who retained=", len(df [ (df['left'] == 0) &  (df['satisfaction_level'] > 0.5 ) ].index) )
print("Total Employees with satisfaction less than 0.5 and who retained=", len(df [ (df['left'] == 0) &  (df['satisfaction_level'] < 0.5 ) ].index) )

plt.scatter(     df.satisfaction_level   , df.left  ,  color='blue'  ,  marker='.')
plt.xlabel('Satisfaction Level')
plt.ylabel('0-left                                                             1-Retained')
plt.title("Fig1")
plt.show(block=True)
plt.interactive(False)
cv2.waitKey(0)
cv2.destroyAllWindows()

time.sleep(3)

print("")
print ('\033[31;42;1m' + 'Fig2 shows Rentention of employess according to the Salary' + '\033[0m')
time.sleep(3)

print("Total Employees with low salary and who left=", len(df [ (df['left'] == 1) &  (df['salary'] == 'low') ].index) )
print("Total Employees with low salary and who retained=", len(df [ (df['left'] == 0) &  (df['salary'] == 'low') ].index) )
print("Total Employees with medium salary and who left=", len(df [ (df['left'] == 1) &  (df['salary'] == 'medium') ].index) )
print("Total Employees with medium salary and who retained=", len(df [ (df['left'] == 0) &  (df['salary'] == 'medium') ].index) )
print("Total Employees with high salary and who left=", len(df [ (df['left'] == 1) &  (df['salary'] == 'high') ].index) )
print("Total Employees with high salary and who retained=", len(df [ (df['left'] == 0) &  (df['salary'] == 'high') ].index) )

pd.crosstab(df.salary , df.left ).plot(kind='bar')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Fig2")
plt.show(block=True)
plt.interactive(False)
cv2.waitKey(0)
cv2.destroyAllWindows()

time.sleep(3)

print("")
print ('\033[31;42;1m' + 'Fig3 shows Rentention of employess according to the Department' + '\033[0m')
time.sleep(3)
dept=df.Department.unique()


for i in dept:
    print('\033[31;43;1m' +"Department="  , i   + '\033[0m ' )
    print("Left=", len(df [ (df['left'] == 0) &  (df['Department'] == i) ].index) )
    print("Retained=" , len(df [ (df['left'] == 1) &  (df['Department'] == i ) ].index) )

pd.crosstab(df.Department,df.left).plot(kind='bar')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Fig3")
plt.show(block=True)
plt.interactive(False)
cv2.waitKey(0)
cv2.destroyAllWindows()

time.sleep(3)


print("")
print ('\033[31;42;1m' + 'Fig4 shows Rentention of employess according to promotion in last 5 years' + '\033[0m')
time.sleep(3)

cars = ['Promotion+left', 'Promotion+Retained', 'Not Promoted+Left',
         'Not Promoted+Retained']
a=len(df [ (df['left'] == 1) &  (df['promotion_last_5years'] == 1 ) ].index)
b=len(df [ (df['left'] == 0) &  (df['promotion_last_5years'] == 1 ) ].index)
c=len(df [ (df['left'] == 1) &  (df['promotion_last_5years'] == 0 ) ].index)
d=len(df [ (df['left'] == 0) &  (df['promotion_last_5years'] == 0 ) ].index)
data = [a,b,c,d]
print("Total Employees with Promotion and who left=",a)
print("Total Employees with Promotion and who retained=", b )
print("Total Employees with Non-Promoted class and who left=",c)
print("Total Employees with Non-Promoted class and who retained=", d )

fig = plt.figure(figsize=(10, 7))
plt.pie(data, labels=cars)
plt.title("Fig4")
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
time.sleep(3)


print("")
print ('\033[31;42;1m' + 'Fig5 shows Overall Dataset analysis ' + '\033[0m')
time.sleep(3)

left = df[df.left==1]
retained = df[df.left==0]
print("Retained Employees Count=", len(retained.index) )
print("Left Employees Count=", len(left.index) )

df_all=df.groupby('left').mean()
print(df_all)

print("From above table we can draw following conclusions\nSatisfaction level seems to be relatively low (0.44) in employees leaving the firm vs the retained ones (0.66)\nAverage monthly hours are higher in employees leaving the firm (199 vs 207)\nEmployees who are given promotion are likely to be retained at firm")

df.hist()
plt.show(block=True)
plt.interactive(False)
cv2.waitKey(0)
cv2.destroyAllWindows()
time.sleep(3)

########################################### LINEAR REGRESSION (ML) ILLUSTRATION ########################################

print("")
print ('\033[31;42;1m' + 'LINEAR REGRESSION (ML) ILLUSTRATION' + '\033[0m')

print("NOTE: We have salary defied as low,high,medium first we will convert them into numeric values by using suitable formula")

df2=pd.read_csv("HR.csv")
for i in range(0,14999):
    sal=(df['salary'][i])
    s=0
    net=0
    if(sal=="low"):
        s=1
    elif(sal=='medium'):
        s=2
    elif(sal=="high"):
        s=3
    net=s*(25000)
    net+=  1000.484 *(df['satisfaction_level'][i]) #Reward for satisfaction
    net += 100.848 * (df['last_evaluation'][i])  # Reward for evaluation performance
    net += 100.84 * (df['number_project'][i])  # Reward for projects
    net += 10.84 * (df['average_montly_hours'][i])  # Reward for monthly-hours
    net += 800.48 * (df['satisfaction_level'][i])  # Reward for time-spend in company
    net += 2000.58 * (df['Work_accident'][i])  # Reward for not doing any accident inside company
    net += 500.848 * (df['left'][i])  # Reward for not leaving the job
    net += 2500.455 * (df['promotion_last_5years'][i])  # Reward for getting promotion
    df2.loc[i, 'salary'] = net

print("")
print ('\033[31;42;1m' + 'Printing our First 20 datasets of HR.csv with salary calculated according to various fields' + '\033[0m')
data20 = df2.iloc[0:20,0:10]
print(data20)
#time.sleep(5)

#Predicting the salary using linear-regression model

dummy=pd.get_dummies(df2.Department)
merge=pd.concat([df2,dummy] , axis='columns')
fin=merge.drop(['Department','IT' ] , axis='columns')
lm=LinearRegression()

X=fin.drop('salary',axis='columns')
y=fin.salary
lm.fit(X,y)

time.sleep(5)
print("")
print ('\033[31;42;1m' + 'Predicting the salary using our model\nEnter the necessary informations required' + '\033[0m')

ch='y'
while(ch=='y'):

    sl = float( input ("Enter satisfaction-level within range of [0-1]:"))
    le = float( input ("Enter last_evaluation within range of [0-1]:"))
    np = int( input ("Enter number_project:"))
    amh = int(input("Enter average_montly_hours:"))
    tsc=int(input("Enter time spend in company:"))
    wa=int(input("Is there any work accident 0-NO 1-YES:"))
    lft=int(input("Whether Employee left or not 0-NO 1-YES:"))
    prm=int(input("Whether Employee got the promotion in last 5 years 0-NO 1-YES:"))
    dept=df.Department.unique()
    print(dept)
    dep=input("Enter the Department from the above list:")


    print("")
    print ('\033[35;46;1m' + '*************** SALARY OF THE EMPLOYEE IS APPROXIMATED BY LINEAR MODEL IS EQUAL TO ****************' + '\033[0m')

    if(dep=='IT'):
        ps=lm.predict([[  sl , le , np , amh , tsc , wa ,lft , prm , 0,0,0,0,0,0,0,0,0         ]])
        print(ps)
    elif(dep=='RandD'):
        ps = lm.predict([[sl, le, np, amh, tsc, wa, lft, prm, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        print(ps)
    elif(dep=='accounting'):
        ps = lm.predict([[sl, le, np, amh, tsc, wa, lft, prm, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        print(ps)
    elif(dep=='hr'):
        ps = lm.predict([[sl, le, np, amh, tsc, wa, lft, prm, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
        print(ps)
    elif(dep=='management'):
        ps = lm.predict([[sl, le, np, amh, tsc, wa, lft, prm, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
        print(ps)
    elif(dep=='marketing'):
        ps = lm.predict([[sl, le, np, amh, tsc, wa, lft, prm, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
        print(ps)
    elif(dep=='product_mng'):
        ps = lm.predict([[sl, le, np, amh, tsc, wa, lft, prm, 0, 0, 0, 0, 0,1, 0, 0, 0]])
        print(ps)
    elif (dep == 'sales'):
        ps = lm.predict([[sl, le, np, amh, tsc, wa, lft, prm, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
        print(ps)
    elif(dep=='support'):
        ps = lm.predict([[sl, le, np, amh, tsc, wa, lft, prm, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
        print(ps)
    else:
        ps = lm.predict([[sl, le, np, amh, tsc, wa, lft, prm, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        print(ps)



    ch = (input("\nDo u want to predict more press y/n: "))

time.sleep(5)





########################################### LOGISTIC REGRESSION (ML) ILLUSTRATION ########################################
print("")
print ('\033[31;42;1m' + 'LOGISTIC REGRESSION (ML) ILLUSTRATION' + '\033[0m')
print("")
print("From the data analysis we can conclude that we will use following variables as independant variables in our model to predict retention of employee\nSatisfaction Level\nAverage Monthly Hours\nPromotion Last 5 Years\nSalary")

time.sleep(6)

subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
print(subdf.head(10))
print("")
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
df_with_dummies.drop('salary',axis='columns',inplace=True)
X = df_with_dummies
y= df.left
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X , y)

print("")
print ('\033[35;46;1m' + 'We will predict whether employee will work for company or he will leave it using logistic regression' + '\033[0m')
time.sleep(4)


ch='y'
while(ch=='y'):

    sl = float( input ("Enter satisfaction-level within range of [0-1]:"))
    amh = int(input("Enter average_montly_hours:"))
    prm=int(input("Whether Employee got the promotion in last 5 years 0-NO 1-YES:"))
    sala=df.salary.unique()
    print(sala)
    sal=input("Enter the salary range from the above list:")

    if(sal=='low'):
        ps = model.predict([[sl, amh, prm, 0, 1, 0]])
        print("")
        if(ps==0):
            print("Employee is happy and he/she will continue to work :) ")
        else:
            print("Employee is not happy and he/she will leave :( ")
    elif(sal=='medium'):
        ps = model.predict([[sl, amh,  prm,  0, 0, 1]])
        print("")
        if (ps == 0):
            print("Employee is happy and he/she will continue to work :) ")
        else:
            print("Employee is not happy and he/she will leave :( ")
    else:
        ps = model.predict([[sl, amh, prm, 1, 0, 0]])
        print("")
        if (ps == 0):
            print("Employee is happy and he/she will continue to work :) ")
        else:
            print("Employee is not happy and he/she will leave :( ")

    ch = (input("\nDo u want to predict more press y/n: "))

time.sleep(2)
print("\nMODEL ACCURACY PERCANTAGE=")
print('\033[35;46;1m' , model.score(X , y)*100  , '\033[0m')
time.sleep(3)

print("")
print ('\033[33;46;1m' + 'Thank-you Have a nice day' + '\033[0m')


