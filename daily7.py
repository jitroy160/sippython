# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Abhijit)s
"""

# -*- coding: utf-8 -*-
#for loop
ICC_Teams = ['India', 'Australia', 'NewZeland', 'Pakistan', 'England', 'West Indies', 'Srilanka', 'Banladesh', 'Afganisthan']
ICC_Teams
print(ICC_Teams)

for i in range(1,8,2): print(i, end=' ')
    for i in [1,3,6,7]: print(i, end=' ')
range?


teamA = ['India', 'Australia','Pakistan', 'England']
teamA
teamA[0], teamA[1]

#team names
for i in teamA : print("ICC" +i)

#characters of the word
for i in teamA[0]: print(i)

for i in teamA:
    if i == 'India' :
        print('India is in Team A', '\t : ' , i)
        break   #exit if India is found otherwise loop over
    else:
        print("India is not in Team A")
    
#x = 'Pakistan'
x = 'Bangladesh'
teamA
for i in teamA:
    if i == x :
        print(x , " is in Team A", '\t : ' , i)
        break   #exit if x is found otherwise loop over
    else:
        print(x , " is not in Team A")

#
range(6)
for x in range(6) : print(x, end = ' ')

range(2,6)
for x in range(2,6) : print(x, end = ' ')

for x in range(2,10,2) : print(x, end = ' ')


#Nested Loops

#Else in the Loop
for x in range(6):
    print(x, end = ' ')
else:
    print("Finished")