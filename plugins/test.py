import numpy as np

arr = np.zeros((8,6))
arr_temp = np.zeros((8,6))


arr[3,0]=4.
arr[3,1]=5.
arr[3,2]=6.
arr[3,3]=7.
arr[2,2]=10.
arr[4,1]=3.

double_row = [3]
i=k=m=0

print(arr)
print("double_row = ",double_row)

while i < 7:
	print("i = ",i," k = ",k)
	if i==double_row[m]:
		print("in double_row")
		for j in range(5):
			arr_temp[i][j]=arr_temp[i+1][j]=arr[k][j]/2.
		i+=1
		if m==0 and len(double_row)>1: 
			m+=1
			double_row[1]+=1
	else:
		for j in range(5):
			arr_temp[i][j] = arr[k][j]

	i+=1
	k+=1
	

print("modifiied cluster")
print(arr_temp)

double_col = [2,3]
j=k=m=0

arr=np.copy(arr_temp)
arr_temp = np.zeros((8,6))
print("double_col = ",double_col)

while j < 5:
	print("j = ",j," k = ",k)
	if j==double_col[m]:
		print("in double_col")
		for i in range(7):
			arr_temp[i][j]=arr_temp[i][j+1]=arr[i][k]/2.
		j+=1
		if m==0 and len(double_col)>1: 
			m+=1
			double_col[1]+=1
	else:
		for i in range(7):
			arr_temp[i][j] = arr[i][k]

	j+=1
	k+=1
	

print("modifiied cluster")
print(arr_temp)