import numpy as np
import pandas as pd



dataset = pd.read_csv('..\datasets\Data.csv')
print("=== csv ===")
print(dataset)
# read and split data
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
#                ^row ^col                   

print("=== X dataset ===")
print(X)
print("=== Y dataset ===")
print(Y)


# handling missing data
# strategy 'mean', 'median', 'most_frequent'
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
print("=== imputer ===")
print(X)



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

#### example different fit vs fit_transfrom
# print("=== example different fit vs fit_transfrom ===")
# cate = X[ : , 0]
# labelencoder_X.fit(cate) # fit เหมือนสร้างโมเดลการปลี่ยน cate เป็นตัวเลข
# print(labelencoder_X.classes_)
# cate_tranformed = labelencoder_X.transform(['France', 'Germany', 'Spain']) # โยนcateเข้าไปเพื่อใน
# print(cate_tranformed)
# print(labelencoder_X.inverse_transform([0, 0, 1]))

# LabelEncoder
# change categorical data into number

X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
print(X[ : , 0])

print("=== inverse_transform label ===")
print(labelencoder_X.inverse_transform(list(X[ : , 0])))

print("=== one hot encoder ===")
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
print(X)
# onehotencoder จะแปลง category ที่อยู่ในรูปคอมลัมตัวเลขคอลัมเดียวเป็น matrix 


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

