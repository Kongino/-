import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
 

dataset = pd.read_csv("C:/Users/inoh9/Desktop/ML/8de2/nonmun.csv")
X=dataset.drop('abnormal', axis=1)
Y=dataset['abnormal']

# 테스트 데이터 30%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

 
# 학습 진행
forest = RandomForestClassifier(n_estimators=100)
#forest = DecisionTreeClassifier(random_state=0)
#forest = SVC(kernel='rbf', C=8, gamma=0.1)
forest.fit(x_train, y_train)
 
# 예측
y_pred = forest.predict(x_test)
 
# 정확도 확인
print('학습 정확도 :', metrics.accuracy_score(y_test, y_pred))
'''
predd = pd.read_csv("C:/Users/inoh9/Desktop/ML/Test_other.csv")
PreX=predd.drop('abnormal', axis=1)
PreY=predd['abnormal']
pred_pred=forest.predict(PreX)
print('별도 테스트 세트 정확도 :', metrics.accuracy_score(PreY, pred_pred))
'''
'''
select = SelectFromModel(forest, prefit=True)

X_selected = select.transform(x_test)

print(X_selected.shape)
print(forest.feature_importances_)
'''
