# -*- coding: utf-8 -*-
#model selection 모듈
#사이킷런의 model_selection 모듈은 학습 데이터와 테스트 데이터 세트를 분리하거나 교차 검증 분함 및 평가, 그리고 esttimator의 하이퍼 파라미터를 튜닝하기 위한 다양한 함수와 클래스를 제공
#1.학습/테스트 데이터 세트 분리 - train_test_split()
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
df_clf = DecisionTreeClassifier()
train_data = iris.data
train_label = iris.target
df_clf.fit(train_data,train_label)

#학습 데이터 세트로 예측 수행
pred = df_clf.predict(train_data)
print('예측 정확도:',accuracy_score(train_label, pred))
#예측 정확도가 1이 나오는 이유는 학습한 데이터 세트를 기반으로 예측했기 때문이다.
#사이킷런은 train_test_split()를 통해 원본 데이터 세트에서 학습 및 테스트 세트를 쉽게 분리할 수 있다.

#테스트 사이즈를 0.3으로 해서 30%의 데이터를 분리한다.
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size = 0.3,random_state=121)
df_clf.fit(X_train,y_train)
pred = df_clf.predict(X_test)
print('예측 정확도:{0:.4f}'.format(accuracy_score(y_test,pred)))
#적은 양의 데이터를 학습했지만 실제로는 더 많은 양의 데이터를 학습하고 테스트를 해봐야 한다.

#2.교차 검증
#알고리즘을 학습시키는 학습 데이터와 이에 대한 예측 성능을 평가하기 위한 별도의 테스트용 데이터가 필요하다.
#그러나 이 방법 역시 과적합에 취약한 약점을 가질 수 있다.
#과적합은 모델이 학습 데이터에만 과도하게 최적화되어, 실체 예측을 다른 데이터로 수행할 경우에는 예측 성능이 과도하게 떨어지는 것이다
#해당 테스트 데이터만 과적합되는 학습 모델이 만들어져 다른 테스토용 데이터가 들어가면 선응이 저하된다.
#이를 해결하기 위해 교차 검증을 하는 것이다. 예를 들면 본고사를 치르기 전에 모의고사를 여러 번 보는 것이다.

#K 폴드 교차 검증
#K 폴드 교차 검증은 가장 보편적으로 사용되는 기법. 먼저 K개의 데이터 폴드 세트를 만들어서 K번만큼 각 폴트 세트에 학습과 검증 평가를 반복적으로 수행하는 방법
#사이킷런에서는 K폴드 교차 검증 프로세스를 구현하기 위해 StratifiedKFold와 KFold클래스를 제공한다.

iris = load_iris()
features = iris.data
label = iris.target
df_clf = DecisionTreeClassifier(random_state=156)

#5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성
kfold = KFold(n_splits = 5)
cv_accuracy = []
print('붓꽃 데이터 세트 크기:',features.shape[0])
#KFold로 KFold 객체를 생성했으니 이제 생성된 KFold 객체의 split()을 호출해 전체 붓꽃 데이터를 5개의 폴드 데이터 세트로 분리한다.
#150개중 학습용은 120개, 테스트 데이터 세트는 30개로 분할된다.
n_iter = 0

#KFold 객체의 split()를 호출하면 폴드 별 합습용, 검증용 테스트의 로우 인덱스를 array로 반환
for train_index, test_index in kfold.split(features): 
    #KFold.split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_index],features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    #학습 및 예측
    df_clf.fit(X_train,y_train)
    pred = df_clf.predict(X_test)
    n_iter += 1
    #반복 시마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test, pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기 : {2}, 검증 데이터 크기 : {3}'.format(n_iter, accuracy,train_size,test_size))
    print('#{0} 검증 세트 인덱스 : {1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)
    
#개별 iteration별 정확도를 합하여 평균 정확도 계산
print('\n## 평균 검증 정확도 : ',np.mean(cv_accuracy))

























