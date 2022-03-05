import pandas as pd
import joblib
df = pd.read_csv("CreditCardDefault.csv")
df=df.dropna()
x = df.loc[:,["loan", "income", "age"]]
y = df.loc[:,["default"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y)

from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=3)

from sklearn.metrics import confusion_matrix
model.fit(x_train,y_train)
pred = model.predict(x_test)
cm = confusion_matrix(y_test, pred)
print(cm)
print((cm[0,0]+cm[1,1])/(sum(sum(cm))))
joblib.dump(model,"CCD_DT")

from sklearn import linear_model
model2 = linear_model.LogisticRegression()
model2.fit(x_train,y_train)
pred2 = model2.predict(x_test)
cm2 = confusion_matrix(y_test, pred2)
print(cm2)
print((cm2[0,0]+cm2[1,1])/(sum(sum(cm2))))
joblib.dump(model2,"CCD_Reg")

from sklearn.neural_network import MLPClassifier
model3 = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(6,6))
model3.fit(x_train,y_train)
pred3 = model3.predict(x_test)
cm3 = confusion_matrix(y_test, pred3)
print(cm3)
print((cm3[0,0]+cm3[1,1])/(sum(sum(cm3))))
joblib.dump(model3,"CCD_NN")

from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier().fit(x_train,y_train)
pred4 = model4.predict(x_test)
cm4 = confusion_matrix(y_test, pred4)
print(cm4)
print((cm4[0,0]+cm4[1,1])/(sum(sum(cm4))))
joblib.dump(model4,"CCD_RFC")

from sklearn.ensemble import GradientBoostingClassifier
model5= GradientBoostingClassifier().fit(x_train,y_train)
pred5 = model5.predict(x_test)
cm5 = confusion_matrix(y_test, pred5)
print(cm5)
print((cm5[0,0]+cm5[1,1])/(sum(sum(cm5))))
joblib.dump(model5,"CCD_GBC")