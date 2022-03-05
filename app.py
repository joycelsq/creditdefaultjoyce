from flask import Flask

app = Flask(__name__)

from flask import request, render_template
import joblib

@app.route("/", methods=["GET","POST"])
def index():
    if request.method =="POST":

        loan = request.form.get("loan")
        income = request.form.get("income")
        age = request.form.get("age")
        loan = float(loan)
        income = float(income)
        age = float(age)
        print(loan, income, age)

        model1 = joblib.load("CCD_DT")
        pred1 = model1.predict([[loan, income, age]])
        s1 = "The score of credit card default based on decision tree is " + str(pred1)

        model2 = joblib.load("CCD_GBC")
        pred2 = model2.predict([[loan, income, age]])
        s2 = "The score of credit card default based on gradient boosting is " + str(pred2)

        model3 = joblib.load("CCD_NN")
        pred3 = model3.predict([[loan, income, age]])
        s3 = "The score of credit card default based on neural network is " + str(pred3)

        model4 = joblib.load("CCD_Reg")
        pred4 = model4.predict([[loan, income, age]])
        s4 = "The score of credit card default based on linear regression is " + str(pred4)

        model5 = joblib.load("CCD_RFC")
        pred5 = model5.predict([[loan, income, age]])
        s5 = "The score of credit card default based on random forest is " + str(pred5)

        return(render_template("index.html",result1=s1, result2=s2, result3=s3, result4=s4, result5=s5))
    else:
        return(render_template("index.html",result1="2", result2="2", result3="2", result4="2", result5="2"))

if __name__ =="__main__":
    app.run()

#pip freeze requirements.txt