import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from PyQt5 import QtCore,QtWidgets
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import * 
from PyQt5.QtCore import Qt


from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# reading the data using pandas
data = pd.read_csv("diabetes-dataset.csv")


# Copy the data into df
df = data.copy(deep = True)

# Replacing zero(0) to NaN 
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Filling all NaN to mean values
df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)

df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)

df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace = True)

df['Insulin'].fillna(df['Insulin'].mean(), inplace = True)

df['BMI'].fillna(df['BMI'].mean(), inplace = True)

# Split independent(x) and dependent(y) data
x=df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]]
y=df["Outcome"]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# Training the data
KNC = KNeighborsClassifier(n_neighbors = 1)
KNC.fit(x_train,y_train)

# Predicting
y_pred_KNC=KNC.predict(x_test)
print("Test set Accuracy: ",accuracy_score(y_test, y_pred_KNC))

def prediction(g,b,s,i,bmi):
    y_pred_KNC=KNC.predict([[g,b,s,i,bmi]])
    print(y_pred_KNC)
    return y_pred_KNC


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1051, 685)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(230, 0, 501, 131))
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setIndent(5)
        self.label.setObjectName("label")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(10, 80, 281, 131))
        self.label_8.setObjectName("label_8")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(230, 570, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(420, 570, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(200, 520, 341, 31))
        self.label_9.setObjectName("label_9")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(180, 180, 411, 321))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 1, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout.addWidget(self.lineEdit_4, 3, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 2, 1, 1, 1)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout.addWidget(self.lineEdit_5, 4, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout.addWidget(self.lineEdit_6, 5, 1, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 5, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_19.setObjectName("label_19")
        self.gridLayout.addWidget(self.label_19, 1, 2, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_20.setObjectName("label_20")
        self.gridLayout.addWidget(self.label_20, 2, 2, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_21.setObjectName("label_21")
        self.gridLayout.addWidget(self.label_21, 3, 2, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_22.setObjectName("label_22")
        self.gridLayout.addWidget(self.label_22, 4, 2, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_23.setObjectName("label_23")
        self.gridLayout.addWidget(self.label_23, 5, 2, 1, 1)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(729, 180, 281, 311))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_11 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_2.addWidget(self.label_11)
        self.label_12 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_2.addWidget(self.label_12)
        self.label_13 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_13.setObjectName("label_13")
        self.verticalLayout_2.addWidget(self.label_13)
        self.label_14 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_14.setObjectName("label_14")
        self.verticalLayout_2.addWidget(self.label_14)
        self.label_15 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_15.setObjectName("label_15")
        self.verticalLayout_2.addWidget(self.label_15)
        self.label_16 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_16.setObjectName("label_16")
        self.verticalLayout_2.addWidget(self.label_16)
        self.label_10 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_10.setObjectName("label_10")
        self.verticalLayout_2.addWidget(self.label_10)
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(740, 120, 261, 51))
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(730, 500, 261, 141))
        self.label_18.setObjectName("label_18")
        self.label.raise_()
        self.label_8.raise_()
        self.pushButton.raise_()
        self.pushButton_2.raise_()
        self.label_9.raise_()
        self.label_2.raise_()
        self.gridLayoutWidget.raise_()
        self.verticalLayoutWidget.raise_()
        self.label_17.raise_()
        self.label_18.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1051, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.pushButton.clicked.connect(self.clickMethod)
        
        self.pushButton_2.clicked.connect(self.clear)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Diabetes Predition"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; font-weight:600;\">DIABETES PREDICTION<br/>USING MACHINE LEARNING</span></p></body></html>"))
        self.label_8.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:16pt; font-weight:600;\">Patient\'s Details</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "SUBMIT"))
        self.pushButton_2.setText(_translate("MainWindow", "CLEAR"))
        self.label_9.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">Glucose:</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">Blood Pressure:</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">Insulin:</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">Skin Thinckness:</span></p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">Body Mass Index:</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">Name:</span></p></body></html>"))
        self.label_19.setText(_translate("MainWindow", "(70-180mg/d)"))
        self.label_20.setText(_translate("MainWindow", "(10-140mm Hg)"))
        self.label_21.setText(_translate("MainWindow", "(25-50mm)"))
        self.label_22.setText(_translate("MainWindow", "(15-276mu U/ml)"))
        self.label_23.setText(_translate("MainWindow", "(10-50)"))
        self.label_11.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Patient\'s name:</span></p></body></html>"))
        self.label_12.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Glucose:</span></p></body></html>"))
        self.label_13.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Blood Pressure:</span></p></body></html>"))
        self.label_14.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Skin Thinckness:</span></p></body></html>"))
        self.label_15.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Insulin:</span></p></body></html>"))
        self.label_16.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Body Mass Index:</span></p></body></html>"))
        self.label_10.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Diabetes:</span></p></body></html>"))
        self.label_17.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:600;\">Reports</span></p></body></html>"))
        self.label_18.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:7pt;\">The model uses KNN Classifier.</span></p><p><span style=\" font-size:7pt;\">Accuracy of model: 99%</span></p><p><span style=\" font-size:7pt;\">Develop By: Binayak Koiralas</span></p></body></html>"))
    
    def clickMethod(self):
        name=self.lineEdit.text()
        Glucose=self.lineEdit_2.text()
        BloodPressure=self.lineEdit_3.text()
        SkinThickness=self.lineEdit_4.text()
        Insulin=self.lineEdit_5.text()
        BMI=self.lineEdit_6.text()
        if(name=="" or Glucose=="" or BloodPressure=="" or SkinThickness=="" or Insulin=="" or BMI==""):
            self.label_9.setAlignment(Qt.AlignCenter) 
            self.label_9.setFont(QFont('Arial', 12)) 
            self.label_9.setStyleSheet("color: red")
            self.label_9.setText("Enter all the details")
            print(type(name))
            
        elif(Glucose.isdigit() and BloodPressure.isdigit() and SkinThickness.isdigit() and Insulin.isdigit() and BMI.isdigit()):
            self.label_9.setText("")
            g=int(Glucose)
            b=int(BloodPressure)
            s=int(SkinThickness)
            i=int(Insulin)
            bmi=int(BMI)
        
            p=prediction(g,b,s,i,bmi)
            print(type(p))
            
            self.label_11.setFont(QFont('Arial', 12))
            self.label_12.setFont(QFont('Arial', 12))
            self.label_13.setFont(QFont('Arial', 12))
            self.label_14.setFont(QFont('Arial', 12))
            self.label_15.setFont(QFont('Arial', 12))
            self.label_16.setFont(QFont('Arial', 12))
            
            self.label_11.setText("Patient's name: "+str(name))
            self.label_12.setText("Glucose: "+str(g))
            self.label_13.setText("Blood Pressure: "+str(b))
            self.label_14.setText("Skin Thickness: "+str(s))
            self.label_15.setText("Insulin: "+str(i))
            self.label_16.setText("Body Mass Index: "+str(bmi))
            
            if(p[0]==1):
                self.label_10.setFont(QFont('Arial', 12)) 
                self.label_10.setText("Diabetes: Positive")
            else:
                self.label_10.setFont(QFont('Arial', 12)) 
                self.label_10.setText("Diabetes: Negative")
            
            print(name,Glucose,BloodPressure,SkinThickness,Insulin,BMI)
        else:
            self.label_9.setFont(QFont('Arial', 12)) 
            self.label_9.setText("Enter the digit(int) from glucose to BMI")

        
    def clear(self):
        self.label_9.setText("")
        self.lineEdit.setText("")
        self.lineEdit_2.setText("")
        self.lineEdit_3.setText("")
        self.lineEdit_4.setText("")
        self.lineEdit_5.setText("")
        self.lineEdit_6.setText("")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

