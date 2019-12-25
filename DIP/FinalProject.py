# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FinalProject.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(453, 682)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(70, 50, 271, 101))
        self.groupBox.setObjectName("groupBox")
        self.preProcessing = QtWidgets.QPushButton(self.groupBox)
        self.preProcessing.setGeometry(QtCore.QRect(80, 30, 112, 34))
        self.preProcessing.setObjectName("preProcessing")
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setGeometry(QtCore.QRect(70, 190, 271, 211))
        self.groupBox_2.setObjectName("groupBox_2")
        self.preTrain = QtWidgets.QPushButton(self.groupBox_2)
        self.preTrain.setGeometry(QtCore.QRect(80, 30, 112, 34))
        self.preTrain.setObjectName("preTrain")
        self.training = QtWidgets.QPushButton(self.groupBox_2)
        self.training.setGeometry(QtCore.QRect(20, 90, 112, 34))
        self.training.setObjectName("training")
        self.pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton.setGeometry(QtCore.QRect(80, 150, 112, 34))
        self.pushButton.setObjectName("pushButton")
        self.crossTrain = QtWidgets.QPushButton(self.groupBox_2)
        self.crossTrain.setGeometry(QtCore.QRect(150, 90, 112, 34))
        self.crossTrain.setObjectName("crossTrain")
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        self.groupBox_3.setGeometry(QtCore.QRect(70, 440, 271, 171))
        self.groupBox_3.setObjectName("groupBox_3")
        self.loadTest = QtWidgets.QPushButton(self.groupBox_3)
        self.loadTest.setGeometry(QtCore.QRect(70, 20, 151, 34))
        self.loadTest.setObjectName("loadTest")
        self.predict = QtWidgets.QPushButton(self.groupBox_3)
        self.predict.setGeometry(QtCore.QRect(70, 120, 151, 34))
        self.predict.setObjectName("predict")
        self.loadModel = QtWidgets.QPushButton(self.groupBox_3)
        self.loadModel.setGeometry(QtCore.QRect(70, 70, 151, 34))
        self.loadModel.setObjectName("loadModel")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "Pre-Processing"))
        self.preProcessing.setText(_translate("Form", "Image Preprocessing"))
        self.groupBox_2.setTitle(_translate("Form", "Training"))
        self.preTrain.setText(_translate("Form", "Pre-Training"))
        self.training.setText(_translate("Form", "Training"))
        self.pushButton.setText(_translate("Form", "Evaluate"))
        self.crossTrain.setText(_translate("Form", "3fold cross validation"))
        self.groupBox_3.setTitle(_translate("Form", "Predict"))
        self.loadTest.setText(_translate("Form", "Load Test Image and Mask"))
        self.predict.setText(_translate("Form", "Predict"))
        self.loadModel.setText(_translate("Form", "Load Model"))

