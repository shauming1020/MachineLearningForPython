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
        Form.resize(694, 490)
        self.imgBt = QtWidgets.QPushButton(Form)
        self.imgBt.setGeometry(QtCore.QRect(30, 380, 111, 41))
        self.imgBt.setObjectName("imgBt")
        self.imgName = QtWidgets.QLineEdit(Form)
        self.imgName.setGeometry(QtCore.QRect(30, 440, 113, 20))
        self.imgName.setObjectName("imgName")
        self.dcText = QtWidgets.QTextEdit(Form)
        self.dcText.setGeometry(QtCore.QRect(470, 70, 161, 351))
        self.dcText.setObjectName("dcText")
        self.gtBt = QtWidgets.QPushButton(Form)
        self.gtBt.setGeometry(QtCore.QRect(170, 380, 111, 41))
        self.gtBt.setObjectName("gtBt")
        self.runBt = QtWidgets.QPushButton(Form)
        self.runBt.setGeometry(QtCore.QRect(310, 380, 111, 41))
        self.runBt.setObjectName("runBt")
        self.maskName = QtWidgets.QLineEdit(Form)
        self.maskName.setGeometry(QtCore.QRect(170, 440, 113, 20))
        self.maskName.setObjectName("maskName")
        self.threshold = QtWidgets.QLineEdit(Form)
        self.threshold.setGeometry(QtCore.QRect(310, 440, 113, 20))
        self.threshold.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.threshold.setObjectName("threshold")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.imgBt.setText(_translate("Form", "Select Image"))
        self.gtBt.setText(_translate("Form", "Select Model"))
        self.runBt.setText(_translate("Form", "RUN"))
        self.threshold.setText(_translate("Form", "0.9"))

