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
        Form.resize(877, 515)
        self.imgBt = QtWidgets.QPushButton(Form)
        self.imgBt.setGeometry(QtCore.QRect(30, 380, 111, 41))
        self.imgBt.setObjectName("imgBt")
        self.imgName = QtWidgets.QLineEdit(Form)
        self.imgName.setGeometry(QtCore.QRect(30, 440, 113, 20))
        self.imgName.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.imgName.setObjectName("imgName")
        self.dcText = QtWidgets.QTextEdit(Form)
        self.dcText.setGeometry(QtCore.QRect(610, 70, 161, 351))
        self.dcText.setObjectName("dcText")
        self.gtBt = QtWidgets.QPushButton(Form)
        self.gtBt.setGeometry(QtCore.QRect(170, 380, 111, 41))
        self.gtBt.setObjectName("gtBt")
        self.runBt = QtWidgets.QPushButton(Form)
        self.runBt.setGeometry(QtCore.QRect(450, 380, 111, 41))
        self.runBt.setObjectName("runBt")
        self.maskName = QtWidgets.QLineEdit(Form)
        self.maskName.setGeometry(QtCore.QRect(170, 440, 113, 20))
        self.maskName.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.maskName.setObjectName("maskName")
        self.threshold = QtWidgets.QLineEdit(Form)
        self.threshold.setGeometry(QtCore.QRect(450, 440, 113, 20))
        self.threshold.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.threshold.setObjectName("threshold")
        self.imgFileLabel = QtWidgets.QLabel(Form)
        self.imgFileLabel.setGeometry(QtCore.QRect(30, 470, 111, 21))
        self.imgFileLabel.setObjectName("imgFileLabel")
        self.gtFileLabel = QtWidgets.QLabel(Form)
        self.gtFileLabel.setGeometry(QtCore.QRect(170, 470, 131, 21))
        self.gtFileLabel.setObjectName("gtFileLabel")
        self.threLabel = QtWidgets.QLabel(Form)
        self.threLabel.setGeometry(QtCore.QRect(450, 470, 131, 18))
        self.threLabel.setObjectName("threLabel")
        self.mdBt = QtWidgets.QPushButton(Form)
        self.mdBt.setGeometry(QtCore.QRect(310, 380, 111, 41))
        self.mdBt.setObjectName("mdBt")
        self.gtFileLabel_2 = QtWidgets.QLabel(Form)
        self.gtFileLabel_2.setGeometry(QtCore.QRect(310, 470, 131, 21))
        self.gtFileLabel_2.setObjectName("gtFileLabel_2")
        self.modelName = QtWidgets.QLineEdit(Form)
        self.modelName.setGeometry(QtCore.QRect(310, 440, 113, 20))
        self.modelName.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.modelName.setObjectName("modelName")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "DIP_FinalProject"))
        self.imgBt.setText(_translate("Form", "Select Image"))
        self.imgName.setText(_translate("Form", "img.png"))
        self.gtBt.setText(_translate("Form", "Select GT"))
        self.runBt.setText(_translate("Form", "RUN"))
        self.maskName.setText(_translate("Form", "mask.png"))
        self.threshold.setText(_translate("Form", "0.9"))
        self.imgFileLabel.setText(_translate("Form", "Img Filename"))
        self.gtFileLabel.setText(_translate("Form", "GT Filename"))
        self.threLabel.setText(_translate("Form", "Output Threshold"))
        self.mdBt.setText(_translate("Form", "Select Model"))
        self.gtFileLabel_2.setText(_translate("Form", "Model Filename"))
        self.modelName.setText(_translate("Form", "BEST.pth"))

