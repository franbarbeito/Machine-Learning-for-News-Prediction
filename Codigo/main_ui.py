# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 936)
        MainWindow.setMinimumSize(QtCore.QSize(1200, 900))
        MainWindow.setMaximumSize(QtCore.QSize(1200, 936))
        MainWindow.setBaseSize(QtCore.QSize(900, 800))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(1200, 900))
        self.centralwidget.setMaximumSize(QtCore.QSize(1200, 900))
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")
        self.tabTraining = QtWidgets.QWidget()
        self.tabTraining.setObjectName("tabTraining")
        self.label = QtWidgets.QLabel(self.tabTraining)
        self.label.setGeometry(QtCore.QRect(110, 30, 291, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setWordWrap(False)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tabTraining)
        self.label_2.setGeometry(QtCore.QRect(110, 80, 311, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setWordWrap(False)
        self.label_2.setObjectName("label_2")
        self.btnRutaOdio = QtWidgets.QPushButton(self.tabTraining)
        self.btnRutaOdio.setGeometry(QtCore.QRect(590, 20, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btnRutaOdio.setFont(font)
        self.btnRutaOdio.setObjectName("btnRutaOdio")
        self.btnRutaNoOdio = QtWidgets.QPushButton(self.tabTraining)
        self.btnRutaNoOdio.setGeometry(QtCore.QRect(590, 70, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btnRutaNoOdio.setFont(font)
        self.btnRutaNoOdio.setObjectName("btnRutaNoOdio")
        self.lblRutaOdio = QtWidgets.QLabel(self.tabTraining)
        self.lblRutaOdio.setGeometry(QtCore.QRect(730, 30, 481, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lblRutaOdio.setFont(font)
        self.lblRutaOdio.setScaledContents(False)
        self.lblRutaOdio.setObjectName("lblRutaOdio")
        self.lblRutaNoOdio = QtWidgets.QLabel(self.tabTraining)
        self.lblRutaNoOdio.setGeometry(QtCore.QRect(730, 80, 491, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lblRutaNoOdio.setFont(font)
        self.lblRutaNoOdio.setObjectName("lblRutaNoOdio")
        self.label_5 = QtWidgets.QLabel(self.tabTraining)
        self.label_5.setGeometry(QtCore.QRect(560, 150, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_5.setTextFormat(QtCore.Qt.PlainText)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTop|QtCore.Qt.AlignTrailing)
        self.label_5.setWordWrap(False)
        self.label_5.setObjectName("label_5")
        self.cbAlgoritmo = QtWidgets.QComboBox(self.tabTraining)
        self.cbAlgoritmo.setGeometry(QtCore.QRect(540, 190, 221, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cbAlgoritmo.setFont(font)
        self.cbAlgoritmo.setEditable(False)
        self.cbAlgoritmo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cbAlgoritmo.setMinimumContentsLength(0)
        self.cbAlgoritmo.setObjectName("cbAlgoritmo")
        self.cbAlgoritmo.addItem("")
        self.cbAlgoritmo.addItem("")
        self.cbAlgoritmo.addItem("")
        self.btnGenerarModelo = QtWidgets.QPushButton(self.tabTraining)
        self.btnGenerarModelo.setGeometry(QtCore.QRect(540, 240, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.btnGenerarModelo.setFont(font)
        self.btnGenerarModelo.setObjectName("btnGenerarModelo")
        self.groupBox_2 = QtWidgets.QGroupBox(self.tabTraining)
        self.groupBox_2.setGeometry(QtCore.QRect(0, 430, 1181, 431))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setGeometry(QtCore.QRect(630, 80, 111, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_6.setFont(font)
        self.label_6.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_6.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setGeometry(QtCore.QRect(369, 80, 101, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_7.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.lblPrecision = QtWidgets.QLabel(self.groupBox_2)
        self.lblPrecision.setGeometry(QtCore.QRect(766, 80, 51, 18))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lblPrecision.setFont(font)
        self.lblPrecision.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lblPrecision.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lblPrecision.setObjectName("lblPrecision")
        self.lblAcierto = QtWidgets.QLabel(self.groupBox_2)
        self.lblAcierto.setGeometry(QtCore.QRect(496, 80, 61, 18))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lblAcierto.setFont(font)
        self.lblAcierto.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lblAcierto.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lblAcierto.setObjectName("lblAcierto")
        self.btnGuardarModelo = QtWidgets.QPushButton(self.groupBox_2)
        self.btnGuardarModelo.setGeometry(QtCore.QRect(510, 330, 281, 41))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.btnGuardarModelo.setFont(font)
        self.btnGuardarModelo.setObjectName("btnGuardarModelo")
        self.lblAlgoritmo = QtWidgets.QLabel(self.groupBox_2)
        self.lblAlgoritmo.setGeometry(QtCore.QRect(10, 254, 16, 18))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.lblAlgoritmo.setFont(font)
        self.lblAlgoritmo.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lblAlgoritmo.setText("")
        self.lblAlgoritmo.setAlignment(QtCore.Qt.AlignCenter)
        self.lblAlgoritmo.setObjectName("lblAlgoritmo")
        self.tableEstadisticas = QtWidgets.QTableWidget(self.groupBox_2)
        self.tableEstadisticas.setGeometry(QtCore.QRect(366, 120, 561, 201))
        self.tableEstadisticas.setMaximumSize(QtCore.QSize(1147, 373))
        self.tableEstadisticas.setObjectName("tableEstadisticas")
        self.tableEstadisticas.setColumnCount(3)
        self.tableEstadisticas.setRowCount(3)
        item = QtWidgets.QTableWidgetItem()
        self.tableEstadisticas.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableEstadisticas.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableEstadisticas.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableEstadisticas.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableEstadisticas.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableEstadisticas.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableEstadisticas.setItem(2, 2, item)
        self.tableEstadisticas.horizontalHeader().setCascadingSectionResizes(True)
        self.tableEstadisticas.horizontalHeader().setDefaultSectionSize(135)
        self.progressBar = QtWidgets.QProgressBar(self.tabTraining)
        self.progressBar.setGeometry(QtCore.QRect(540, 330, 231, 31))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.tabWidget.addTab(self.tabTraining, "")
        self.tabTesting = QtWidgets.QWidget()
        self.tabTesting.setObjectName("tabTesting")
        self.label_3 = QtWidgets.QLabel(self.tabTesting)
        self.label_3.setGeometry(QtCore.QRect(200, 20, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setWordWrap(False)
        self.label_3.setObjectName("label_3")
        self.btnRutaModelo = QtWidgets.QPushButton(self.tabTesting)
        self.btnRutaModelo.setGeometry(QtCore.QRect(450, 20, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btnRutaModelo.setFont(font)
        self.btnRutaModelo.setObjectName("btnRutaModelo")
        self.btnRutaNoticias = QtWidgets.QPushButton(self.tabTesting)
        self.btnRutaNoticias.setGeometry(QtCore.QRect(790, 20, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btnRutaNoticias.setFont(font)
        self.btnRutaNoticias.setObjectName("btnRutaNoticias")
        self.label_4 = QtWidgets.QLabel(self.tabTesting)
        self.label_4.setGeometry(QtCore.QRect(540, 20, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setWordWrap(False)
        self.label_4.setObjectName("label_4")
        self.lblRutaNoticias = QtWidgets.QLabel(self.tabTesting)
        self.lblRutaNoticias.setGeometry(QtCore.QRect(700, 70, 291, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lblRutaNoticias.setFont(font)
        self.lblRutaNoticias.setObjectName("lblRutaNoticias")
        self.lblRutaModelo = QtWidgets.QLabel(self.tabTesting)
        self.lblRutaModelo.setGeometry(QtCore.QRect(360, 70, 291, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lblRutaModelo.setFont(font)
        self.lblRutaModelo.setObjectName("lblRutaModelo")
        self.groupBox = QtWidgets.QGroupBox(self.tabTesting)
        self.groupBox.setGeometry(QtCore.QRect(240, 490, 761, 371))
        self.groupBox.setAutoFillBackground(True)
        self.groupBox.setObjectName("groupBox")
        self.tableResultados = QtWidgets.QTableWidget(self.groupBox)
        self.tableResultados.setGeometry(QtCore.QRect(10, 20, 741, 221))
        self.tableResultados.setObjectName("tableResultados")
        self.tableResultados.setColumnCount(2)
        self.tableResultados.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableResultados.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableResultados.setHorizontalHeaderItem(1, item)
        self.tableResultados.horizontalHeader().setDefaultSectionSize(369)
        self.btnExportar = QtWidgets.QPushButton(self.groupBox)
        self.btnExportar.setGeometry(QtCore.QRect(290, 250, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.btnExportar.setFont(font)
        self.btnExportar.setObjectName("btnExportar")
        self.btnClasificar = QtWidgets.QPushButton(self.tabTesting)
        self.btnClasificar.setGeometry(QtCore.QRect(500, 440, 241, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.btnClasificar.setFont(font)
        self.btnClasificar.setObjectName("btnClasificar")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tabTesting)
        self.groupBox_3.setGeometry(QtCore.QRect(-10, 110, 1191, 311))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_8 = QtWidgets.QLabel(self.groupBox_3)
        self.label_8.setGeometry(QtCore.QRect(530, 40, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_8.setFont(font)
        self.label_8.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_8.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.groupBox_3)
        self.label_9.setGeometry(QtCore.QRect(930, 30, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_9.setFont(font)
        self.label_9.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_9.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.lblPrecisionTesting = QtWidgets.QLabel(self.groupBox_3)
        self.lblPrecisionTesting.setGeometry(QtCore.QRect(640, 40, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lblPrecisionTesting.setFont(font)
        self.lblPrecisionTesting.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lblPrecisionTesting.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lblPrecisionTesting.setObjectName("lblPrecisionTesting")
        self.lblAciertoTesting = QtWidgets.QLabel(self.groupBox_3)
        self.lblAciertoTesting.setGeometry(QtCore.QRect(1040, 30, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lblAciertoTesting.setFont(font)
        self.lblAciertoTesting.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lblAciertoTesting.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lblAciertoTesting.setObjectName("lblAciertoTesting")
        self.lblAlgoritmoTesting = QtWidgets.QLabel(self.groupBox_3)
        self.lblAlgoritmoTesting.setGeometry(QtCore.QRect(70, 30, 211, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lblAlgoritmoTesting.setFont(font)
        self.lblAlgoritmoTesting.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lblAlgoritmoTesting.setText("")
        self.lblAlgoritmoTesting.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lblAlgoritmoTesting.setObjectName("lblAlgoritmoTesting")
        self.label_10 = QtWidgets.QLabel(self.groupBox_3)
        self.label_10.setGeometry(QtCore.QRect(20, 40, 371, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_10.setFont(font)
        self.label_10.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_10.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.tableEstadisticasTesting = QtWidgets.QTableWidget(self.groupBox_3)
        self.tableEstadisticasTesting.setGeometry(QtCore.QRect(340, 80, 561, 211))
        self.tableEstadisticasTesting.setObjectName("tableEstadisticasTesting")
        self.tableEstadisticasTesting.setColumnCount(3)
        self.tableEstadisticasTesting.setRowCount(3)
        item = QtWidgets.QTableWidgetItem()
        self.tableEstadisticasTesting.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableEstadisticasTesting.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableEstadisticasTesting.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableEstadisticasTesting.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableEstadisticasTesting.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableEstadisticasTesting.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.tableEstadisticasTesting.setItem(1, 2, item)
        self.tableEstadisticasTesting.horizontalHeader().setDefaultSectionSize(135)
        self.tabWidget.addTab(self.tabTesting, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Clasificador de Noticias"))
        self.label.setText(_translate("MainWindow", "Noticias delito de odio"))
        self.label_2.setText(_translate("MainWindow", "Noticias de delito no odio"))
        self.btnRutaOdio.setText(_translate("MainWindow", "Abrir ruta"))
        self.btnRutaNoOdio.setText(_translate("MainWindow", "Abrir ruta"))
        self.lblRutaOdio.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Ruta: </span></p></body></html>"))
        self.lblRutaNoOdio.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt;\">Ruta: </span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "Algoritmo"))
        self.cbAlgoritmo.setCurrentText(_translate("MainWindow", "??rbol de decisi??n"))
        self.cbAlgoritmo.setItemText(0, _translate("MainWindow", "??rbol de decisi??n"))
        self.cbAlgoritmo.setItemText(1, _translate("MainWindow", "K-nn"))
        self.cbAlgoritmo.setItemText(2, _translate("MainWindow", "Naive Bayes"))
        self.btnGenerarModelo.setText(_translate("MainWindow", "Generar Modelo"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Estad??sticas"))
        self.label_6.setText(_translate("MainWindow", "Precisi??n"))
        self.label_7.setText(_translate("MainWindow", "Acierto"))
        self.lblPrecision.setText(_translate("MainWindow", "%"))
        self.lblAcierto.setText(_translate("MainWindow", "%"))
        self.btnGuardarModelo.setText(_translate("MainWindow", "Guardar Modelo"))
        item = self.tableEstadisticas.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "Pred. Odio"))
        item = self.tableEstadisticas.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "Pred. No Odio"))
        item = self.tableEstadisticas.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "Recall"))
        item = self.tableEstadisticas.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "true Odio"))
        item = self.tableEstadisticas.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "true No Odio"))
        item = self.tableEstadisticas.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Precision"))
        __sortingEnabled = self.tableEstadisticas.isSortingEnabled()
        self.tableEstadisticas.setSortingEnabled(False)
        self.tableEstadisticas.setSortingEnabled(__sortingEnabled)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabTraining), _translate("MainWindow", "Training"))
        self.label_3.setText(_translate("MainWindow", "Modelo "))
        self.btnRutaModelo.setText(_translate("MainWindow", "Abrir Modelo"))
        self.btnRutaNoticias.setText(_translate("MainWindow", "Abrir ruta"))
        self.label_4.setText(_translate("MainWindow", "Noticias"))
        self.lblRutaNoticias.setText(_translate("MainWindow", "Ruta: "))
        self.lblRutaModelo.setText(_translate("MainWindow", "Ruta: "))
        self.groupBox.setTitle(_translate("MainWindow", "Resultados"))
        item = self.tableResultados.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Noticia"))
        item = self.tableResultados.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Predicci??n"))
        self.btnExportar.setText(_translate("MainWindow", "Exportar"))
        self.btnClasificar.setText(_translate("MainWindow", "Clasificar Noticias"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Modelo"))
        self.label_8.setText(_translate("MainWindow", "Precisi??n"))
        self.label_9.setText(_translate("MainWindow", "Acierto"))
        self.lblPrecisionTesting.setText(_translate("MainWindow", "%"))
        self.lblAciertoTesting.setText(_translate("MainWindow", "%"))
        self.label_10.setText(_translate("MainWindow", "Algoritmo"))
        item = self.tableEstadisticasTesting.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "Pred. Odio"))
        item = self.tableEstadisticasTesting.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "Pred. No Odio"))
        item = self.tableEstadisticasTesting.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "Recall"))
        item = self.tableEstadisticasTesting.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "true Odio"))
        item = self.tableEstadisticasTesting.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "true No Odio"))
        item = self.tableEstadisticasTesting.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Precision"))
        __sortingEnabled = self.tableEstadisticasTesting.isSortingEnabled()
        self.tableEstadisticasTesting.setSortingEnabled(False)
        self.tableEstadisticasTesting.setSortingEnabled(__sortingEnabled)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabTesting), _translate("MainWindow", "Testing"))

