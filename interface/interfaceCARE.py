# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interfaceCARE.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


from PyQt5.QtCore import QRunnable, pyqtSlot,QThreadPool

import source_rc
import sys,os
import napari
from tifffile import imread
import numpy
from functools import partial
import time


import napari

sys.path.append('..')
from training import load,train
from prediction import predict, plot_results

class WorkerTr(QtCore.QRunnable):
    '''
    Worker thread
    '''

    def __init__(self, val_split, tr_steps,nb_epochs,path_data,model_path,patch_size):
        super(WorkerTr,self).__init__()

        self.val_split = val_split
        self.tr_steps = tr_steps
        self.nb_epochs = nb_epochs
        self.path_data = path_data 
        self.model_path = model_path
        self.patch_size = patch_size


        
    @QtCore.pyqtSlot()
    def run(self):
        '''
        Your code goes in this function
        '''
        print("Thread training started") 
        axes='XYZ'
        (X,Y), (X_val,Y_val) = load(self.path_data,axes,self.val_split,self.patch_size)
        history = train(X,Y,X_val,Y_val,axes ,self.tr_steps,self.nb_epochs,model_name='my_model')
        #self.lineEdit_ModPath.setText( path_data+model_name )
        self.modelPath.setText( path_data+model_name )


        print("Thread training completed")

class WorkerPred(QtCore.QRunnable):
    '''
    Worker thread
    '''

    def __init__(self, path_data, show_res, stack_nb, filter_data):
        super(WorkerPred,self).__init__()

        self.path_data = path_data 
        self.show_res = show_res
        self.stack_nb = stack_nb
        self.filter_data = filter_data

        
    @QtCore.pyqtSlot()
    def run(self):
        '''
        Your code goes in this function
        '''
        plot = False
        print("Thread prediction started") 
        axes='XYZ'
        n_tiles = (1,4,4)
        name_model = 'my_model_steps_ep_200'
        if self.show_res.isChecked():
            plot = True
        predict(self.path_data, name_model, n_tiles, axes, plot, self.stack_nb, self.filter_data)

        print("Thread prediction completed")



class Ui_Window(object):
    def setupUi(self, Window):
        Window.setObjectName("Window")
        Window.resize(772, 784)
        Window.setWindowOpacity(1.0)
        Window.setAutoFillBackground(False)
        Window.setStyleSheet("background-color: rgb(217, 214, 219);\n"
"\n"
"")
        Window.setModal(False)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(Window)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_Tr = QtWidgets.QLabel(Window)
        self.label_Tr.setStyleSheet("background-color: rgb(136, 135, 165);")
        self.label_Tr.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_Tr.setTextFormat(QtCore.Qt.PlainText)
        self.label_Tr.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Tr.setObjectName("label_Tr")
        self.verticalLayout.addWidget(self.label_Tr)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEdit_TrPath = QtWidgets.QLineEdit(Window)
        self.lineEdit_TrPath.setStyleSheet("color: rgb(255, 255, 255);")
        self.lineEdit_TrPath.setObjectName("lineEdit_TrPath")
        self.horizontalLayout_3.addWidget(self.lineEdit_TrPath)
        self.toolButton_TrPath = QtWidgets.QToolButton(Window)
        self.toolButton_TrPath.setObjectName("toolButton_TrPath")
        self.horizontalLayout_3.addWidget(self.toolButton_TrPath)
        self.comboBoxTr = QtWidgets.QComboBox(Window)
        self.comboBoxTr.setMaxVisibleItems(2)
        self.comboBoxTr.setObjectName("comboBoxTr")
        self.comboBoxTr.addItem("Clean")
        self.comboBoxTr.addItem("Noisy")
        self.horizontalLayout_3.addWidget(self.comboBoxTr)
        self.pushButton_TrPreview = QtWidgets.QPushButton(Window)
        self.pushButton_TrPreview.setObjectName("pushButton_TrPreview")
        self.horizontalLayout_3.addWidget(self.pushButton_TrPreview)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.label_Param = QtWidgets.QLabel(Window)
        self.label_Param.setObjectName("label_Param")
        self.verticalLayout.addWidget(self.label_Param)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_ValSplit = QtWidgets.QLabel(Window)
        self.label_ValSplit.setObjectName("label_ValSplit")
        self.horizontalLayout_7.addWidget(self.label_ValSplit)
        self.doubleSpinBoxTr = QtWidgets.QDoubleSpinBox(Window)
        self.doubleSpinBoxTr.setMaximum(1.0)
        self.doubleSpinBoxTr.setSingleStep(0.1)
        self.doubleSpinBoxTr.setObjectName("doubleSpinBoxTr")
        self.horizontalLayout_7.addWidget(self.doubleSpinBoxTr)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_TrSteps = QtWidgets.QLabel(Window)
        self.label_TrSteps.setObjectName("label_TrSteps")
        self.horizontalLayout_8.addWidget(self.label_TrSteps)
        self.spinBox_TrSteps = QtWidgets.QSpinBox(Window)
        self.spinBox_TrSteps.setMinimum(1)
        self.spinBox_TrSteps.setMaximum(100000)
        self.spinBox_TrSteps.setSingleStep(100)
        self.spinBox_TrSteps.setObjectName("spinBox_TrSteps")
        self.horizontalLayout_8.addWidget(self.spinBox_TrSteps)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_NbEpochs = QtWidgets.QLabel(Window)
        self.label_NbEpochs.setObjectName("label_NbEpochs")
        self.horizontalLayout_9.addWidget(self.label_NbEpochs)
        self.spinBox_NbEpochs = QtWidgets.QSpinBox(Window)
        self.spinBox_NbEpochs.setMaximum(1000)
        self.spinBox_NbEpochs.setObjectName("spinBox_NbEpochs")
        self.horizontalLayout_9.addWidget(self.spinBox_NbEpochs)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_PatchSize = QtWidgets.QLabel(Window)
        self.label_PatchSize.setObjectName("label_PatchSize")
        self.horizontalLayout_14.addWidget(self.label_PatchSize)
        self.lineEdit_PatchSize = QtWidgets.QLineEdit(Window)
        self.lineEdit_PatchSize.setObjectName("lineEdit_PatchSize")
        self.horizontalLayout_14.addWidget(self.lineEdit_PatchSize)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem3)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem4)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem5)
        self.verticalLayout.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem6)
        self.pushButtonTr = QtWidgets.QPushButton(Window)
        self.pushButtonTr.setObjectName("pushButtonTr")
        self.horizontalLayout_5.addWidget(self.pushButtonTr)
        self.progressBar_Tr = QtWidgets.QProgressBar(Window)
        self.progressBar_Tr.setProperty("value", 24)
        self.progressBar_Tr.setObjectName("progressBar_Tr")
        self.horizontalLayout_5.addWidget(self.progressBar_Tr)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem7)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.label_Pred = QtWidgets.QLabel(Window)
        self.label_Pred.setStyleSheet("background-color: rgb(136, 135, 165);")
        self.label_Pred.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_Pred.setTextFormat(QtCore.Qt.PlainText)
        self.label_Pred.setScaledContents(False)
        self.label_Pred.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Pred.setObjectName("label_Pred")
        self.verticalLayout.addWidget(self.label_Pred)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lineEdit_PredPath = QtWidgets.QLineEdit(Window)
        self.lineEdit_PredPath.setStyleSheet("color: rgb(255, 255, 255);")
        self.lineEdit_PredPath.setObjectName("lineEdit_PredPath")
        self.horizontalLayout_2.addWidget(self.lineEdit_PredPath)
        self.toolButtonPredPath = QtWidgets.QToolButton(Window)
        self.toolButtonPredPath.setObjectName("toolButtonPredPath")
        self.horizontalLayout_2.addWidget(self.toolButtonPredPath)
        self.comboBoxSelect = QtWidgets.QComboBox(Window)
        self.comboBoxSelect.setObjectName("comboBoxSelect")
        self.horizontalLayout_2.addWidget(self.comboBoxSelect)
        self.comboBoxSelect.addItem("all")
        self.comboBoxSelect.addItem("ch0")
        self.comboBoxSelect.addItem("ch1")
        self.lineEditStack = QtWidgets.QLineEdit(Window)
        self.lineEditStack.setStyleSheet("color: rgb(255, 255, 255);")
        self.lineEditStack.setObjectName("lineEditStack")
        self.horizontalLayout_2.addWidget(self.lineEditStack)
        self.lineEditStack.setFixedWidth(65)
        self.pushButton_PredPreview = QtWidgets.QPushButton(Window)
        self.pushButton_PredPreview.setObjectName("pushButton_PredPreview")
        self.horizontalLayout_2.addWidget(self.pushButton_PredPreview)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.lineEdit_PredictedPath = QtWidgets.QLineEdit(Window)
        self.lineEdit_PredictedPath.setStyleSheet("color: rgb(255, 255, 255);")
        self.lineEdit_PredictedPath.setObjectName("lineEdit_PredictedPath")
        self.horizontalLayout_13.addWidget(self.lineEdit_PredictedPath)
        self.toolButton_PredictedPath = QtWidgets.QToolButton(Window)
        self.toolButton_PredictedPath.setObjectName("toolButton_PredictedPath")
        self.horizontalLayout_13.addWidget(self.toolButton_PredictedPath)
        self.verticalLayout.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.checkBox_ShowRes = QtWidgets.QCheckBox(Window)
        self.checkBox_ShowRes.setObjectName("checkBox_ShowRes")
        self.horizontalLayout_15.addWidget(self.checkBox_ShowRes)
        self.verticalLayout.addLayout(self.horizontalLayout_15)
        self.label_Model = QtWidgets.QLabel(Window)
        self.label_Model.setObjectName("label_Model")
        self.verticalLayout.addWidget(self.label_Model)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit_ModPath = QtWidgets.QLineEdit(Window)
        self.lineEdit_ModPath.setStyleSheet("color: rgb(255, 255, 255);")
        self.lineEdit_ModPath.setObjectName("lineEdit_ModPath")
        self.horizontalLayout.addWidget(self.lineEdit_ModPath)
        self.toolButton_ModPath = QtWidgets.QToolButton(Window)
        self.toolButton_ModPath.setObjectName("toolButton_ModPath")
        self.horizontalLayout.addWidget(self.toolButton_ModPath)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem9)
        self.pushButton_Pred = QtWidgets.QPushButton(Window)
        self.pushButton_Pred.setObjectName("pushButton_Pred")
        self.horizontalLayout_4.addWidget(self.pushButton_Pred)
        self.progressBar_Pred = QtWidgets.QProgressBar(Window)
        self.progressBar_Pred.setProperty("value", 24)
        self.progressBar_Pred.setObjectName("progressBar_Pred")
        self.horizontalLayout_4.addWidget(self.progressBar_Pred)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem10)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.verticalLayout.addLayout(self.horizontalLayout_11)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.verticalLayout.addLayout(self.horizontalLayout_12)
        self.label_Image = QtWidgets.QLabel(Window)
        self.label_Image.setStyleSheet("image: url(:/newPrefix/wehi_logo.png);")
        self.label_Image.setText("")
        self.label_Image.setObjectName("label_Image")
        self.verticalLayout.addWidget(self.label_Image)
        self.horizontalLayout_6.addLayout(self.verticalLayout)

        
        self.retranslateUi(Window)
        self.comboBoxTr.setCurrentIndex(-1)

        QtCore.QMetaObject.connectSlotsByName(Window)

        self.threadpool = QtCore.QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

    def retranslateUi(self, Window):
        _translate = QtCore.QCoreApplication.translate
        Window.setWindowTitle(_translate("Window", "CARE denoising"))
        self.label_Tr.setText(_translate("Window", "TRAINING"))
        self.lineEdit_TrPath.setText(_translate("Window", "Enter training data path"))
        self.toolButton_TrPath.setText(_translate("Window", "..."))
        self.pushButton_TrPreview.setText(_translate("Window", "Preview"))
        self.label_Param.setText(_translate("Window", "Parameters (optional)"))
        self.label_ValSplit.setText(_translate("Window", "Validation split "))
        self.label_TrSteps.setText(_translate("Window", "Training steps"))
        self.label_NbEpochs.setText(_translate("Window", "Nb epochs"))
        self.label_PatchSize.setText(_translate("Window", "Patch size "))
        self.pushButtonTr.setText(_translate("Window", "TRAIN"))
        self.label_Pred.setText(_translate("Window", "PREDICTION"))
        self.lineEdit_PredPath.setText(_translate("Window", "Enter prediction data path"))
        self.toolButtonPredPath.setText(_translate("Window", "..."))
        self.lineEditStack.setText(_translate("Window", "Stack nb"))
        self.pushButton_PredPreview.setText(_translate("Window", "Preview"))
        self.lineEdit_PredictedPath.setText(_translate("Window", "Enter path to save predicted data"))
        self.toolButton_PredictedPath.setText(_translate("Window", "..."))
        self.checkBox_ShowRes.setText(_translate("Window", "Show results"))
        self.label_Model.setText(_translate("Window", "Model"))
        self.lineEdit_ModPath.setText(_translate("Window", "Enter model path"))
        self.toolButton_ModPath.setText(_translate("Window", "..."))
        self.pushButton_Pred.setText(_translate("Window", "PREDICT"))

        #Connect buttons
        self.toolButton_TrPath.clicked.connect(partial(self.browseSlot,self.lineEdit_TrPath))
        self.pushButtonTr.clicked.connect(self.train)
        self.pushButton_TrPreview.clicked.connect(partial(self.preview, self.lineEdit_TrPath))
        self.toolButtonPredPath.clicked.connect(partial(self.browseSlot, self.lineEdit_PredPath))
        self.toolButton_PredictedPath.clicked.connect(partial(self.browseSlot, self.lineEdit_PredictedPath))
        self.pushButton_PredPreview.clicked.connect(partial(self.preview, self.lineEdit_PredPath))
        self.pushButton_Pred.clicked.connect(self.predict)
        self.toolButton_ModPath.clicked.connect(partial(self.browseSlot, self.lineEdit_ModPath))
        #self.pushButtonTr.clicked.connect(self.train_fake)
        #self.pushButton_Pred.clicked.connect(self.predict_fake)


    def debugPrint( self, msg ):
            '''Print the message in the text edit at the bottom of the
            horizontal splitter.
            '''
            print( msg )

    def browseSlot( self, lineEdit ):
            ''' Called when the user presses the Browse button
            '''
            #self.debugPrint( "Browse button pressed" )
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            fileName = QtWidgets.QFileDialog.getExistingDirectory(None,"Select directory")
            if fileName:
                self.debugPrint( "setting file name: " + fileName )
                lineEdit.setText( fileName )

    def train(self):
        workerTr = WorkerTr(self.doubleSpinBoxTr.value(),self.spinBox_TrSteps.value(),self.spinBox_NbEpochs.value(),self.lineEdit_TrPath.text(),self.lineEdit_ModPath)
        self.threadpool.start(workerTr)
        # val_split = self.doubleSpinBoxTr.value()
        # tr_steps = self.spinBox_TrSteps.value()
        # nb_epochs = self.spinBox_NbEpochs.value()
        # axes = 'XYZ'
        # path_data = self.lineEdit_TrPath.text()

        # (X,Y), (X_val,Y_val) = load(path_data,axes,val_split)
        # history = train(X,Y,X_val,Y_val,axes,tr_steps,nb_epochs,model_name='my_model')
        # self.lineEdit_ModPath.setText( path_data+model_name )

    def preview(self,lineEdit):
        folder = self.comboBoxTr.currentText()
        if lineEdit.text().split('/')[-1]=='to_predict':
            path_data = lineEdit.text() + '/'
        else:
            path_data = lineEdit.text() + '/' + folder + '/'

        #im = imread(path_data + [f for f in os.listdir(path_data) if f.endswith('.tif')][0])
        #imarray=numpy.array(im)
        with napari.gui_qt():
            viewer = napari.Viewer()
            for f in os.listdir(path_data): 
                if f.endswith('.tif'):
                    im = imread(path_data + f)
                    imarray=numpy.array(im)
                    viewer.add_image(imarray,name = f)


    def predict(self):
        workerPred = WorkerPred(self.lineEdit_PredPath.text(), self.checkBox_ShowRes, self.lineEditStack.text(),self.comboBoxSelect.currentText())
        self.threadpool.start(workerPred)
        # axes='XYZ'
        # path_data = self.lineEdit_PredPath.text()
        # n_tiles = (1,4,4)
        # name_model = 'my_model'
        # x, restored = predict(path_data, name_model, n_tiles, axes)
        # if self.checkBox_ShowRes.isChecked():
        #     plot_results(x,restored)




if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    Window = QtWidgets.QDialog()
    ui = Ui_Window()
    ui.setupUi(Window)
    Window.show()

    sys.exit(app.exec_())

