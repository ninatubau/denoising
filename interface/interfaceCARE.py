# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interfaceCARE.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import sys
import os
import napari
import numpy
import subprocess
from subprocess import PIPE

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QThread
from tifffile import imread
from functools import partial
import sys
sys.path.append('../../')
from denoising.src.training import load, train
from denoising.src.prediction import predict
import faulthandler

class WorkerTr(QThread):
    """
    Worker thread
    """

    def __init__(self, val_split, tr_steps, nb_epochs, path_data, model_name, patch_size):
        super(WorkerTr, self).__init__()

        self.val_split = val_split
        self.tr_steps = tr_steps
        self.nb_epochs = nb_epochs
        self.path_data = path_data
        self.model_name = model_name
        self.patch_size = patch_size

    def run(self):
        """
        Your code goes in this function
        """

        print("Thread training started")
        axes = 'XYZ'
        csv_file = 'loss.csv'
        kwargs = {'train_steps_per_epoch': self.tr_steps, 'train_epochs': self.nb_epochs}
        data_name = 'data_prepared'
        (X, Y), (X_val, Y_val) = load(self.path_data, axes, self.val_split, self.patch_size, data_name)
        train(X, Y, X_val, Y_val, axes, self.model_name, csv_file, True, self.val_split, self.patch_size, **kwargs)
        # self.lineEdit_ModPath.setText( path_data+model_name )






class WorkerPrediction(QThread):
    """
    Worker thread
    """

    def __init__(self, path_data, stack_nb, filter_data, model_name):
        super(WorkerPrediction, self).__init__()

        self.path_data = path_data
        self.stack_nb = stack_nb
        self.filter_data = filter_data
        self.model_name = model_name

    def run(self):
        """
        Your code goes in this function
        """
        plot = False
        axes = 'XYZ'
        n_tiles = (1, 4, 4)
        predict(self.path_data, self.model_name, n_tiles, axes, plot, self.stack_nb, self.filter_data)

class UiWindow(object):
    def __init__(self):
        self.label_Image = QtWidgets.QLabel(Window)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()

        self.pushButton_Pred = QtWidgets.QPushButton(Window)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.toolButton_ModPath = QtWidgets.QToolButton(Window)
        self.lineEdit_ModPath = QtWidgets.QLineEdit(Window)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.label_Model = QtWidgets.QLabel(Window)
        #self.checkBox_ShowRes = QtWidgets.QCheckBox(Window)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.toolButton_PredictedPath = QtWidgets.QToolButton(Window)
        self.lineEdit_PredictedPath = QtWidgets.QLineEdit(Window)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.pushButton_PredPreview = QtWidgets.QPushButton(Window)
        self.lineEditStack = QtWidgets.QLineEdit(Window)
        self.comboBoxSelect = QtWidgets.QComboBox(Window)
        self.toolButtonPredPath = QtWidgets.QToolButton(Window)
        self.lineEdit_PredPath = QtWidgets.QLineEdit(Window)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.label_Pred = QtWidgets.QLabel(Window)
        self.pushButtonTr = QtWidgets.QPushButton(Window)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.spinBox_z = QtWidgets.QSpinBox(Window)
        self.spinBox_y = QtWidgets.QSpinBox(Window)
        self.spinBox_x = QtWidgets.QSpinBox(Window)

        self.label_PatchSize = QtWidgets.QLabel(Window)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.spinBox_NbEpochs = QtWidgets.QSpinBox(Window)
        self.label_NbEpochs = QtWidgets.QLabel(Window)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.spinBox_TrSteps = QtWidgets.QSpinBox(Window)
        self.label_TrSteps = QtWidgets.QLabel(Window)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.doubleSpinBoxTr = QtWidgets.QDoubleSpinBox(Window)
        self.label_ValSplit = QtWidgets.QLabel(Window)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.label_Param = QtWidgets.QLabel(Window)
        self.pushButton_TrPreview = QtWidgets.QPushButton(Window)
        self.comboBoxTr = QtWidgets.QComboBox(Window)
        self.toolButton_TrPath = QtWidgets.QToolButton(Window)
        self.lineEdit_TrPath = QtWidgets.QLineEdit(Window)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.label_Tr = QtWidgets.QLabel(Window)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(Window)

        #Initialise with training default values
        self.spinBox_x.setValue(16)
        self.spinBox_y.setValue(64)
        self.spinBox_z.setValue(64)
        self.doubleSpinBoxTr.setValue(0.2)
        self.spinBox_TrSteps.setValue(100)
        self.spinBox_NbEpochs.setValue(100)

        #check if running on docker or local
        bash_cmd = '''ls -ali / | sed '2!d' |awk {'print $1'}'''
        results = subprocess.run(bash_cmd, shell=True, universal_newlines=True, check=True, stdout=PIPE)
        value_location = int(results.stdout.splitlines()[0])

        if value_location != 2:
            print('Disabling <Preview> button because running in docker')
            self.pushButton_TrPreview.setDisabled(True)
            self.pushButton_PredPreview.setDisabled(True)





    def setup_ui(self, window):
        window.setObjectName("Window")
        window.resize(772, 784)
        window.setWindowOpacity(1.0)
        window.setAutoFillBackground(False)
        window.setStyleSheet("background-color: rgb(217, 214, 219);\n"
                             "\n"
                             "")
        window.setModal(False)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_Tr.setStyleSheet("background-color: rgb(136, 135, 165);")
        self.label_Tr.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_Tr.setTextFormat(QtCore.Qt.PlainText)
        self.label_Tr.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Tr.setObjectName("label_Tr")
        self.verticalLayout.addWidget(self.label_Tr)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.lineEdit_TrPath.setStyleSheet("color: rgb(255, 255, 255);")
        self.lineEdit_TrPath.setObjectName("lineEdit_TrPath")
        self.horizontalLayout_3.addWidget(self.lineEdit_TrPath)
        self.toolButton_TrPath.setObjectName("toolButton_TrPath")
        self.horizontalLayout_3.addWidget(self.toolButton_TrPath)
        self.comboBoxTr.setMaxVisibleItems(2)
        self.comboBoxTr.setObjectName("comboBoxTr")
        self.comboBoxTr.addItem("clean")
        self.comboBoxTr.addItem("noisy")
        self.comboBoxTr.addItem("both")
        self.horizontalLayout_3.addWidget(self.comboBoxTr)
        self.pushButton_TrPreview.setObjectName("pushButton_TrPreview")
        self.horizontalLayout_3.addWidget(self.pushButton_TrPreview)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.label_Param.setObjectName("label_Param")
        self.verticalLayout.addWidget(self.label_Param)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_ValSplit.setObjectName("label_ValSplit")
        self.horizontalLayout_7.addWidget(self.label_ValSplit)
        self.doubleSpinBoxTr.setMaximum(1.0)
        self.doubleSpinBoxTr.setSingleStep(0.1)
        self.doubleSpinBoxTr.setObjectName("doubleSpinBoxTr")
        self.horizontalLayout_7.addWidget(self.doubleSpinBoxTr)
        spacer_item = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacer_item)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_TrSteps.setObjectName("label_TrSteps")
        self.horizontalLayout_8.addWidget(self.label_TrSteps)
        self.spinBox_TrSteps.setMinimum(0)
        self.spinBox_TrSteps.setMaximum(100000)
        self.spinBox_TrSteps.setSingleStep(100)
        self.spinBox_TrSteps.setObjectName("spinBox_TrSteps")
        self.horizontalLayout_8.addWidget(self.spinBox_TrSteps)
        spacer_item1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacer_item1)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_NbEpochs.setObjectName("label_NbEpochs")
        self.horizontalLayout_9.addWidget(self.label_NbEpochs)
        self.spinBox_NbEpochs.setMinimum(0)
        self.spinBox_NbEpochs.setMaximum(1000)
        self.spinBox_NbEpochs.setObjectName("spinBox_NbEpochs")
        self.horizontalLayout_9.addWidget(self.spinBox_NbEpochs)
        spacer_item2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacer_item2)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_PatchSize.setObjectName("label_PatchSize")
        self.horizontalLayout_14.addWidget(self.label_PatchSize)
        self.spinBox_x.setObjectName("spinBox_x")
        self.horizontalLayout_14.addWidget(self.spinBox_x)
        self.spinBox_y.setObjectName("spinBox_y")
        self.horizontalLayout_14.addWidget(self.spinBox_y)
        self.spinBox_z.setObjectName("spinBox_z")
        self.horizontalLayout_14.addWidget(self.spinBox_z)
        spacer_item3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacer_item3)
        spacer_item4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacer_item4)
        spacer_item5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacer_item5)
        self.verticalLayout.addLayout(self.horizontalLayout_14)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_modelName = QtWidgets.QLabel(Window)
        self.label_modelName.setObjectName("label_modelName")
        self.horizontalLayout_10.addWidget(self.label_modelName)
        self.lineEdit_modelName = QtWidgets.QLineEdit(Window)
        self.lineEdit_modelName.setObjectName("lineEdit_modelName")
        self.horizontalLayout_10.addWidget(self.lineEdit_modelName)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem6)
        self.verticalLayout_2.addLayout(self.horizontalLayout_10)
        self.verticalLayout.addLayout(self.verticalLayout_2)


        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacer_item6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacer_item6)
        self.pushButtonTr.setObjectName("pushButtonTr")
        self.horizontalLayout_5.addWidget(self.pushButtonTr)
        spacer_item7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacer_item7)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.label_Pred.setStyleSheet("background-color: rgb(136, 135, 165);")
        self.label_Pred.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_Pred.setTextFormat(QtCore.Qt.PlainText)
        self.label_Pred.setScaledContents(False)
        self.label_Pred.setAlignment(QtCore.Qt.AlignCenter)
        self.label_Pred.setObjectName("label_Pred")
        self.verticalLayout.addWidget(self.label_Pred)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lineEdit_PredPath.setStyleSheet("color: rgb(255, 255, 255);")
        self.lineEdit_PredPath.setObjectName("lineEdit_PredPath")
        self.horizontalLayout_2.addWidget(self.lineEdit_PredPath)
        self.toolButtonPredPath.setObjectName("toolButtonPredPath")
        self.horizontalLayout_2.addWidget(self.toolButtonPredPath)
        self.comboBoxSelect.setObjectName("comboBoxSelect")
        self.horizontalLayout_2.addWidget(self.comboBoxSelect)
        self.comboBoxSelect.addItem("all")
        self.comboBoxSelect.addItem("ch0")
        self.comboBoxSelect.addItem("ch1")
        self.lineEditStack.setStyleSheet("color: rgb(255, 255, 255);")
        self.lineEditStack.setObjectName("lineEditStack")
        self.horizontalLayout_2.addWidget(self.lineEditStack)
        self.lineEditStack.setFixedWidth(65)
        self.pushButton_PredPreview.setObjectName("pushButton_PredPreview")
        self.horizontalLayout_2.addWidget(self.pushButton_PredPreview)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.lineEdit_PredictedPath.setStyleSheet("color: rgb(255, 255, 255);")
        self.lineEdit_PredictedPath.setObjectName("lineEdit_PredictedPath")
        self.horizontalLayout_13.addWidget(self.lineEdit_PredictedPath)
        self.toolButton_PredictedPath.setObjectName("toolButton_PredictedPath")
        self.horizontalLayout_13.addWidget(self.toolButton_PredictedPath)
        self.verticalLayout.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        #self.checkBox_ShowRes.setObjectName("checkBox_ShowRes")
        #self.horizontalLayout_15.addWidget(self.checkBox_ShowRes)
        self.verticalLayout.addLayout(self.horizontalLayout_15)
        self.label_Model.setObjectName("label_Model")
        self.verticalLayout.addWidget(self.label_Model)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit_ModPath.setStyleSheet("color: rgb(255, 255, 255);")
        self.lineEdit_ModPath.setObjectName("lineEdit_ModPath")
        self.horizontalLayout.addWidget(self.lineEdit_ModPath)
        self.toolButton_ModPath.setObjectName("toolButton_ModPath")
        self.horizontalLayout.addWidget(self.toolButton_ModPath)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacer_item9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacer_item9)
        self.pushButton_Pred.setObjectName("pushButton_Pred")
        self.horizontalLayout_4.addWidget(self.pushButton_Pred)
        spacer_item10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacer_item10)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.verticalLayout.addLayout(self.horizontalLayout_11)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.verticalLayout.addLayout(self.horizontalLayout_12)
        self.label_Image.setStyleSheet("image: url(:/newPrefix/wehi_logo.png);")
        self.label_Image.setText("")
        self.label_Image.setObjectName("label_Image")
        self.verticalLayout.addWidget(self.label_Image)
        self.horizontalLayout_6.addLayout(self.verticalLayout)
        self.retranslateUi(window)
        self.comboBoxTr.setCurrentIndex(-1)

        

        QtCore.QMetaObject.connectSlotsByName(window)

    def retranslateUi(self, window):
        _translate = QtCore.QCoreApplication.translate
        window.setWindowTitle(_translate("Window", "CARE denoising"))
        self.label_Tr.setText(_translate("Window", "TRAINING"))
        self.lineEdit_TrPath.setText(_translate("Window", "Enter training data path"))
        self.toolButton_TrPath.setText(_translate("Window", "..."))
        self.pushButton_TrPreview.setText(_translate("Window", "Preview"))
        self.label_Param.setText(_translate("Window", "Parameters (optional)"))
        self.label_ValSplit.setText(_translate("Window", "Validation split "))
        self.label_TrSteps.setText(_translate("Window", "Training steps"))
        self.label_NbEpochs.setText(_translate("Window", "Nb epochs"))
        self.label_PatchSize.setText(_translate("Window", "Patch size"))
        self.pushButtonTr.setText(_translate("Window", "TRAIN"))
        self.label_Pred.setText(_translate("Window", "PREDICTION"))
        self.lineEdit_PredPath.setText(_translate("Window", "Enter prediction data path"))
        self.toolButtonPredPath.setText(_translate("Window", "..."))
        self.lineEditStack.setText(_translate("Window", "Stack nb"))
        self.pushButton_PredPreview.setText(_translate("Window", "Preview"))
        self.lineEdit_PredictedPath.setText(_translate("Window", "Enter path to save predicted data"))
        self.toolButton_PredictedPath.setText(_translate("Window", "..."))
        #self.checkBox_ShowRes.setText(_translate("Window", "Show results"))
        self.label_Model.setText(_translate("Window", "Model"))
        self.lineEdit_ModPath.setText(_translate("Window", "Enter model path"))
        self.toolButton_ModPath.setText(_translate("Window", "..."))
        self.pushButton_Pred.setText(_translate("Window", "PREDICT"))

        self.label_modelName.setText(_translate("Window", "Model name"))
        #self.lineEdit_modelName.setText(_translate("Window", "my_model"))

        # Connect buttons
        self.toolButton_TrPath.clicked.connect(partial(self.browse_slot, self.lineEdit_TrPath))
        self.pushButtonTr.clicked.connect(self.train)
        self.pushButton_TrPreview.clicked.connect(partial(self.preview, self.lineEdit_TrPath))
        self.toolButtonPredPath.clicked.connect(partial(self.browse_slot, self.lineEdit_PredPath))
        self.toolButton_PredictedPath.clicked.connect(partial(self.browse_slot, self.lineEdit_PredictedPath))
        self.pushButton_PredPreview.clicked.connect(partial(self.preview, self.lineEdit_PredPath))
        self.pushButton_Pred.clicked.connect(self.prediction)
        self.toolButton_ModPath.clicked.connect(partial(self.browse_slot, self.lineEdit_ModPath))
        # self.pushButtonTr.clicked.connect(self.train_fake)
        # self.pushButton_Pred.clicked.connect(self.predict_fake)


    def debug_print(self, msg):
        """Print the message in the text edit at the bottom of the
            horizontal splitter. """
        print(msg)

    def browse_slot(self, lineEdit):
        """ Called when the user presses the Browse button
            """
        # self.debugPrint( "Browse button pressed" )
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_name = QtWidgets.QFileDialog.getExistingDirectory(None, "Select directory")
        if file_name:
            lineEdit.setText(file_name)

    def train(self):
        patch_size = (self.spinBox_x.value(), self.spinBox_y.value(), self.spinBox_z.value())

        self.workerTr = WorkerTr(self.doubleSpinBoxTr.value(), self.spinBox_TrSteps.value(),
                                 self.spinBox_NbEpochs.value(), self.lineEdit_TrPath.text(), self.lineEdit_modelName.text(),
                                 patch_size)
        self.workerTr.start()
        model_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.lineEdit_ModPath.setText(model_path+'/models/'+self.lineEdit_modelName.text())
        if self.workerTr.isFinished():
            self.workerTr.exit()

    def view_images(self, path_data):
        with napari.gui_qt():
            viewer = napari.Viewer()
            for f in os.listdir(path_data)[:5]:
                if f.endswith('.tif'):
                    im = imread(path_data + f)
                    im_array = numpy.array(im)
                    viewer.add_image(im_array, name=f + path_data.split()[-1])

    def preview(self, line_edit):
        folder = self.comboBoxTr.currentText()
        if line_edit.text().split('/')[-1] == 'to_predict':
            path_data = line_edit.text() + '/'
            self.view_images(path_data)
        elif folder == 'both':
            path_data = line_edit.text() + '/clean/'
            self.view_images(path_data)
            path_data = line_edit.text() + '/noisy/'
            self.view_images(path_data)

        else:
            path_data = line_edit.text() + '/' + folder + '/'
            self.view_images(path_data)

    def prediction(self):
        self.workerPred = WorkerPrediction(self.lineEdit_PredPath.text(),
                                           self.lineEditStack.text(),
                                           self.comboBoxSelect.currentText(), self.lineEdit_ModPath.text())
        self.workerPred.start()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    #app = QtCore.QCoreApplication(sys.argv)
    Window = QtWidgets.QDialog()
    faulthandler.enable()
    ui = UiWindow()
    ui.setup_ui(Window)
    Window.show()

    #grview.show()

    #grview = QGraphicsView()
    #rc = app.exec_()
    #del grview
    #del scene
    #del app
    #sys.exit()
    sys.exit(app.exec_())
