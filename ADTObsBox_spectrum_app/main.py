from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QListWidget, QListWidgetItem, QDialog
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT, FigureManager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import os
import numpy as np
import pandas as pd
import sys
import datetime
import pytz
from pathlib import Path
import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import getpass
import logging
import nx2pd as nx
from nxcals.spark_session_builder import get_or_create, Flavor

import warnings
warnings.filterwarnings("ignore")

class MainWindow(QtWidgets.QMainWindow):
    def closeEvent(self, event):
        sys.exit()

    def __init__(self):
        super(MainWindow, self).__init__()

        # Load ui
        #uic.loadUi('form.ui', self)
        ui_file = os.path.join(os.path.dirname(__file__), "form.ui")
        uic.loadUi(ui_file, self)

        # Set WindowTitle
        self.setWindowTitle("LHC ADTObsBox spectrum app")
        
        # Menu options
        self.actionScreenshot.triggered.connect(self.take_screenshot)
        self.actionQuit.triggered.connect(self.closeEvent)
 
        # Load fill info
        self.button_loadinfo.clicked.connect(self.func_load_fill_info)

        # Load time range info
        self.button_loadinfo_2.clicked.connect(lambda: self.func_load_timerange_info(True))
        self.button_loadinfo_3.clicked.connect(lambda: self.func_load_timerange_info(False))

        self.button_fill_pressed = False


        # Load bunch info
        self.pushButton_bunch_info.clicked.connect(self.bunch_info)

        # nxcals variable names
        self.update_nxcals_beamplane()
        
        for i in range(0,4):
            list_item = self.list_beamplane.item(i) # Replace 0 with the index of the item you want to enable
            flags = list_item.flags()
            flags = flags | QtCore.Qt.ItemIsEnabled
            list_item.setFlags(flags)
            if i==0:
                list_item.setCheckState(QtCore.Qt.Checked)
            else:
                list_item.setCheckState(QtCore.Qt.Unchecked)

        self.list_beamplane.itemChanged.connect(self.update_nxcals_beamplane)
        self.default_nxcals.textChanged.connect(self.read_text)

        # default values for python and output path
        self.output_path.setPlainText(default_output_path)
        self.python_path.setPlainText(default_python_path)

        # Run button
        button_style = '''
            QPushButton {
                background-color: lightblue;
                border: 2px solid black;
                border-radius: 10px;
                color: black;
                font-weight: bold;
                font-size: 18px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: lightblue;
            }
            QPushButton:pressed {
                background-color: red;
                border: 2px solid #707070;
            }
        '''
        self.run_button.setStyleSheet(button_style)
        self.run_button.clicked.connect(self.on_button_clicked)
        self.run_button.setText("\U00002192 Run")

        # self fmin, fmax for spectrograms
        self.fmin = self.spin_fmin.value() 
        self.fmax = self.spin_fmax.value()
        self.update_fmin_fmax.clicked.connect(self.fmin_value_changed)

        self.tab_plot.setStyleSheet("background-color: white;")

        self.now_button.setFont(QFont("Arial", 12, QFont.Bold))
        
        # Set stylesheet to change button appearance
        button_style = '''
            QPushButton {
                background-color: blue;
                border: 2px solid #b1b1b1;
                border-radius: 10px;
                color: white;
                font-weight: bold;
                font-size: 10px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #b1b1b1;
                border: 2px solid #707070;
            }
        '''
        self.now_button.setStyleSheet(button_style)


        self.button_loadinfo_3.setFont(QFont("Arial", 12, QFont.Bold))
        self.button_loadinfo_3.setStyleSheet("background-color: green; color: white;")
        
        self.button_loadinfo_2.setFont(QFont("Arial", 12, QFont.Bold))
        self.button_loadinfo_2.setStyleSheet("background-color: orange; color: white;")

        self.button_loadinfo.setFont(QFont("Arial", 12, QFont.Bold))
        self.button_loadinfo.setStyleSheet("background-color: green; color: white;")
        
        self.final_output_path = self.output_path.toPlainText()


        # Initialize nxcals
        if True:
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)
            os.environ['PYSPARK_PYTHON'] = "./environment/bin/python"
            username = getpass.getuser()
            logging.info('Executing the kinit')
            os.system(f'kinit -f -r 5d -kt {os.path.expanduser("~")}/{getpass.getuser()}.keytab {getpass.getuser()}');
            logging.info('Creating the spark instance')
            #spark = get_or_create(flavor=Flavor.LOCAL)
            logging.info('Creating the spark instance')
            #spark = get_or_create(flavor=Flavor.LOCAL)
            #spark = get_or_create(flavor=Flavor.YARN_SMALL, master='yarn')
            spark = get_or_create(flavor=Flavor.LOCAL,
            conf=nxcals_conf)
            self.sk  = nx.SparkIt(spark)
            logging.info('Spark instance created.')
        
        #self.plot_beam_modes.stateChanged.connect(self.on_checkbox_state_changed)
        self.plot_beam_modes.toggled.connect(self.on_checkbox_state_changed)
        
        self.update_manual_2.clicked.connect(lambda: self.plot_manual_code(redraw=True))
        self.update_manual.clicked.connect(lambda: self.plot_manual_code(redraw=False))

    # Manually plot something in the spectrograms
    def plot_manual_code(self, redraw=False):

        for beamplane in self.all_figs.keys():
            if redraw==True:
                self.plot_beam_modes.setChecked(False)
                all_figs = self.plot_spectr(self.files_for_spectr, self.fmin, self.fmax)
                self.all_figs = all_figs
                self.show_spectr(all_figs[beamplane], beamplane)
            tab_name    = f"tab_{beamplane}"
            tab_widget = self.tab_plot.findChild(QWidget, tab_name)
            tab_layout = tab_widget.layout()
            canvas = tab_layout.itemAt(0).widget()
            #axes = canvas.figure.gca()
            axes = canvas.figure.get_axes()[0]

            if plt.gcf() != axes.get_figure():
                plt.figure(axes.get_figure().number)

            code = self.python_code.toPlainText()
            for line in code.splitlines():
                try:
                    exec(line)
                except Exception as e:
                    self.create_message_popup(f"Error running line {line}: {e}") 
            canvas.draw()

    # Plot beam modes
    def on_checkbox_state_changed(self, state):
        if not hasattr(self, 'beammodes_info'):
            self.create_message_popup(f"Error! You need to load fill info") 
            self.plot_beam_modes.setChecked(False)
            return

        if state:
            try:
                tol = pd.Timedelta(minutes=1)
                modes = self.beammodes_info[(self.beammodes_info['tstart'] >=self.final_t1_utc-tol) & (self.beammodes_info['tend'] <= self.final_t2_utc + tol) ]
            except:
                modes = self.beammodes_info[(self.beammodes_info['tstart'] >=self.final_t1_utc)  ]

            for beamplane in self.all_figs.keys():
                tab_name = f"tab_{beamplane}"
                tab_widget = self.tab_plot.findChild(QWidget, tab_name)
                tab_layout = tab_widget.layout()
                canvas = tab_layout.itemAt(0).widget()
                axes = canvas.figure.get_axes()[0]

                if plt.gcf() != axes.get_figure():
                    plt.figure(axes.get_figure().number)

                for i in range(len(modes)):
                    tt = modes.iloc[i]['tstart']
                    mode = modes.iloc[i]['mode']
                    plt.axhline(tt, c='k', lw=2)
                    xlim = plt.xlim()
                    xstart = xlim[0]
                    xend = xlim[1]
                    if i%2==0:
                        plt.text(xstart, tt, mode, c='k', ha='left')
                    else:
                        plt.text(xend, tt, mode, c='k', ha='right')

                canvas.draw()

        else:
            for beamplane in self.all_figs.keys():
                tab_name = f"tab_{beamplane}"
                tab_widget = self.tab_plot.findChild(QWidget, tab_name)
                tab_layout = tab_widget.layout()
                canvas = tab_layout.itemAt(0).widget()
                axes = canvas.figure.gca()
                for line in axes.lines:
                    if line.get_color() == 'k':
                        line.remove()
                for text in axes.texts:
                    if text.get_color() == 'k':
                        text.remove()   
                canvas.draw()

    # Get current time
    def on_now_button_clicked(self):
        now = QDateTime.currentDateTime()
        self.to_t2_2.setDateTime(now)

    # Change fmin-fmax ranges in spectrogram (spectrogram will be re-plotted not simply updated)
    def fmin_value_changed(self):
        self.fmin = self.spin_fmin.value()
        self.plot_beam_modes.setChecked(False)
        self.fmax = self.spin_fmax.value()
        all_figs = self.plot_spectr(self.files_for_spectr, self.fmin, self.fmax)

        self.all_figs = all_figs
        for beamplane in all_figs.keys():
            self.show_spectr(all_figs[beamplane], beamplane)
 
    # Plot spectrograms in tabs 
    def show_spectr(self, all_figs, beamplane):
            canvas = FigureCanvas(all_figs)
            toolbar = NavigationToolbar2QT(canvas, self)
            toolbar.setIconSize(QSize(16, 16))

            tab_name = f"tab_{beamplane}"
            tab_widget = self.tab_plot.findChild(QWidget, tab_name)
            tab_layout = tab_widget.layout()
            
            # remove the existing layout otherwise it will create may subplots. It gets the items from th tab layout and removes them. Then, add to the layout the new plots
            if tab_layout is not None:
                while tab_layout.count():
                    item = tab_layout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.setParent(None)
                tab_layout.addWidget(canvas)
                tab_layout.addWidget(toolbar)
            else:
                # layout for this tab does not exist, Create new layout and set
                tab_layout = QVBoxLayout()
                tab_layout.addWidget(canvas)
                tab_layout.addWidget(toolbar)
                tab_widget.setLayout(tab_layout)

    def read_text(self):
        text = self.default_nxcals.toPlainText()
        self.update_nxcals_beamplane()

    # Which NXCALS variables to consider
    def update_nxcals_beamplane(self):
        default_text = self.default_nxcals.toPlainText()
        checked_elements = []
        for i in range(self.list_beamplane.count()):
            item = self.list_beamplane.item(i)
            if item.checkState() == Qt.Checked:
                checked_elements.append(item.text())

        result = [default_text.replace("%", item) for item in checked_elements]
        
        model = QStringListModel(result)
        self.nxcals_beamplane.setModel(model)
    
    # Download info from nxcals concerning the number of bunches in the machine
    def bunch_info(self):
        start_time = time.time() 
        my_t1 = self.final_t1_cet
        my_t2 = self.final_t2_cet
        self.overwrite_bunch_info = self.check_overwrite_info_2.isChecked()
        self.textEdit_bunchinfo.clear()
        
        self.bunch_message = f"Considering time range from {my_t1} to {my_t2} (CET)\n\n"
        
        Path(f"{self.final_output_path}").mkdir(parents=True, exist_ok=True)
        save_path = f"{self.final_output_path}/bunch_info_{my_t1}_{my_t2}.parquet"
        
        if (self.overwrite_bunch_info) or ((not self.overwrite_bunch_info) and (not os.path.isfile(save_path) ) ):
            df = self.sk.nxcals_df(["LHC.BQM.B%:NO_BUNCHES"],
                  my_t1,
                  my_t2,
                  pandas_processing=[
                     nx.pandas_get,
                     nx.pandas_pivot,
                     ]
                 )
            df.to_parquet(save_path)
        df = pd.read_parquet(save_path)
        df_b1  = df["LHC.BQM.B1:NO_BUNCHES"].dropna().drop_duplicates().sort_index(ascending=False)
        df_b2 = df["LHC.BQM.B2:NO_BUNCHES"].dropna().drop_duplicates().sort_index(ascending=False)
        df_b1.index = [pd.Timestamp(i).tz_localize('UTC').tz_convert('CET') for i in df_b1.index]
        df_b2.index = [pd.Timestamp(i).tz_localize('UTC').tz_convert('CET') for i in df_b2.index]
        
        self.bunch_message += f"{df_b1}\n\n"
        self.bunch_message += f"{df_b2}\n\n"
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.bunch_message += f"\n\n Time needed to download bunch info {elapsed_time} seconds\n\n"

        self.textEdit_bunchinfo.setPlainText(self.bunch_message)

    # Download info from NXCALS in the time range specified
    def func_load_timerange_info(self, nxcals_flag):
        self.start_time = time.time()
        self.plot_beam_modes.setChecked(False) 
        self.button_fill_pressed = False

        myt1 = self.from_t1_2.dateTime().toPyDateTime()
        myt2 = self.to_t2_2.dateTime().toPyDateTime()
        self.final_t1_cet = pd.Timestamp(myt1).tz_localize('CET')
        self.final_t1_utc = self.final_t1_cet.tz_convert('UTC')
        self.final_t2_cet = pd.Timestamp(myt2).tz_localize('CET')
        self.final_t2_utc = self.final_t2_cet.tz_convert('UTC')
        if self.final_t1_utc>self.final_t2_utc:
            self.create_message_popup(f"Error! Start time cannot be before end time")
            return
       
        if nxcals_flag:
            start_time = time.time()
            if os.path.exists(f'{self.final_output_path}/info_{self.final_t1_cet}_{self.final_t2_cet}.parquet'):
                df = pd.read_parquet(f'{self.final_output_path}/info_{self.final_t1_cet}_{self.final_t2_cet}.parquet')
                mydf = df.copy()
                mydf['tstart'] = [pd.Timestamp(i).tz_localize('UTC') for i in mydf.index]
                mydf['mode'] = mydf['HX:BMODE']
            else:
            
                df = self.sk.nxcals_df(["HX:FILLN", "HX:BMODE"],
                      self.final_t1_cet,
                      self.final_t2_cet,
                      pandas_processing=[
                      nx.pandas_get,
                      nx.pandas_pivot,
                    ]
                    )
                df.index = [pd.Timestamp(i) for i in df.index]
                mydf = df.copy()
                mydf['tstart'] = [pd.Timestamp(i).tz_localize('UTC') for i in mydf.index]
                mydf['mode'] = mydf['HX:BMODE']
            self.beammodes_info = mydf
            try:        
              df['HX:FILLN'] = df['HX:FILLN'].ffill(axis=0)
              df.to_parquet(f'{self.final_output_path}/info_{self.final_t1_cet}_{self.final_t2_cet}.parquet',coerce_timestamps="us", allow_truncated_timestamps= True)

              group = df[df["HX:FILLN"].isna()]

              self.message = ''
              if len(group)>0:
                  self.message += f"No fill info \n\n Modes: {group['HX:BMODE'].dropna().unique()} \n\n" 
              
              for key, group in df.groupby("HX:FILLN"):
                  self.message += f"Fill {key} starting at {group.index[0].tz_localize('UTC').tz_convert('CET')} ('CET') \n\n Modes: {group['HX:BMODE'].dropna().unique()} \n\n"

            except:
                self.message = f"No fill info \n\n Modes: {df}"
                df.to_parquet(f'{self.final_output_path}/info_{self.final_t1_cet}_{self.final_t2_cet}.parquet',coerce_timestamps="us", allow_truncated_timestamps= True)
            end_time = time.time()
            elapsed_time=end_time-start_time
            self.message += f"\n\nTime needed to download info: {elapsed_time} seconds\n\n"
            self.info_time.setPlainText(self.message)
        
        self.find_files()
    
    # Download fill info from NXCALS
    def func_load_fill_info(self):
        self.start_time = time.time()
        self.plot_beam_modes.setChecked(False) 

        self.button_fill_pressed = True

        self.from_t1.clear()
        self.to_t2.clear()
        self.overwrite_fill_info = self.check_overwrite_info.isChecked()

        self.fill_nb = int(self.fillnb.text())
        self.beammode_start = str(self.box_from.currentText())
        self.beammode_end = str(self.box_to.currentText())

        Path(f"{self.final_output_path}").mkdir(parents=True, exist_ok=True)
        save_path = f"{self.final_output_path}/info_{self.fill_nb}.parquet"
        
        if (self.overwrite_fill_info) or ((not self.overwrite_fill_info) and (not os.path.isfile(save_path) ) ):
          aux = self.sk.get_fill_time(self.fill_nb)
          myaux = pd.DataFrame.from_dict(aux)
          myaux["tstart"] = myaux.apply(lambda x: x['modes']['start'], axis=1)
          myaux["tend"] = myaux.apply(lambda x: x['modes']["end"], axis=1)
          myaux["mode"] = myaux.apply(lambda x: x['modes']["mode"], axis=1)
      
          myaux.to_parquet(save_path, coerce_timestamps="us", allow_truncated_timestamps= True)

        else:
            myaux = pd.read_parquet(save_path)
        
        self.beammodes_info = myaux

        t1 = myaux[myaux['mode'] == self.beammode_start]["tstart"].values
        t2 = myaux[myaux['mode'] == self.beammode_end]["tend"].values

        if len(t1) == 1:
            self.final_t1_utc = pd.Timestamp(t1[0]).tz_localize('UTC')
            self.final_t1_cet = self.final_t1_utc.tz_convert("CET")
            self.from_t1.setText(self.final_t1_cet.strftime('%Y-%m-%d %H:%M:%S'))
        elif len(t1) == 0:
            self.create_message_popup(f"Error! Beam mode {self.beammode_start} does not exist for this fill")
            self.from_t1.clear()
            return
            
        elif len(t1)>1:
            #if several modes with same name are found, let the user decide
            list_widget = QListWidget()
            elements = t1
            for element in elements:
                element_t1_utc = pd.Timestamp(element).tz_localize('UTC')
                element_t1_cet = element_t1_utc.tz_convert("CET")
                item = QListWidgetItem(pd.Timestamp(element_t1_cet).strftime('%Y-%m-%d %H:%M:%S'))
                list_widget.addItem(item)
            dialog = QDialog()
            dialog.setWindowTitle(f"Select start time for {self.beammode_start}:")
            layout = QVBoxLayout()
            layout.addWidget(list_widget)
            #list_widget.itemSelectionChanged.connect(lambda: print(pd.Timestamp(list_widget.currentItem().text()).tz_localize('CET')))
            list_widget.itemSelectionChanged.connect(lambda: set_selected_item(pd.Timestamp(list_widget.currentItem().text()).tz_localize('CET')))

            def set_selected_item(item):
                self.final_t1_cet = item
                self.final_t1_utc = self.final_t1_cet.tz_convert("UTC")
            
            dialog.setLayout(layout)
            dialog.exec_()
        
            self.from_t1.setText(self.final_t1_cet.strftime('%Y-%m-%d %H:%M:%S'))
        
        if len(t2) == 1:
            self.final_t2_utc = pd.Timestamp(t2[0]).tz_localize('UTC')
            self.final_t2_cet = self.final_t2_utc.tz_convert("CET")
            self.to_t2.setText(self.final_t2_cet.strftime('%Y-%m-%d %H:%M:%S'))
        elif len(t2) == 0:
            self.create_message_popup(f"Error! Beam mode {self.beammode_end} does not exist for this fill")
            self.to_t2.clear()
        elif len(t2)>1:
            #if several modes with same name are found, let the user decide
            list_widget = QListWidget()
            elements = t2
            for element in elements:
                element_t2_utc = pd.Timestamp(element).tz_localize('UTC')
                element_t2_cet = element_t2_utc.tz_convert("CET")
                item = QListWidgetItem(pd.Timestamp(element_t2_cet).strftime('%Y-%m-%d %H:%M:%S'))
                list_widget.addItem(item)
            dialog = QDialog()
            dialog.setWindowTitle(f"Select end time for {self.beammode_end}:")
            layout = QVBoxLayout()
            layout.addWidget(list_widget)
            list_widget.itemSelectionChanged.connect(lambda: set_selected_item(pd.Timestamp(list_widget.currentItem().text()).tz_localize('CET')))

            def set_selected_item(item):
                self.final_t2_cet = item
                self.final_t2_utc = self.final_t2_cet.tz_convert("UTC")
            
            dialog.setLayout(layout)
            dialog.exec_()
        
            self.to_t2.setText(self.final_t2_cet.strftime('%Y-%m-%d %H:%M:%S'))

        try:
            if self.final_t2_cet<self.final_t1_cet:
                self.create_message_popup(f"Error! Start time cannot be before end time")
                self.to_t2.clear()
                self.from_t1.clear()
        except:
            pass

        self.find_files()


    # Possibility to take a screenshot of the mainwindow
    def take_screenshot(self):
        screenshot = self.grab()
        file_name, ok = QInputDialog.getText(self, "Save Screenshot", "Enter png file name:")
        if ok:
            file_name = str(file_name) + ".png"
            screenshot.save(file_name)

    
    # Function that creates popup messages
    def create_message_popup(self, text, icon=QMessageBox.Critical):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f"An error occurred: {text}")
            msg.setWindowTitle("Error")
            msg.exec_()

    # Launch analysis
    def on_button_clicked(self):
        #self.run_button.setStyleSheet("background-color: red;")

        button_style = '''
            QPushButton {
                background-color: red;
                border: 2px solid black;
                border-radius: 10px;
                color: black;
                font-weight: bold;
                font-size: 18px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: red;
            }
            QPushButton:pressed {
                background-color: red;
                border: 2px solid #707070;
            }
        '''
        self.run_button.setStyleSheet(button_style)

        self.run_button.setText("Running...")
        QApplication.processEvents()
        loop = QEventLoop()
        loop.processEvents()
        self.click_run()
        button_style = '''
            QPushButton {
                background-color: lightblue;
                border: 2px solid black;
                border-radius: 10px;
                color: black;
                font-weight: bold;
                font-size: 18px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: lightblue;
            }
            QPushButton:pressed {
                background-color: red;
                border: 2px solid #707070;
            }
        '''

        self.run_button.setStyleSheet(button_style)
        self.run_button.setText("\U00002192 Run")
        loop.exit()

    # Determines which files to download or which files already exist if any if the user has unchecked the "overwrite files" checkbox. Will also split the data to be downloaded from NXCALS into chunks of "Time range splits (mins)"
    def files_to_download(self):
            pd.set_option('display.max_columns', 100)

            if not self.check_overwrite.isChecked():
                self.message_debug +=f"Overwrite files option is de-activated. Looking for any relevant files..\n\n"
                FFT_files = [i for i in os.listdir(self.final_output_path) if i.startswith("FFT") and i.endswith("parquet")]
                
                all_files = []
                all_beamplanes  = []
                all_timestamps1 = []
                all_timestamps2  = []
                if len(FFT_files)>0:
                    import re
                    pattern = re.compile(r'FFT_(\w+)_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\.parquet')
                    matches = [re.findall(pattern, string) for string in FFT_files]
                    for counter, match in enumerate(matches):
                        if match:
                            all_files.append(FFT_files[counter])
                            beamplane, timestamp1, timestamp2 = match[0]
                            timestamp1 = pd.Timestamp(timestamp1, tz='UTC')
                            timestamp2 = pd.Timestamp(timestamp2, tz='UTC')
                            all_beamplanes.append(beamplane)
                            all_timestamps1.append(timestamp1)
                            all_timestamps2.append(timestamp2)
                myffts = pd.DataFrame({"filename": all_files, "t1": all_timestamps1, "t2": all_timestamps2, "beamplane": all_beamplanes})
            else:
                self.message_debug +=f"Overwrite files option active. Will download all the data\n\n"
            
            for beamplane, nxcals_var in zip(self.checked_elements, self.final_nxcals_vars):
                self.message_debug +=f"\n\n#########################\n\n"
                self.message_debug +=f"\n\n{beamplane}\n\n"
                self.message_debug +=f"\n\n#########################\n\n"
                myt1 = self.final_t1_utc
                myt2 = self.final_t2_utc
                
                if self.check_overwrite.isChecked():
                    df_all = pd.DataFrame(columns = ['t1', 't2', 'flag'])
                else:
                    myffts_beamplane = myffts[myffts["beamplane"] == beamplane]
                    # Sort the dataframe by t1
                    myffts_beamplane = myffts_beamplane.sort_values(by='t1')
                
                    # Find which time ranges to download and which to omit
                    df_all = myffts_beamplane[['t1', 't2']]
                    df_all['flag'] = 'exists'

                    # if one range is included fully in another, drop
                    df_all = df_all.sort_values(by='t1') 
                    deduplicated_df_all = pd.DataFrame(columns=['t1', 't2', 'flag']) 
                    for i in range(len(df_all)):
                        if i == 0:
                            deduplicated_df_all = deduplicated_df_all.append(df_all.iloc[i])
                        else:
                            if df_all['t1'].iloc[i] >= deduplicated_df_all['t1'].iloc[-1] and df_all['t2'].iloc[i] <= deduplicated_df_all['t2'].iloc[-1]:
                                continue
                            else:
                                deduplicated_df_all = deduplicated_df_all.append(df_all.iloc[i])
                
                    df_all = deduplicated_df_all.reset_index(drop=True)
                    mask = ((df_all['t1'] >= self.final_t1_utc) & (df_all['t1'] <= self.final_t2_utc)) | \
                        ((df_all['t2'] >= self.final_t1_utc) & (df_all['t2'] <= self.final_t2_utc)) | \
                        ((df_all['t1'] <= self.final_t1_utc) & (df_all['t2'] >= self.final_t2_utc))

                    # filter the rows using the mask
                    df_all = df_all[mask]


                # Find missing gaps                
                df_all = df_all.sort_values(by='t1')

                # df_all with be empty if no files found or if overwrite files option is active
                if len(df_all)>0:
                    gaps = []
                    
                    if df_all['t1'].iloc[0] > myt1:
                        gaps.append([myt1, df_all['t1'].iloc[0], 'gap'])
                    
                    i = 0
                    while i < len(df_all) - 1:
                        if df_all['t2'].iloc[i] < df_all['t1'].iloc[i+1]:
                            gaps.append([df_all['t2'].iloc[i], df_all['t1'].iloc[i+1], 'gap'])
                        i += 1
                    
                    if df_all['t2'].iloc[-1] < myt2:
                        gaps.append([df_all['t2'].iloc[-1], myt2, 'gap'])
                    
                    result = pd.concat([df_all, pd.DataFrame(gaps, columns=['t1', 't2', 'flag'])])
                    df_all = result.sort_values(by='t1')

                else:
                    df_all = pd.DataFrame({"t1": [myt1], "t2": [myt2], "flag": ["gap"]})

                df_all.reset_index(inplace=True, drop=True)
                self.message_debug += f"All files: {df_all}\n\n"
                df_all['dt'] = (df_all['t2']-df_all['t1']).dt.total_seconds()

                # remove very small time gap ranges
                df_all = df_all[df_all['dt'] >60.0]
                self.message_debug += f"After removing small gaps: {df_all}\n\n" 
                df_gaps = df_all[df_all["flag"] == "gap"]
                df_exists = df_all[df_all["flag"] == "exists"]


                # split df_gaps further if needed
                split_into = self.spinbox.value()
                if (df_gaps['dt'] > split_into*60.0).any():
                    chunks = []
                    # Loop through the dataframe
                    for i in range(len(df_gaps)):
                        t1 = df_gaps['t1'].iloc[i]
                        t2 = df_gaps['t2'].iloc[i]
                        flag = df_gaps['flag'].iloc[i]
                        chunk_start = t1
                        chunk_end = chunk_start + pd.Timedelta(minutes=split_into)
                        while chunk_end < t2:
                            chunks.append([chunk_start, chunk_end, flag])
                            chunk_start = chunk_end
                            chunk_end = chunk_start + pd.Timedelta(minutes=split_into)
                        chunks.append([chunk_start, t2, flag])
                    
                    # Combine the original time ranges and the chunks into a single dataframe
                    result = pd.concat([df_gaps, pd.DataFrame(chunks, columns=['t1', 't2', 'flag'])])
                    
                    # Sort the result by t1
                    df_gaps = result.sort_values(by='t1')
                    df_gaps['dt'] = (df_gaps['t2']-df_gaps['t1']).dt.total_seconds()
                    df_gaps = df_gaps[df_gaps['dt'] <=split_into*60.0]
                    df_all = pd.concat([df_gaps, df_exists], axis=0)
                    df_all.sort_values(by='t1', inplace=True)
                    df_all.reset_index(inplace=True, drop=True)

                df_gaps = df_all[df_all["flag"] == "gap"]
                df_exists = df_all[df_all["flag"] == "exists"]


                # split df_gaps further if needed
                split_into = self.spinbox.value()
                if (df_gaps['dt'] > split_into*60.0).any():
                    chunks = []
                    # Loop through the dataframe
                    for i in range(len(df_gaps)):
                        t1 = df_gaps['t1'].iloc[i]
                        t2 = df_gaps['t2'].iloc[i]
                        flag = df_gaps['flag'].iloc[i]
                        chunk_start = t1
                        chunk_end = chunk_start + pd.Timedelta(minutes=split_into)
                        while chunk_end < t2:
                            chunks.append([chunk_start, chunk_end, flag])
                            chunk_start = chunk_end
                            chunk_end = chunk_start + pd.Timedelta(minutes=split_into)
                        chunks.append([chunk_start, t2, flag])
                    
                    # Combine the original time ranges and the chunks into a single dataframe
                    result = pd.concat([df_gaps, pd.DataFrame(chunks, columns=['t1', 't2', 'flag'])])
                    
                    # Sort the result by t1
                    df_gaps = result.sort_values(by='t1')
                    df_gaps['dt'] = (df_gaps['t2']-df_gaps['t1']).dt.total_seconds()
                    df_gaps = df_gaps[df_gaps['dt'] <=split_into*60.0]
                    df_all = pd.concat([df_gaps, df_exists], axis=0)
                    df_all.sort_values(by='t1', inplace=True)
                    df_all.reset_index(inplace=True, drop=True)


                df_gaps = df_all[df_all["flag"] == "gap"]
                df_exists = df_all[df_all["flag"] == "exists"]


                self.message_debug +=f"Files that already exist: {df_exists}\n\n"
                self.message_debug +=f"Files that will be downloaded: {df_gaps}\n\n"
                self.data_to_download[beamplane] = df_all
                self.end_time = time.time()
                elapsed_time = self.end_time - self.start_time
                self.message_debug += f"\n\nElapsed time: {elapsed_time} seconds\n\n"
                (self.text_debug.setPlainText(self.message_debug))

    # Function that determines which data to consider before downloading
    def find_files(self):
        
        self.message_debug = ''

        if self.button_fill_pressed:
            # read from fill tab
            self.message_debug += "Reading time range from fill tab: "
            myt1 = self.from_t1.text()
            myt2 = self.to_t2.text()
        else:
            # read from time range tab
            self.message_debug += "Reading time range from time range tab: "
            myt1 = self.from_t1_2.dateTime().toPyDateTime()
            myt2 = self.to_t2_2.dateTime().toPyDateTime()
        self.final_t1_cet = pd.Timestamp(myt1).tz_localize('CET')
        self.final_t1_utc = self.final_t1_cet.tz_convert('UTC')
        self.final_t2_cet = pd.Timestamp(myt2).tz_localize('CET')
        self.final_t2_utc = self.final_t2_cet.tz_convert('UTC')
        self.message_debug +=f"from {self.final_t1_cet} to {self.final_t2_cet} \n\n"

        # nxcals variables
        default_text = self.default_nxcals.toPlainText()
        self.checked_elements = []
        for i in range(self.list_beamplane.count()):
            item = self.list_beamplane.item(i)
            if item.checkState() == Qt.Checked:
                self.checked_elements.append(item.text())
        result = [default_text.replace("%", item) for item in self.checked_elements]
        self.final_nxcals_vars = result

        self.final_output_path = self.output_path.toPlainText()

        self.final_overwrite = self.check_overwrite.isChecked()

        self.split_into = self.spinbox.value()
        
        self.data_to_download = {}
        self.files_to_download()
        
        return

    # Main function that will download data from nxcals or read the pre-existing files
    def click_run(self):
        self.start_time = time.time()
        self.run_button.setStyleSheet("background-color : red")
        self.files_for_spectr = {}
        for beamplane, nxcals_var in zip(self.checked_elements, self.final_nxcals_vars):
                self.message_debug += f"Downloading data for {beamplane}..\n\n"
                df_all = self.data_to_download[beamplane]

                df_gaps = df_all[df_all["flag"] == "gap"]
                df_exists = df_all[df_all["flag"] == "exists"]

                appended_df = []
                for i in range(len(df_exists)):
                    tfirst = df_exists.iloc[i]['t1'].tz_convert(None)
                    tlast =  df_exists.iloc[i]['t2'].tz_convert(None)
                    mydf = pd.DataFrame(pd.read_parquet(f"{self.final_output_path}/FFT_{beamplane}_{tfirst}_{tlast}.parquet")[nxcals_var].dropna())
                    tol = pd.Timedelta(minutes=1)
                    if (self.final_t1_utc>=tfirst.tz_localize('UTC')-tol) and (self.final_t2_utc<=tlast.tz_localize('UTC')+tol):
                        
                        mydf_copy = mydf.copy()
                        mydf_copy.index = [pd.Timestamp(i).tz_localize('UTC') for i in mydf_copy.index]
                        
                    
                        mask = (mydf_copy.index>=self.final_t1_utc) & (mydf_copy.index<=self.final_t2_utc)
                        self.message_debug +=f"Will use pre-existing file only from {self.final_t1_utc} to {self.final_t2_utc} \n\n"

                        appended_df.append(mydf[mask])    
                    else:
                        appended_df.append(mydf)

                appended_df_gap = []
                for i in range(len(df_gaps)):
                    tfirst = df_gaps.iloc[i]['t1'].tz_convert(None)
                    tlast =  df_gaps.iloc[i]['t2'].tz_convert(None)
                    print("Downloading..", df_gaps.iloc[i], [nxcals_var], tfirst, tlast)
                    df = self.sk.nxcals_df([nxcals_var], tfirst, tlast,pandas_processing=[nx.pandas_get,nx.pandas_pivot,])
                    if len(df)>0:
                        appended_df_gap.append(df)
                        self.message_debug +=f"Downloading data for {tfirst} to {tlast} \n"

                if (len(appended_df)>0) and (len(appended_df_gap)>0):
                    appended_df = pd.concat(appended_df, axis=0)
                    appended_df_gap = pd.concat(appended_df_gap, axis=0)
                    df = pd.concat([appended_df, appended_df_gap], axis=0)
                elif (len(appended_df)>0):
                    appended_df = pd.concat(appended_df, axis=0)
                    df = appended_df
                elif (len(appended_df_gap)>0):
                    appended_df_gap = pd.concat(appended_df_gap, axis=0)
                    df = appended_df_gap
                else:
                    self.create_message_popup(f"Error! Empty dataframe in the results")

                df = df[~df.index.duplicated(keep='first')]
                df.sort_index(inplace=True)

                tfirst = pd.Timestamp(df.index[0])
                tlast = pd.Timestamp(df.index[-1])
                df = pd.DataFrame(df, columns = [nxcals_var])
                try:
                    os.makedirs(self.final_output_path)
                except FileExistsError:
                    pass
                
                df.to_parquet(f"{self.final_output_path}/FFT_{beamplane}_{tfirst}_{tlast}.parquet")
                self.files_for_spectr[beamplane] = f"{self.final_output_path}/FFT_{beamplane}_{tfirst}_{tlast}.parquet"
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        self.message_debug += f"Elapsed time: {elapsed_time} seconds" 
        (self.text_debug.setPlainText(self.message_debug))
        all_figs = self.plot_spectr(self.files_for_spectr, self.fmin, self.fmax)
        self.all_figs = all_figs
        for beamplane in all_figs.keys():
            self.show_spectr(all_figs[beamplane], beamplane)
        return
        
    # Function that plots the spectrograms
    def plot_spectr(self, files_for_spectr, fmin, fmax, frev = 11245.5, beammodes=None):
        all_figs = {}
        for beamplane in files_for_spectr.keys():
            df = pd.read_parquet(files_for_spectr[beamplane])
            df = df.dropna()

            df['fourier'] = df.apply(lambda x: x[0]['elements'], axis=1)
            df = df[[not all(np.isnan(row)) for row in df['fourier']]]
            df.index = [pd.Timestamp(i).tz_localize("UTC") for i in df.index]

            x_lims = mdates.date2num(df.index.values)
            
            freqs = np.linspace(0, frev, len(df['fourier'].iloc[0]))
            fourier_abs = np.array(df['fourier'].to_list())

            myfilter = (freqs>fmin) & (freqs<fmax)
            fig, ax = plt.subplots(figsize=(9.,8.5))
            plt.pcolormesh(freqs[myfilter], df.index.values, np.array(np.log10(fourier_abs)[:, myfilter]), cmap='jet', shading='auto')

            plt.xlim(freqs[myfilter][0], freqs[myfilter][-1])
            plt.ylim(df.index.values[0], df.index.values[-1])
            date_format = mdates.DateFormatter('%H:%M:%S')
            ax.yaxis.set_major_formatter(date_format)

            plt.xlabel('f (Hz)')
            plt.ylabel(f"UTC time {df.index[0].day}/{df.index[0].month}/{df.index[0].year}")

            if beammodes is not None:
                for i in range(len(beammodes)):
                    tt = beammodes.iloc[i]['tstart']
                    mode = beammodes.iloc[i]['mode']
                    print(tt, mode)
                    plt.axhline(tt, c='k', lw=2)
                    plt.text(self.fmin, tt, mode, c='k')

            fig.tight_layout()
            all_figs[beamplane] = fig
        return all_figs

#Some default names and paths
current_date = datetime.datetime.now().strftime("%d%m%Y")
current_path = os.getcwd()
default_output_path  = f"{current_path}/results_{current_date}"
default_python_path = sys.executable
nxcals_conf = {'spark.driver.maxResultSize': '8g',
                'spark.executor.memory':'8g',
                'spark.driver.memory': '16g',
                'spark.executor.instances': '20',
                'spark.executor.cores': '2',
                }
def main():
    app = QtWidgets.QApplication([])
    application = MainWindow()
    application.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
