import sys
import joblib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox
)
from PyQt5.QtGui import QFont


class GBDTApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = joblib.load('model.pkl')  # 加载GBDT模型

    def initUI(self):
        layout = QVBoxLayout()
        formLayout = QFormLayout()

        # 创建字体并设置大小
        font = QFont()
        font.setPointSize(20)  # 设置字体大小为12

        # 创建输入框
        self.inputs = [QLineEdit() for _ in range(15)]

        varibles = ['Trouble sleep', 'Life satisfaction', 'Mobility limitation', 'Self-rated health', 'Gender',
                    'Social inactivity', 'Handgrip strength',  'Loneliness', 'Psychiatric problems',
                    'Social inactivity', 'Body pain', 'Vision impairment', 'Receive pension',
                    'Physical inactivity', 'Number of chronic diseases']

        for i in range(15):
            # 创建并设置变量名称标签
            variable_label = QLabel(f'{varibles[i]}:')
            variable_label.setFont(font)  # 应用字体

            # 设置输入框的字体
            self.inputs[i].setFont(font)

            # 将标签和输入框添加到布局
            formLayout.addRow(variable_label, self.inputs[i])

        layout.addLayout(formLayout)

        # Create predict button
        self.predictButton = QPushButton('Predict')
        self.predictButton.setFont(font)  # Apply font
        self.predictButton.clicked.connect(self.predict)
        layout.addWidget(self.predictButton)

        self.resultLabel = QLabel('Prediction Result: ')
        self.resultLabel.setFont(font)  # Apply font
        self.probLabel = QLabel('Prediction Probability: ')
        self.probLabel.setFont(font)  # Apply font
        layout.addWidget(self.resultLabel)
        layout.addWidget(self.probLabel)

        self.setLayout(layout)
        self.setWindowTitle('GBM Prediction')
        self.show()

    def predict(self):
        # Get input variable values
        try:
            inputs = [float(input_box.text()) for input_box in self.inputs]
            inputs_array = np.array(inputs).reshape(1, -1)

            # Make prediction
            prediction = self.model.predict(inputs_array)[0]
            prediction_proba = self.model.predict_proba(inputs_array)[0]

            # Update result labels
            self.resultLabel.setText(f'Prediction Result: {prediction}')
            self.probLabel.setText(f'Prediction Probability: {prediction_proba}')
        except ValueError:
            QMessageBox.warning(self, 'Input Error', 'Please ensure all input values are numbers.')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GBDTApp()
    sys.exit(app.exec_())
