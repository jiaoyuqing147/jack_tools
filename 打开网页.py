from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import sys

class SimpleBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("简易浏览器")
        self.setGeometry(100, 100, 1000, 700)

        # 浏览器控件
        self.browser = QWebEngineView()
        self.browser.load(QUrl("http://210.45.92.67"))

        # 输入框和按钮
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("请输入网址，例如 http://example.com")
        self.open_button = QPushButton("打开网页")
        self.open_button.clicked.connect(self.load_url)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.url_input)
        layout.addWidget(self.open_button)
        layout.addWidget(self.browser)
        self.setLayout(layout)

    def load_url(self):
        url = self.url_input.text().strip()
        if not url.startswith("http"):
            url = "http://" + url
        self.browser.load(QUrl(url))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleBrowser()
    window.show()
    sys.exit(app.exec_())
