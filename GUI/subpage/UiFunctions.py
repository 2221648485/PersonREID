from GUI.subpage.custom_grips import CustomGrip
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QEvent, QTimer
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import time

GLOBAL_STATE = False    # max min flag
GLOBAL_TITLE_BAR = True

class UIFuncitons:
    # 展开/收起左侧菜单
    def toggleMenu(self, enable):
        if enable:
            standard = 40        # 收起状态宽度
            maxExtend = 180      # 展开状态宽度
            width = self.LeftMenuBg.width()
            print(width)
            # 根据当前宽度切换状态
            if width == 40:
                widthExtended = maxExtend
            else:
                widthExtended = standard
            
            # 创建宽度动画
            self.animation = QPropertyAnimation(self.LeftMenuBg, b"minimumWidth")
            self.animation.setDuration(300)       # 动画时长300ms
            self.animation.setStartValue(width)   # 初始值
            self.animation.setEndValue(widthExtended)  # 结束值
            self.animation.setEasingCurve(QEasingCurve.InOutQuint)  # 缓动曲线
            self.animation.start()

    # 右侧设置面板展开/收起
    def settingBox(self, enable):
        if enable:
            # 获取当前尺寸
            widthRightBox = self.prm_page.width()   # 右侧面板当前宽度
            widthLeftBox = self.LeftMenuBg.width()  # 左侧菜单当前宽度
            maxExtend = 220      # 最大展开宽度
            standard = 0         # 收起状态宽度

            # 切换右侧面板状态
            if widthRightBox == 0:
                widthExtended = maxExtend
            else:
                widthExtended = standard

            # 左侧菜单收缩动画（固定缩到40px）
            self.left_box = QPropertyAnimation(self.LeftMenuBg, b"minimumWidth")
            self.left_box.setDuration(500)
            self.left_box.setStartValue(widthLeftBox)
            self.left_box.setEndValue(40)
            self.left_box.setEasingCurve(QEasingCurve.InOutQuart)

            # 右侧面板展开动画
            self.right_box = QPropertyAnimation(self.prm_page, b"minimumWidth")
            self.right_box.setDuration(500)
            self.right_box.setStartValue(widthRightBox)
            self.right_box.setEndValue(widthExtended)
            self.right_box.setEasingCurve(QEasingCurve.InOutQuart)

            # 并行执行动画组
            self.group = QParallelAnimationGroup()
            self.group.addAnimation(self.left_box)
            self.group.addAnimation(self.right_box)
            self.group.start()

    # 窗口最大化/还原切换
    def maximize_restore(self):
        global GLOBAL_STATE
        status = GLOBAL_STATE  # 获取当前窗口状态
        if status == False:
            GLOBAL_STATE = True
            self.showMaximized()    # 最大化窗口
            self.max_sf.setToolTip("Restore")  # 修改工具提示
            # 隐藏四周边框调整手柄
            self.left_grip.hide()
            self.right_grip.hide()
            self.top_grip.hide()
            self.bottom_grip.hide()
        else:
            GLOBAL_STATE = False
            self.showNormal()       # 还原窗口
            self.resize(self.width()+1, self.height()+1)  # 微调尺寸触发重绘
            self.max_sf.setToolTip("Maximize")
            # 显示四周边框调整手柄
            self.left_grip.show()
            self.right_grip.show()
            self.top_grip.show()
            self.bottom_grip.show()
    
    # 窗口控件定义
    def uiDefinitions(self):
        # 双击标题栏最大化
        def dobleClickMaximizeRestore(event):
            if event.type() == QEvent.MouseButtonDblClick:
                # 延迟250ms执行防止误触
                QTimer.singleShot(250, lambda: UIFuncitons.maximize_restore(self))
        self.top.mouseDoubleClickEvent = dobleClickMaximizeRestore
        
        # 窗口拖拽移动逻辑
        def moveWindow(event):
            if GLOBAL_STATE:         # 最大化状态下点击会先恢复窗口
                UIFuncitons.maximize_restore(self)
            if event.buttons() == Qt.LeftButton:
                # 计算窗口新位置：当前位置 + 鼠标移动差值
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()  # 更新拖拽基准点
        self.top.mouseMoveEvent = moveWindow

        # 初始化四周边框调整手柄
        self.left_grip = CustomGrip(self, Qt.LeftEdge, True)   # 左侧手柄
        self.right_grip = CustomGrip(self, Qt.RightEdge, True) # 右侧手柄
        self.top_grip = CustomGrip(self, Qt.TopEdge, True)     # 顶部手柄
        self.bottom_grip = CustomGrip(self, Qt.BottomEdge, True) # 底部手柄

        # 按钮信号连接
        self.min_sf.clicked.connect(lambda: self.showMinimized())  # 最小化
        self.max_sf.clicked.connect(lambda: UIFuncitons.maximize_restore(self)) # 最大化/还原
        self.close_button.clicked.connect(self.close)  # 关闭窗口

    # 调整边框手柄位置
    def resize_grips(self):
        # 设置手柄几何位置（留出10px边距）
        self.left_grip.setGeometry(0, 10, 10, self.height())        # 左侧垂直居中
        self.right_grip.setGeometry(self.width() - 10, 10, 10, self.height()) # 右侧
        self.top_grip.setGeometry(0, 0, self.width(), 10)          # 顶部水平铺开
        self.bottom_grip.setGeometry(0, self.height() - 10, self.width(), 10) # 底部

    # 添加阴影效果
    def shadow_style(self, widget, Color):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setOffset(8, 8)       # 阴影偏移量（右下方8px）
        shadow.setBlurRadius(38)     # 阴影模糊半径
        shadow.setColor(Color)       # 阴影颜色参数
        widget.setGraphicsEffect(shadow)  # 应用阴影到指定控件        shadow.setOffset(8, 8)  # offset
        shadow.setBlurRadius(38)    # shadow radius
        shadow.setColor(Color)    # shadow color
        widget.setGraphicsEffect(shadow) 