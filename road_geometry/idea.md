# 如何实现这个功能？

## 目前对方框的标注原理

标注图片的核心逻辑是在./GUI/canvas.py中完成的。
首先这个canvas类继承了Qwidget，是一个pyqt的组件。Qwiget和window的区别是：<https://blog.csdn.net/u011699626/article/details/116332144>

当前标记的流程：

1. 选择标注功能
2. 选择标注种类
3. 鼠标移动到图片上之后会变成十字，然后点击之后开始画框，释放左键之后变回正常。
4. 对框双击可以修改框的属性、可以拖拽框的位置、拖拽角点可以修改框的大小和形状。

canvas的初始化流程：

- 读取视频，并保存为frame数组imgFrames;
- 创建文件读取器和目标跟踪器fileWorker和trackWorker
- 初始化整个的paint画图流程
  - 用当前帧创建一个可以paint的pixmap，用来显示在整个canvas里；
  - 创建一个绘图用的painter;
  - 每一个方框都是一个shape;
  - 开始监听鼠标的移动事件和焦点事件;

canvas 的绘制过程：

- 当鼠标在canvas上移动的时候就触发mouseMoveEvent
  - 更新鼠标当前所在的坐标（从widget相对坐标变成相对图片的坐标）
  - 更换鼠标的图标种类
  - 如果当前有在画的框，就显示xywh
  - 如果当前框中心在图片外，就裁剪到图像边缘
  - 更新十字参考线的参数
  - 调用repaint()，对全图触发paintEvent()绘制整张图片
    - <https://blog.csdn.net/chenyazhou88/article/details/106646418>

paintEvent的绘制过程：

- 创建painter
- 在canvas上画出整张原图
- 在原图上调用shape的paint画出所有的已标注shape
  - <https://blog.csdn.net/yy471101598/article/details/78543459>
- 如果正在拖拽则画出当前的矩形
  - 原理上和小画板是一样的，只不过背景是图片。
  - <https://www.jianshu.com/p/879a4f2dd77b>
  - <https://blog.csdn.net/W3Chhhhhh/article/details/105468389>
- 画出当前的十字参考线
- 绘制完当前的示意图之后，保存shape并重新绘制，防止草稿影响查看。
  
## 需要进行的改进

当前标记流程的变化：

1. 添加一种标注功能按钮
2. 研究如何标注出连续直线
3. 左键开始标注，右键停止
4. 对关键点可以拖拽，对直线进行删除
5. 双击线可以修改直线的属性

更新之后canvas的#绘制#过程：

- 仍然是调用mouseMoveEvent
  - 左键点击开始画图<https://blog.csdn.net/ye281842_/article/details/116396708> <http://cn.voidcc.com/question/p-khpwvxne-bmr.html>
  - 下次左键点击画下一个关键点
  - 最后的右键点击画下最后一个关键点，停止绘制。

对1，需要修改gui，添加一个按钮和一个下拉列表，决定是哪一种车道线。
<https://blog.csdn.net/m0_63993933/article/details/121732044>
1.5需要做的是研究明白这里面的signal和event之间的关系。
对2，需要研究qt里是否有这个shape，如果有的话可以直接大量复用代码。
