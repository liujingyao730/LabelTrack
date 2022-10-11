# 如何实现这个功能？

## 目前对方框的标注原理

标注图片的核心逻辑是在./GUI/canvas.py中完成的。
首先这个canvas类继承了Qwidget，是一个pyqt的组件。Qwiget和window的区别是：<https://blog.csdn.net/u011699626/article/details/116332144>

canvas的初始化流程：

- 读取视频，并保存为frame数组imgFrames;
- 创建文件读取器和目标跟踪器fileWorker和trackWorker
- 初始化整个的paint画图流程
  - 用当前帧创建一个可以paint的pixmap，用来显示在整个canvas里；
  - 创建一个绘图用的painter;
  - 每一个方框都是一个shape;
