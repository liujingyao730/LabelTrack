import os
import functools
import torch
import numpy as np
import sys
sys.path.append("../../")
sys.path.append("./Tracking")


from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

# TODO

from PyQt5.QtCore import *
from GUI.shape import Shape
from GUI.utils import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # print("y: {}".format(y))
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# VISDRONE_CLASSES = (
#     "small",
#     "car",
#     "truck",
#     "bus"
# )

def update_shape(id, frameId, cls_id, tlwh, score, auto = 'M'):
        shapes = []
        detectPos = Shape()
        detectPos.id = id
        detectPos.frameId = frameId
        label = VISDRONE_CLASSES[cls_id]
        detectPos.label = label
        detectPos.score = score
        detectPos.auto = auto
        # generate_line_color, generate_fill_color = generate_color_by_text(detectPos.label)
        # self.set_shape_label(detectPos, detectPos.label, detectPos.id, generate_line_color, generate_fill_color)
        leftTop = QPointF(tlwh[0], tlwh[1])
        rightTop = QPointF(tlwh[0] + tlwh[2], tlwh[1])
        rightDown = QPointF(tlwh[0] + tlwh[2], tlwh[1] + tlwh[3])
        leftDown = QPointF(tlwh[0], tlwh[1] + tlwh[3])
        pointPos = [leftTop, rightTop, rightDown, leftDown]
        for pos in pointPos:
            # if self.out_of_pixmap(pos):
            #     size = self.pixmap.size()
            #     clipped_x = min(max(0, pos.x()), size.width())
            #     clipped_y = min(max(0, pos.y()), size.height())
            #     pos = QPointF(clipped_x, clipped_y)
            detectPos.add_point(pos)
        
        detectPos.close()
        shapes.append(detectPos)
        detectPos = None
        # self.set_hiding(False)
        return shapes[0]

def save_labels(shapes, savedPath, numFrames, video_width, video_height):
    def convert(shape, box):
        dw = 1. / shape[0]
        dh = 1. / shape[1]
        x = (box[0] + box[2] / 2) * dw
        y = (box[1] + box[3] / 2) * dh
        w = box[2] * dw
        h = box[3] * dh
        return (x, y, w, h)

    results = []
    for shape in shapes:
        min_x = sys.maxsize
        min_y = sys.maxsize
        max_x = 0
        max_y = 0
        for point in shape.points:
            min_x = round(min(min_x, point.x()))
            min_y = round(min(min_y, point.y()))
            max_x = round(max(max_x, point.x()))
            max_y = round(max(max_y, point.y()))
        w = max_x - min_x
        h = max_y - min_y
        # classId = utils.VISDRONE_CLASSES.index(shape.label)
        classId = VISDRONE_CLASSES.index(shape.label)
        if shape.auto == 'M':
                    for i in range(1, numFrames + 1):
                        results.append(
                        f"{i},{shape.id},{min_x},{min_y},{w},{h},{shape.score:.2f},{classId},0,0\n"
                    )
        else:
            results.append(
                f"{shape.frameId},{shape.id},{min_x},{min_y},{w},{h},{shape.score:.2f},{classId},0,0\n"
            )
    with open(savedPath, 'w') as f:
            f.writelines(results)
            print(f"save results to {savedPath}")


def filename_cmp(a: str, b: str):
    a = a.replace(".txt", "").split('_')
    b = b.replace(".txt", "").split('_')
    ka, frame_a = int(a[-2]), int(a[-1])
    kb, frame_b = int(b[-2]), int(b[-1])
    if(ka > kb): return 1
    elif(ka < kb): return -1
    else:
        if(frame_a > frame_b): return 1
        elif(frame_a < frame_b): return -1
        else: return 0


def frames_track(test_size, labels, config):
    """
    test_size: image size
    labels: list[list[tuple]], tuple: class, xywh, conf, 里面List：同一帧检测框, 外面list: 不同帧
    config: 配置
    """
    shape = []
    tracker = BYTETracker(config, frame_rate=config.fps)
    results = []
    resultImg = []
    height = 2176
    width = 3840
    for frame_id, frame_labels in enumerate(labels):
        online_targets = tracker.update(frame_labels, [height, width], test_size)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            cid = int(t.cls_id)
            vertical = tlwh[2] / tlwh[3] > config.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > config.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                # save results
                results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )
                # 更新图像信息
                # canvas.update_shape(tid, frame_id, cid, tlwh, t.score, 'A')
                shape.append(update_shape(tid, frame_id, cid, tlwh, t.score, 'A'))
    return shape


if __name__ == '__main__':
    WH = (3840, 2160)  # 不同视频需要调整
    labels_path = "D:\\project\\LabelTrack\\Tracking\\detect\\03"
    label_path_list = []
    for root, dirs, files in os.walk(labels_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            print("dir_path: ", dir_path)
            for root_dir, _, txt_files in os.walk(dir_path):
                for txt_file in txt_files:
                    # print("txt_files ", txt_files)
                    if(txt_files is not None):
                        label_path_list.append(os.path.join(root_dir, txt_file))
    label_path_list = sorted(label_path_list, key=functools.cmp_to_key(filename_cmp))
    # print(label_path_list)

    all_frames_labels = []
    print("begin load labels...")
    for label_path in label_path_list:
        # TODO: 读标签文件，并传递给track 函数
        # print("loading labels: ", label_path)
        frame_label = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                line = line.split(" ")
                # line: [class, xywh, conf] -> [xyxy, conf, class]
                xywh = np.array([float(x) for x in line[1:5]]).reshape(1, 4)
                # 将xywh转为真实数据而不是比率
                xywh[:, 0] *= WH[0]
                xywh[:, 1] *= WH[1]
                xywh[:, 2] *= WH[0]
                xywh[:, 3] *= WH[1]
                # print(xywh.shape)
                label = [int(line[0])]
                conf = [float(line[5])]
                xyxy = xywh2xyxy(xywh).reshape(4)
                # print("xywh: {}, conf: {}, label: {}".format(xywh, conf, label))
                out = np.concatenate((xyxy, conf, label), axis=0)
                frame_label.append(out)
        frame_label = np.array(frame_label)
        all_frames_labels.append(frame_label)
    print("all labels are load done!")
    test_size = (2176, 3840, 3)
    from Tracking.configs.configs import v5configs, configs
    cfg = v5configs("./Tracking/configs/tph_yolov5.yaml")
    shapes = frames_track(test_size, all_frames_labels, cfg)
    height = 2176
    width = 3840
    frames_length = len(all_frames_labels)
    out_label_path = r"./output_img/03.txt"
    save_labels(shapes, out_label_path, frames_length, width, height)
    print("labels are saved!")
    