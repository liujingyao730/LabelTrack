import argparse
import os
import os.path as osp
import time
import cv2
import numpy as np
import yaml
import torch

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

# TODO
import sys
sys.path.append("../../")
from PyQt5.QtCore import *
from GUI.shape import Shape
from GUI.utils import *



IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--source", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False,
        yoloid=0
    ):
        if yoloid == 0:
            self.model = model
            self.decoder = decoder
            self.num_classes = exp.num_classes
            self.confthre = exp.test_conf
            self.nmsthre = exp.nmsthre
            # nmsthre与iou_thres是同样的含义，分别是batched_nms(yolox) 和 nms(yolov5)的参数
            self.test_size = exp.test_size
            self.device = device
            self.fp16 = fp16
            self.yoloid = yoloid
            self.rgb_means = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
            if trt_file is not None:
                from torch2trt import TRTModule
                model_trt = TRTModule()
                model_trt.load_state_dict(torch.load(trt_file))

                x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
                self.model(x)
                self.model = model_trt
        elif yoloid == 1:
            self.model = model
            self.test_size = exp.imgsz
            self.fp16 = fp16
            self.rgb_means = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
            self.cfg = exp
            self.yoloid = yoloid

    # 该函数由tph_yolov5下的detect.py修改得到
    # 去除了保存、可视化所需的参数以及tensorflow的部分
    # 如要添加更多功能请去detect.py寻找相关代码
    @torch.no_grad()
    def v5inference(self,
                    model,
                    image,
                    imgsz=640,  # inference size (pixels)
                    conf_thres=0.25,  # confidence threshold
                    iou_thres=0.45,  # NMS IOU threshold
                    max_det=1000,  # maximum detections per image
                    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                    classes=None,  # filter by class: --class 0, or --class 0 2 3
                    agnostic_nms=False,  # class-agnostic NMS
                    augment=False,  # augmented inference
                    half=False,  # use FP16 half-precision inference
        ):
        import numpy as np
        from tph_yolov5.utils.augmentations import letterbox
        from tph_yolov5.utils.general import non_max_suppression, scale_coords
        from PIL import Image
        # Initialize 和 Load model 在 trackworker.py
        stride = int(model.stride.max())
        img0 = image
        img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # Run inference
        dt, seen = [0.0, 0.0, 0.0], 0
        img = torch.from_numpy(img).to(device)
        # print("shape of img0: {}".format(img.shape))
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        # print("shape of img1: {}".format(img.shape))
        pred = model(img, augment=augment)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0 = img0.copy()

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                pred[i] = det
        return pred

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        if self.fp16:
            img = img.half()  # to FP16

        if self.yoloid == 1:
            outputs = self.v5inference(model=self.model, image=img,
                                       imgsz=self.cfg.imgsz, device=self.cfg.device)
            # 注意原来的代码返回的类型是numpy类型
            outputs = [x.cpu().numpy() for x in outputs]

        else:
            img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
            img_info["ratio"] = ratio
            img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                timer.tic()
                outputs = self.model(img)
                if self.decoder is not None:
                    outputs = self.decoder(outputs, dtype=outputs.type())
                # outputs - list of detections(x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                #logger.info("Infer time: {:.4f}s".format(time.time() - t0))

        return outputs, img_info


def is_in_filter_area(bbox, filter_area):
    """
    判断bbox是否在filter_area中
    Args:
        bbox: (x1, y1, x2, y2)
        filter_area: (t, l, b, r)

    Returns: True if in filter area, else False

    """
    x1, y1, x2, y2 = bbox
    for area in filter_area:
        ft, fl, fb, fr = area
        if x1 > fl and x2 < fr and y1 > ft and y2 < fb:
            # print("bbox: {}, area: {}".format(bbox, area))
            return True
    return False


def frames_track(test_size, predictor, img_list, config, signal, canvas):
    tracker = BYTETracker(config, frame_rate=config.fps)
    results = []
    resultImg = []
    timer = Timer()
    detectPos = None
    #statusbar.showMessage()
    filter_area = None
    if canvas.isFilter:
        # 将canvas中的filter_shapes转换为（t, l, b, r）的形式，并存入filter_area列表
        filter_area = []
        for area in canvas.filter_shapes:
            # print("area[0]: {}".format(area[0]))
            # print("x, y: {}, {}".format(area[0].x(), area[0].y()))
            l, t = area[0].x(), area[0].y()
            r, b = area[2].x(), area[2].y()
            filter_area.append([t, l, b, r])
            # print("t: {}, l: {}, b: {}, r:{}".format(t, l, b, r))

    for frame_id, img in enumerate(img_list, 1):
        outputs, img_info = predictor.inference(img, timer)
        # outputs: [1, dets, 6:{x1, y1, x2, y2, conf, cls_pred}], type: numpy.array

        # filter area without detection
        if canvas.isFilter:
            # 删除outputs中在filter_areas中的bbox
            delete_index = []
            for i, output in enumerate(outputs[0]):
                if filter_area is not None:
                    if is_in_filter_area(output[:4], filter_area):
                        delete_index.append(i)
            for i in sorted(delete_index, reverse=True):
                # print("delete bbox: {}".format(outputs[0][i]))
                outputs[0] = np.delete(outputs[0], i, axis=0)

        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], test_size)
            T3 = time.time()
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                # detectPos = Shape()
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
                    canvas.update_shape(tid, frame_id, cid, tlwh, t.score, 'A')
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
            resultImg.append(online_im)
        else:
            timer.toc()
            online_im = img_info['raw_img']

        if frame_id % 5 == 0:
            signal.emit("已处理帧数 {} / {} ({:.2f} fps)".format(frame_id + 1, len(img_list), 1. / max(1e-5, timer.average_time)))
            canvas.numFrames = len(resultImg)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            
    canvas.numFrames = len(resultImg)
    signal.emit("所有图片帧已处理完毕")

    return resultImg
        

def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.source == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.source == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            #print(outputs)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)
            # ch = cv2.waitKey(1)
            # if ch == 27 or ch == ord("q") or ch == ord("Q"):
            #     break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.source == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.source == "video" or args.source == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
