from tph_yolov5 import detect
from multiprocessing import Pool, Manager
import os



def detect_video(args):
    lock = args[0]
    video_path = args[1]
    print(video_path)
    lock.acquire()    
    weights = r"./weights/yolov5l-xs-1.pt"
    imgsz = (3840, 3840)
    device = "0"
    project = "detect/03"
    name = "test"
    nosave = True
    save_txt = True
    save_conf = True
    save_crop = True
    detect.run(weights=weights, source=video_path, imgsz=imgsz,
            conf_thres=0.25, iou_thres=0.3, device=device,
            save_txt=save_txt, save_crop=save_crop, save_conf=save_conf,
            nosave=nosave, project=project, name=name)
    lock.release()


if __name__ == "__main__":
    videos_dir = "D:\\project\\cut_video\\03"
    lock = Manager().Lock()
    for root, dirs, files in os.walk(videos_dir):
        video_paths = [[lock, os.path.join(root, file)] for file in files]
    
    with Pool(1) as p:
        p.map(detect_video, video_paths)
