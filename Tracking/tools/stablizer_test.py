import time
import cv2
import numpy as np


class Stable:
    __video_path = None

    __surf = {
        'surf': None,   # the surf keypoint detector itself
        'kp': None,     # the keypoints in the first frame
        'des': None,    # the descriptor of the points in 'kp'
        'template_kp': None  # first and the last frame keypoint matches
    }

    # capture
    __capture = {
        'cap': None,    # the video capturer
        'size': None,   # the video frame size
        'frame_count': None,  # the number of the frame in the video
        'fps': None,    # the fps of the input video
        'video': None   # the output videowriter
    }

    __config = {
        # if the count is decreased, the time will not be decrease.
        # detect 5000 keypoints in one frame
        'key_point_count': 5000,
        'index_params': dict(algorithm=0, trees=5),
        'search_params': dict(checks=50),
        # the quality of the matching of keypoints. range(0.0, 1.0)
        # bigger ratio, quality is better
        'ratio': 0.5,
        'frame_count': 20000  # frame number limit of the output video
    }

    __current_frame = 0
    __handle_count = 0
    __handle_timer = {
        'init': 0,
        'handle': 0,
        'read': 0,
        'key': 0,
        'matrix': 0,
        'flann': 0,
        'perspective': 0,
        'write': 0,
        'other': 0,
    }

    def __init_capture(self):
        self.__capture['cap'] = cv2.VideoCapture(self.__video_path)
        self.__capture['size'] = (int(self.__capture['cap'].get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(self.__capture['cap'].get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.__capture['fps'] = self.__capture['cap'].get(cv2.CAP_PROP_FPS)

        self.__capture['video'] = cv2.VideoWriter(self.__video_path.replace('.', '_stable.'),
                                                  cv2.VideoWriter_fourcc(*"mp4v"),
                                                  self.__capture['fps'],
                                                  self.__capture['size'])

        self.__capture['frame_count'] = int(self.__capture['cap'].get(cv2.CAP_PROP_FRAME_COUNT))

        self.__handle_count = min(self.__config['frame_count'], self.__capture['frame_count'])

    def __init_surf(self):

        st = time.time()
        self.__capture['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
        state, first_frame = self.__capture['cap'].read()

        self.__capture['cap'].set(cv2.CAP_PROP_POS_FRAMES, self.__capture['frame_count'] - 20)
        state, last_frame = self.__capture['cap'].read()

        # SURF is abondoned because it need specified opencv version,
        # which version is conflict with the pyqt5 environment.
        # self.__surf['surf'] = cv2.xfeatures2d.SURF_create(
        #     self.__config['key_point_count'], 1, 1, 1, 1)
        self.__surf['surf'] = cv2.SIFT_create(self.__config['key_point_count'])
        self.__surf['kp'], self.__surf['des'] = self.__surf['surf'].detectAndCompute(
            first_frame, None)
        kp, des = self.__surf['surf'].detectAndCompute(last_frame, None)

        flann = cv2.FlannBasedMatcher(self.__config['index_params'], self.__config['search_params'])
        matches = flann.knnMatch(self.__surf['des'], des, k=2)

        good_match = []
        for m, n in matches:
            if m.distance < self.__config['ratio'] * n.distance:
                good_match.append(m)

        self.__surf['template_kp'] = []
        for f in good_match:
            self.__surf['template_kp'].append(self.__surf['kp'][f.queryIdx])

        self.__capture['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.__handle_timer['init'] = int((time.time() - st) * 1000)

        print("[INFO] init time:{}ms".format(self.__handle_timer['init']))

    # def __init_data(self):
    #     __frame_queue = None
    #     __write_frame_queue = None
    #     __surf_list = []

    def __init(self):
        self.__init_capture()
        self.__init_surf()
        # self.__init_data()

    def __process(self):

        self.__current_frame = 1

        while True:

            if self.__current_frame > self.__handle_count:
                break

            start_time = time.time()

            success, frame = self.__capture['cap'].read()
            self.__handle_timer['read'] = int((time.time() - start_time) * 1000)

            if not success:
                return

            frame = self.detect_compute(frame)

            st = time.time()
            self.__capture['video'].write(frame)
            self.__handle_timer['write'] = int((time.time() - st) * 1000)

            self.__handle_timer['handle'] = int((time.time() - start_time) * 1000)

            self.__current_frame += 1

            self.print_handle_time()

    def stable(self, path):
        self.__video_path = path
        self.__init()
        self.__process()

    def print_handle_time(self):
        print(
            "[INFO] handle frame:{}/{} time:{}ms(read:{}ms key:{}ms flann:{}ms matrix:{}ms perspective:{}ms write:{}ms)".
            format(self.__current_frame,
                   self.__handle_count,
                   self.__handle_timer['handle'],
                   self.__handle_timer['read'],
                   self.__handle_timer['key'],
                   self.__handle_timer['flann'],
                   self.__handle_timer['matrix'],
                   self.__handle_timer['perspective'],
                   self.__handle_timer['write']))

    def detect_compute(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        st = time.time()
        kp, des = self.__surf['surf'].detectAndCompute(frame_gray, None)
        self.__handle_timer['key'] = int((time.time() - st) * 1000)

        st = time.time()
        flann = cv2.FlannBasedMatcher(self.__config['index_params'], self.__config['search_params'])
        matches = flann.knnMatch(self.__surf['des'], des, k=2)
        self.__handle_timer['flann'] = int((time.time() - st) * 1000)

        st = time.time()
        good_match = []
        for m, n in matches:
            if m.distance < self.__config['ratio'] * n.distance:
                good_match.append(m)

        p1, p2 = [], []
        for f in good_match:
            if self.__surf['kp'][f.queryIdx] in self.__surf['template_kp']:
                p1.append(self.__surf['kp'][f.queryIdx].pt)
                p2.append(kp[f.trainIdx].pt)

        H, _ = cv2.findHomography(np.float32(p2), np.float32(p1), cv2.RHO)
        self.__handle_timer['matrix'] = int((time.time() - st) * 1000)

        st = time.time()
        output_frame = cv2.warpPerspective(
            frame, H, self.__capture['size'], borderMode=cv2.BORDER_REPLICATE)
        self.__handle_timer['perspective'] = int((time.time() - st) * 1000)

        return output_frame


if __name__ == '__main__':
    s = Stable()
    s.stable('D:/BHK/Dataset/dataset_2018TVCG/myvideo/DJI_0002.mp4')
