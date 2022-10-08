import time
import cv2
import numpy as np

# TODO: refactor this small tool to increase the efficiency


class Stable:
    __video_path = None

    __surf = {
        # TODO: SURF is little bit too old for this project
        'surf': None,
        'kp': None,
        'des': None,
        'template_kp': None
    }

    # capture
    __capture = {
        'cap': None,
        'size': None,
        'frame_count': None,
        'fps': None,
        'video': None
    }

    __config = {
        # TODO: if the count is decreased, the time will not be decrease.
        'key_point_count': 5000,
        'index_params': dict(algorithm=0, trees=5),
        'search_params': dict(checks=50),
        'ratio': 0.5,
        'frame_count': 20000
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

    def __init__(self, video_path=None) -> None:
        if video_path is not None:
            self.__video_path = video_path
        self.__init_capture()
        self.__init_surf()

    def __init_capture(self):
        self.__capture['cap'] = cv2.VideoCapture(self.__video_path)
        self.__capture['size'] = (int(self.__capture['cap'].get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  int(self.__capture['cap'].get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.__capture['fps'] = self.__capture['cap'].get(cv2.CAP_PROP_FPS)

        # self.__capture['video'] = cv2.VideoWriter(self.__video_path.replace('.', '_stable.'),
        #                                           cv2.VideoWriter_fourcc(*"mp4v"),
        #                                           self.__capture['fps'],
        #                                           self.__capture['size'])

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

    def __init_data(self):
        __frame_queue = None

        __write_frame_queue = None

        __surf_list = []

    def __init(self):
        self.__init_capture()
        self.__init_surf()
        self.__init_data()

    def __process(self):

        self.__current_frame = 1

        while True:

            if self.__current_frame > self.__handle_count:
                break

            start_time = time.time()

            # 抽帧
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

    def detect_perspective(self, frame):
        """input the current frame, and calculate the projective transformation
        matrix for 2D images.

        Args:
            frame (cv2.ndarray): the frame to be compared.

        Return:
            H (numpy.ndarray): homography matrix, (3x3).
        """
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

        self.print_handle_time()

        return H

    def detect_perspective_from_id(self, frame_id):
        """input the current frame, and calculate the projective transformation
            matrix for 2D images.

            Args:
                frame_id (int): the id of frame to be compared with. starts from 1.

            Return:
                H (numpy.ndarray): homography matrix, (3x3).
            """
        # TODO: find  out if it starts from 1.
        self.__capture['cap'].set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        state, current_frame = self.__capture['cap'].read()
        frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

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

        self.print_handle_time()

        return H


if __name__ == '__main__':
    s = Stable()
    s.stable('C:/Users/zying/Desktop/03.mp4')
