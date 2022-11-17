import numpy as np
from typing import List
import matplotlib.pyplot as plt


class trajectory():

    def __init__(self, position=None, frame=1, miss_prefix=False) -> None:
        
        self.traj = []
        self.segs = []
        self.num_segs = 0
        if position:
            self.add_new_seg(frame, position, self.num_segs)
        
        self.missing_prefix = miss_prefix
        self.missing_tail = False
        self.flicker = False

    def add_position(self, frame:int, position:list) -> None:

        find = False
        for i in range(self.num_segs):
    
            if frame == self.segs[i][-1] + 1:
                self.add_into_seg(frame, position, i, is_tail=True)
                find = True
                break
            elif frame < self.segs[i][0] - 1:
                self.add_new_seg(frame, position, i)
                find = True
                break
            elif frame == self.segs[i][0] - 1:
                self.add_into_seg(frame, position, i, is_tail=False)
                find = True
                break
            elif frame >= self.segs[i][0] and frame <= self.segs[i][-1]:
                raise RuntimeError("frame " + str(frame) + " already exits")
        
        if not find:
            self.add_new_seg(frame, position, self.num_segs)

    def add_into_seg(self, frame:int, position:list, index:int, is_tail=True) -> None:

        if is_tail:
            i = len(self.segs[index])
        else:
            i = 0
        self.segs[index].insert(i, frame)
        self.traj[index].insert(i, position)

    def add_new_seg(self, frame:int, position:list, index:int) -> None:

        self.num_segs += 1
        self.segs.insert(index, [frame])
        self.traj.insert(index, [position])
        self.flicker = True
    
    def merge_seg(self) -> None:

        i = 0
        while i < self.num_segs-1:
            if self.segs[i][-1] + 1 == self.segs[i+1][0]:
                self.segs[i].extend(self.segs[i+1])
                self.segs.pop(i+1)
                self.num_segs -= 1
            elif self.segs[i][-1] >= self.segs[i+1][0]:
                raise RuntimeError("frame overlap: ")
            i += 1

def merge_trajectory(t1:trajectory, t2:trajectory) -> trajectory:

    assert t1.segs[-1][-1] <= t2.segs[0][0]

    t1.segs.extend(t2.segs)
    t1.traj.extend(t2.traj)
    t1.segs += t2.segs
    t1.merge_seg()

    t1.missing_tail = t2.missing_tail
    t1.flicker = not t1.num_segs == 1

    return t1
    
def load_from_file(file:str) -> dict:

    results = {}
    max_frame = 0

    with open(file, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = [float(i) for i in line.strip('\n').split(',')]
            frame, id, x, y, w, h, score, classid, _, _  = line
            id = int(id)
            position = [x + w / 2, y + h / 2]
            frame = max(max_frame, frame)
            if id not in results.keys():
                missing_prefix = False
                if frame > 1:
                    state = judge_position(x, y)
                    if state == "obstacle" or state == "missing":
                        missing_prefix = True
                results[id] = trajectory(position=position, frame=frame, miss_prefix=missing_prefix)
            else:
                results[id].add_position(frame, position)
    
    for t in results.values():
        position = t.traj[-1][-1]
        if t.segs[-1][-1] < max_frame and judge_position(position[0], position[1]) == "missing":
            t.missing_tail = True
    
    return results

                
def judge_position(x, y) -> str:
    
    if((x <= 40 or x >= 3800) and (y <= 40 or y >= 1880)):
        return "normal"
    elif(((x >= 1530 and x <= 1575) or (x >= 2270 or x <= 2310)) and ((y >= 605 and y <= 650) or (y >= 1540 and y <= 1585))):
        return "obstacle"
    else:
        return "missing"

def image_show(tra_dict, max_num=None) -> None:
    cnt = 0
    for traj in tra_dict.values():
        cnt += 1
        if max_num and cnt > max_num:
            break
        segs = [np.array(tra) for tra in traj.traj]
        # print ("drawing trajectory with segs: ", segs)
        for part in segs:
            plt.plot(part[:,0], part[:,1])

    plt.savefig("0.png")

    plt.show()



def fix_blink(trajs, min_distance=2) -> None:
    """fix the blinks in the whole trajectory.
    By comparing the distance from a lost tail one and a lost prefix one,
    if the distance is smaller than min_distance, then concat these two trajectory.

    Args:
        trajs (dict): the trajectories.
        min_distance (int, optional): the acceptable min distance between two trajs. Defaults to 2.
    """
    miss_prfxs = [traj for id, traj in trajs.items() if traj.missing_prefix is True]
    # miss_tail = [traj for id, traj in trajs.items() if traj.missing_tail is True]

    for _, traj in trajs.items():
        distance = []
        if traj.missing_tail is True:
            for miss_prfx in miss_prfxs:
                if miss_prfx.segs[-1][-1] <= traj.segs[-1][-1]:
                    continue
                if len(traj.traj) < 2:
                    distance.append(get_distance_from_point_to_line(miss_prfx.traj[0], traj.traj[-1], traj.traj[-1]))
                else:
                    distance.append(get_distance_from_point_to_line(miss_prfx.traj[0], traj.traj[-2], traj.traj[-1]))
            if min(distance) > min_distance:
                continue
            else:
                merge_trajectory(traj, miss_prfx[distance.index(min(distance))])
            
def get_distance_from_point_to_line(point, line_point1, line_point2):
    if line_point1 == line_point2:
        point_array = np.array(point)
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    return distance

if __name__ == "__main__":

    file = "0.txt"
    trajs = load_from_file(file)
    image_show(trajs, 50)
