import numpy as np
from typing import List
import matplotlib.pyplot as plt
import os
from scipy.optimize import linear_sum_assignment

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
            position = [x + w / 2, y + h / 2]
            max_frame = max(max_frame, frame)
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
    # elif(((x >= 1530 and x <= 1575) or (x >= 2270 or x <= 2310)) and ((y >= 605 and y <= 650) or (y >= 1540 and y <= 1585))):
    #     return "obstacle"
    else:
        return "missing"

def metric(p1:trajectory, p2:trajectory) -> float:

    x1 = p1.traj[-1][-1]
    x2 = p2.traj[0][0]
    t1 = p1.segs[-1][-1]
    t2 = p2.segs[0][0]
    time_distance = t2 - t1 if t2 > t1 else 10000

    return np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2) + time_distance

def match_traj(missing_tail_trajs:List[trajectory], missing_prefix_trajs:List[trajectory], threshold=0.0001) -> List[trajectory]:

    m, n = len(missing_tail_trajs), len(missing_prefix_trajs)
    weights = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            fr = missing_tail_trajs[i]
            to = missing_prefix_trajs[j]
            w = 1 / metric(fr, to)
            weights[i, j] = w
    
    row_ind, col_ind = linear_sum_assignment(weights)
    ans = []

    for i in range(len(col_ind)):
        if weights[row_ind[i], col_ind[i]] >= threshold:
            ans.append(merge_trajectory(missing_tail_trajs[row_ind[i]], missing_prefix_trajs[col_ind[i]]))
            missing_tail_trajs.pop(row_ind[i])
            missing_prefix_trajs.pop(col_ind[i])

    ans.extend(missing_tail_trajs)
    ans.extend(missing_prefix_trajs)

    return ans

def fix_trajectory(trajectories:List[trajectory]) -> List[trajectory]:

    miss_tail = []
    miss_prefix = []
    i = 0
    while i < len(trajectories):
        if trajectories[i].missing_tail:
            miss_tail.append(trajectories[i])
            trajectories.pop(i)
        elif trajectories[i].missing_prefix:
            miss_prefix.append(trajectories[i])
            trajectories.pop(i)
        else:
            i += 1
    
    if miss_prefix and miss_tail:
        trajectories.extend(match_traj(miss_tail, miss_prefix))
    else:
        trajectories.extend(miss_prefix)
        trajectories.extend(miss_tail)

    return trajectories



if __name__ == "__main__":

    file = "4.txt"
    trajs = load_from_file(file)
    trajs = [t for t in trajs.values()]
    trajs = fix_trajectory(trajs)
    a = 1
