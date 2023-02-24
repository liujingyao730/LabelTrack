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

def image_show(tra_list:List[trajectory]) -> None:

    t = len(tra_list)
    for i in range(t):
        traj = tra_list[i].traj
        segs = [np.array(tra) for tra in traj]
        

if __name__ == "__main__":

    file = "0.txt"
    trajs = load_from_file(file)
    a = 1
