import numpy as np
import heapq
import matplotlib.pyplot as plt
import copy
import os
import imageio
import shutil
import matplotlib.pyplot as plt
from matplotlib import cm
import time

#visualization lib
from vedo import *
from vedo import Text, Cube, Line, Grid, merge, show, Sphere
import natsort
from PIL import Image
import imageio

class PriorityQueue:
    def __init__(self):
        self.elements =[]

    def empty(self):
        return not self.elements

    def put(self, coordination, priority):
        heapq.heappush(self.elements, (priority, coordination))

    def get(self,):
        return heapq.heappop(self.elements)[1]

class astar_3d_algo:
    def __init__(self,mapsize,start_point,end_point,wall_point,cost_map):
        self.map_size=mapsize
        self.start_point = start_point
        self.end_point = end_point
        self.wall_point = wall_point
        self.cost_map = cost_map

    def get_neighbors(self,current):
        neighbors = set()
        # Down
        if current[0]+1<self.map_size[0]:
            neighbors.add((current[0]+1,current[1], current[2]))
        # Left
        if current[1]-1>=0:
            neighbors.add((current[0],current[1]-1, current[2]))
        # Up
        if current[0]-1>=0:
            neighbors.add((current[0]-1,current[1], current[2]))
        # Right
        if current[1]+1<self.map_size[1]:
            neighbors.add((current[0],current[1]+1, current[2]))
        # Z-Up
        if current[2]+1<self.map_size[2]:
            neighbors.add((current[0], current[1], current[2]+1))
        # Z-Down
        if current[2]-1>=0:
            neighbors.add((current[0], current[1], current[2]-1))

        return neighbors

    def get_distance(self, current, end_point):
        distance = np.sqrt(np.abs(current[0]-end_point[0])**2 + np.abs(current[1]-end_point[1])**2 + np.abs(current[2]-end_point[2])**2)
        return distance

    def find_path(self,):
        frontier = PriorityQueue()
        frontier.put(self.start_point,0)

        came_from = dict()
        came_from[self.start_point]=None

        cost_value = dict()
        cost_value[self.start_point]=0


        st = time.time()
        while not frontier.empty():
            current = frontier.get()
            if current == self.end_point:
                break
            neighbors = self.get_neighbors(current)
            for next in neighbors:
                new_cost = cost_value[current] + self.cost_map[next]
                if next not in cost_value or new_cost < cost_value[next]:
                    cost_value[next] = new_cost 
                    # Dijkstra's Algorithm
                    # priority = new_cost
                    # A* Algorithm
                    priority = new_cost + self.get_distance(next, self.end_point)
                    frontier.put(next,priority)
                    came_from[next]=current
        et = time.time()
        print((et-st))

        # Reconstruction path --> (Backwards from the goal to the start)
        current = self.end_point
        path = []
        while current != self.start_point:
            path.append(current)
            current = came_from[current]
        path.append(self.start_point)
        path.reverse()

    def visualization_3d(self,path,start,end,wall):
        start = list(start)
        end = list(end)


        #Em vuong cai cho tao world(construct grid)
        world = Volume((0,0,0), self.map_size,c='black').scale([1,1,1]).lineWidth(0.1)
        
        # Walls
        pts_wall, cubes_wall = [], []
        [cubes_wall.append(Cube([pos[0]+.5, pos[1]+.5, pos[2]+.5])) for pos in wall]
        walls = merge(cubes_wall).clean().flat().color('slategray').lineWidth(0.1)

        # Start / End
        pts_start_end, cubes_start_end = [], []
        cubes_start_end.append(Cube([start[0]+.5, start[1]+.5, start[2]+.5]))
        cubes_start_end.append(Cube([end[0]+.5, end[1]+.5, end[2]+.5]))
        start_end = merge(cubes_start_end).clean().flat().color([128,0,0])

        show(world,walls,start_end, azimuth=15/len(path), elevation=15/len(path),axes=0,zoom=1.2,interactive=False)
        io.screenshot('./PROBLEM.png')
        show(world,start_end, azimuth=15/len(path), elevation=15/len(path),axes=0,zoom=1.2,interactive=False)
        io.screenshot('./INITIAL.png')

        # Paths
        # Make cubes for 3d visualization: the wall
        if not os.path.isdir('temp'):
            os.mkdir('temp')
        pts_path, cubes_path = [], []
        for i,pos in enumerate(path):
            cubes_path.append(Cube([pos[0]+.5, pos[1]+.5, pos[2]+.5]))

            paths = merge(cubes_path).clean().flat().color([128,0,0])

            show(world,walls,paths,start_end, azimuth=15/len(path), elevation=15/len(path),axes=0,zoom=1.2,interactive=False)
            io.screenshot('./temp/' + str(i) + '.png')
        # Gif
        img_dir = ['./temp/' + i for i in natsort.natsorted(os.listdir('./temp/'))]
        img_dirs = [Image.open(i) for i in img_dir]
        imageio.mimsave('./result.gif', img_dirs, fps=15)

        shutil.rmtree('./temp')

        # Paths
        # Make cubes for 3d visualization: the wall
        if not os.path.isdir('temp'):
            os.mkdir('temp')
        pts_path, cubes_path = [], []
        for i,pos in enumerate(path):
            cubes_path.append(Cube([pos[0]+.5, pos[1]+.5, pos[2]+.5]))

            paths = merge(cubes_path).clean().flat().color([128,0,0])

            show(world,paths,start_end, azimuth=15/len(path), elevation=15/len(path),axes=0,zoom=1.2,interactive=False)
            io.screenshot('./temp/' + str(i) + '.png')
        # Gif
        img_dir = ['./temp/' + i for i in natsort.natsorted(os.listdir('./temp/'))]
        img_dirs = [Image.open(i) for i in img_dir]
        imageio.mimsave('./result_PATH.gif', img_dirs, fps=15)

        shutil.rmtree('./temp')
        

        

if __name__ == '__main__':
    # A* Algorithm
    # Define Map
    start_point = (0,0,0)
    end_point = (22,29,29)
    map_size = (30,30,30)
    map = np.zeros((map_size[0], map_size[1], map_size[2]))

    # Generate wall
    wall_ratio=0.7
    max_x, max_y, max_z = int(map_size[0]),int(map_size[1]),int(map_size[2])
    wall_point = [[np.random.randint(0,max_x,1)[0],np.random.randint(0,max_y,1)[0],np.random.randint(0,max_z,1)[0]] for _ in range(int(max_x*max_y*max_z*wall_ratio))]
    for pos in wall_point:
        map[pos[0], pos[1], pos[2]]=255

    # A* Algorithm
    path_finder = astar_3d_algo(map_size,start_point,end_point,wall_point,map)
    path = path_finder.find_path()

