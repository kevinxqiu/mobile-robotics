"""

Voronoi Road Map Planner

author: Atsushi Sakai (@Atsushi_twi)

author: Celinna Ju 

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from dijkstra_search import DijkstraSearch
from scipy.spatial import cKDTree, Voronoi
import cv2



class VoronoiRoadMapPlanner:

    def __init__(self):
        # parameter
        self.N_KNN = 10.0  # number of edge from one sampled point
        self.MAX_EDGE_LEN = 20.0  # [m] Maximum edge length

    def planning(self, sx, sy, gx, gy, ox, oy, robot_radius,show_animation):
        obstacle_tree = cKDTree(np.vstack((ox, oy)).T)

        sample_x, sample_y = self.voronoi_sampling(sx, sy, gx, gy, ox, oy)
        if show_animation:  # pragma: no cover
            plt.plot(sample_x, sample_y, ".b")

        road_map_info = self.generate_road_map_info(
            sample_x, sample_y, robot_radius, obstacle_tree)

        rx, ry = DijkstraSearch(show_animation).search(sx, sy, gx, gy,
                                                       sample_x, sample_y,
                                                       road_map_info)
        return rx, ry

    def is_collision(self, sx, sy, gx, gy, rr, obstacle_kd_tree):
        x = sx
        y = sy
        dx = gx - sx
        dy = gy - sy
        yaw = math.atan2(gy - sy, gx - sx)
        d = math.hypot(dx, dy)

        if d >= self.MAX_EDGE_LEN:
            return True

        D = rr
        n_step = round(d / D)

        for i in range(n_step):
            dist, _ = obstacle_kd_tree.query([x, y])
            if dist <= rr:
                return True  # collision
            x += D * math.cos(yaw)
            y += D * math.sin(yaw)

        # goal point check
        dist, _ = obstacle_kd_tree.query([gx, gy])
        if dist <= rr:
            return True  # collision

        return False  # OK

    def generate_road_map_info(self, node_x, node_y, rr, obstacle_tree):
        """
        Road map generation

        node_x: [m] x positions of sampled points
        node_y: [m] y positions of sampled points
        rr: Robot Radius[m]
        obstacle_tree: KDTree object of obstacles
        """

        road_map = []
        n_sample = len(node_x)
        node_tree = cKDTree(np.vstack((node_x, node_y)).T)

        for (i, ix, iy) in zip(range(n_sample), node_x, node_y):

            dists, indexes = node_tree.query([ix, iy], k=n_sample)

            edge_id = []

            for ii in range(1, len(indexes)):
                nx = node_x[indexes[ii]]
                ny = node_y[indexes[ii]]

                if not self.is_collision(ix, iy, nx, ny, rr, obstacle_tree):
                    edge_id.append(indexes[ii])

                if len(edge_id) >= self.N_KNN:
                    break

            road_map.append(edge_id)

            #plot_road_map(road_map, sample_x, sample_y)

        return road_map

    @staticmethod
    def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

        for i, _ in enumerate(road_map):
            for ii in range(len(road_map[i])):
                ind = road_map[i][ii]

                plt.plot([sample_x[i], sample_x[ind]],
                         [sample_y[i], sample_y[ind]], "-k")

    @staticmethod
    def voronoi_sampling(sx, sy, gx, gy, ox, oy):
        oxy = np.vstack((ox, oy)).T

        # generate voronoi point
        vor = Voronoi(oxy)
        sample_x = [ix for [ix, _] in vor.vertices]
        sample_y = [iy for [_, iy] in vor.vertices]

        sample_x.append(sx)
        sample_y.append(sy)
        sample_x.append(gx)
        sample_y.append(gy)

        return sample_x, sample_y
    


def get_path(img,show_animation,start,goal):
    print(__file__ + " start!!")

    robot_size = 100  # [mm]
    factor = 15

    # Convert start and end positions based on new factor size
    start = np.array(start)/factor
    goal = np.array(goal)/factor
    robot_size = robot_size/factor
    
    #print(start)
    
    row, col = img.shape[:2]
    bordersize = 5
    
    border = cv2.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    
    # Map size is 1188 x 840
    X,Y = img.shape # X and Y are flipped here
    
    pixel2mmx = 840 / X
    pixel2mmy = 1188 / Y
    # pixel2mmx = 2.56
    # pixel2mmy = 2.14


    new_img = cv2.resize(border,(int(pixel2mmy*Y/factor), int(pixel2mmx*X/factor))) 
    
    ret, thresh = cv2.threshold(new_img,127,255,cv2.THRESH_BINARY_INV)
    thresh = np.rot90(thresh,k=1, axes=(1,0))
    #thresh = cv2.flip(thresh,0)
    
    coords = np.column_stack(np.where(thresh == 255))
    coords = np.transpose(coords)
    
    ox = coords[0,:].tolist()
    oy = coords[1,:].tolist()
    
    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(start[0], start[1], "^r")
        plt.plot(goal[0], goal[1], "^c")
        plt.grid(True)
        plt.axis("equal")

    rx, ry = VoronoiRoadMapPlanner().planning(start[0], start[1], goal[0], goal[1], ox, oy,
                                              robot_size, show_animation)
    
    assert rx, 'Cannot found path'

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.1)
        plt.show()
        
    # Convert scaled value back to mm values in integer format
    rx = (np.array(rx)*factor).astype(int)
    ry = (np.array(ry)*factor).astype(int)
    
    
    path = np.stack((rx,ry),axis = -1)
    return path


# if __name__ == '__main__':
#     # start and goal position
#     start = np.array([130, 700])
#     end = np.array([1020, 200])
    
#     img = 'map2.jpg'
#     gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#     path  = get_path(gray,True,start,end)
#     print(path)