#!/usr/bin/env python
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.draw import circle
from scipy.linalg import block_diag

global vel_max
global rot_vel_max

vel_max = 0.5
rot_vel_max = 0.4

#Map Handling Functions
def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)
    return im_np

def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
        map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, trajectory, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        self.traj_from_parent = trajectory
        vels = np.linspace(0.01,vel_max,3)
        rot_vels = np.linspace(-rot_vel_max,rot_vel_max,4)
        self.opts = np.array(np.meshgrid(vels, rot_vels)).T.reshape(-1, 2)
        all_zeros_index = (np.abs(self.opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.opts = np.delete(self.opts, all_zeros_index, axis=0)
        self.num_chosen = 0 # The number of times it is chosen as the closest point
        self.is_dead = False # If the node is a dead end
        return

#Path Planner
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0] # x_min
        self.bounds[1, 0] = self.map_settings_dict["origin"][1] # y_min
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = vel_max #m/s (Feel free to change!)
        self.rot_vel_max = rot_vel_max #0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, [], 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5

        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self,offx=[-2,-12],offy=[3,20]):
        #Return an [x,y] coordinate to drive the robot towards
        #print("TO DO: Sample point to drive towards")
        pt_x = np.random.uniform(low=offx[0], high=self.bounds[0,1]+offx[1], size=1)
        pt_y = np.random.uniform(low=self.bounds[1,0]+offy[0], high=offy[1], size=1)
        pt = np.array([pt_x,pt_y])
        # while self.check_if_duplicate(pt):
        #     pt_x = np.random.uniform(low=offx[0], high=self.bounds[0,1]+offx[1], size=1)
        #     pt_y = np.random.uniform(low=self.bounds[1,0]+offy[0], high=offy[1], size=1)
        #     pt = np.array([pt_x,pt_y])
        return pt

    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        #print("TO DO: Check that nodes are not duplicates")

        for node in self.nodes:
            if np.linalg.norm(node.point[0:2,:].reshape((2,1))-point.reshape((2,1))) < 0.1:
                return True

        return False

    def closest_node(self, point):
        #Returns the index of the closest node
        #print("TO DO: Implement a method to get the closest node to a sapled point")
        min_dist = float('inf')
        min_idx = -1
        d_list = []
        for i, node in enumerate(self.nodes):

            point1 = np.array([node.point[0],node.point[1]])
            point2 = np.array([point[0],point[1]])

            distance = np.linalg.norm(point1-point2)
            if distance < min_dist and not node.is_dead:
                min_dist = distance
                min_idx = i

        return min_idx

    def near_node(self, node_id):
        #Returns the index of near nodes
        threshold = 0.5
        d_list = []
        node = self.nodes[node_id]
        for i, n in enumerate(self.nodes):
            if i == node_id:
                continue
            point1 = np.array([n.point[0],n.point[1]])
            point2 = np.array([node.point[0],node.point[1]])

            distance = np.linalg.norm(point1-point2)
            if distance < threshold and not node.is_dead:
                d_list.append(i)

        return d_list

    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        # print("TO DO: Implment a method to simulate a trajectory given a sampled point")

        vel, rot_vel = self.robot_controller(node_i, point_s)

        robot_traj = self.trajectory_rollout(vel, rot_vel)

        base_theta = node_i.point[2,0]

        robot_dist = np.sqrt(np.square(robot_traj[0,:]) + np.square(robot_traj[1,:]))

        robot_traj[1,:] = robot_dist * np.sin(base_theta+robot_traj[2,:])
        robot_traj[0,:] = robot_dist * np.cos(base_theta+robot_traj[2,:])

        robot_traj[2, :] = [x - 2*np.pi if x > np.pi else x for x in robot_traj[2, :]]
        robot_traj[2, :] = [x + 2*np.pi if x < -np.pi else x for x in robot_traj[2, :]]

        global_traj = robot_traj + np.reshape(node_i.point,(3,1))

        return global_traj

    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        # print("TO DO: Implement a control scheme to drive you towards the sampled point")
        # return 0, 0

        # the idea is to make the robot turn towards the point using a rot_vel
        # while depending on the distance, we set a reasonable linear vel

        # depends on the angle differnce between the pose of the robot and the direction it needs
        # to head towards, set a reasonable rotation velocit

        min_dist = np.linalg.norm(point_s.copy()-node_i.point.copy()[0:2])

        base_theta = node_i.point[2,0].copy()

        delete_opt = []
        delete_i = None

        for i in range(node_i.opts.shape[0]):

            cur_vel, cur_rot_vel = node_i.opts[i]
            robot_traj = self.trajectory_rollout(cur_vel,cur_rot_vel)

            # line segment distance
            robot_dist = np.sqrt(np.square(robot_traj[0,:].copy()) + np.square(robot_traj[1,:].copy()))

            # line segment x, y in global frame
            robot_traj[1,:] = robot_dist * np.sin(robot_traj[2,:]+base_theta.copy())
            robot_traj[0,:] = robot_dist * np.cos(robot_traj[2,:]+base_theta.copy())

            global_traj = robot_traj + np.reshape(node_i.point,(3,1))

            R, C = self.points_to_robot_circle(global_traj[0:2,:].copy())

            if self.occupancy_map[R, C].size == 0:
                continue

            if np.min(self.occupancy_map[R, C]) <= 0:
                delete_opt.append(i)
                continue

            dist = np.linalg.norm(global_traj[0:2,-1].copy()-point_s.copy().flatten())

            if dist < min_dist:
                min_dist = dist
                vel, rot_vel = cur_vel, cur_rot_vel
                delete_i= i

        if delete_i is None:
            vel, rot_vel = 0.01, 0.01
        else:
            delete_opt.append(delete_i)
            node_i.opts = np.delete(node_i.opts,delete_opt,axis=0)

        return vel, rot_vel

    def trajectory_rollout(self, vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # print("TO DO: Implement a way to rollout the controls chosen")

        nt = self.num_substeps
        trajectory_o = np.zeros((3, nt))
        tt = np.linspace(0, self.timestep, nt)

        # obtain the new point (x,y,theta) using the unicycle model
        trajectory_o[2, :] = rot_vel * tt
        trajectory_o[0, :] = np.multiply(np.divide(vel, rot_vel), np.sin(trajectory_o[2, :]))
        trajectory_o[1, :] = np.multiply(np.divide(vel, rot_vel), 1 - np.cos(trajectory_o[2, :]))
        trajectory_o[2, :] = [x - 2*np.pi if x > np.pi else x for x in trajectory_o[2, :]]
        trajectory_o[2, :] = [x + 2*np.pi if x < -np.pi else x for x in trajectory_o[2, :]]

        return trajectory_o

    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        #print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        pt = point.copy()
        pt[0,:] = self.map_shape[0] - ((point[1,:] - self.map_settings_dict["origin"][1]) / self.map_settings_dict["resolution"])
        pt[1,:] = (point[0,:] - self.map_settings_dict["origin"][0]) / self.map_settings_dict["resolution"]

        return pt.astype(int)

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        #print("TO DO: Implement a method to get the pixel locations of the robot path")

        # obtain pixel coordinates (cell) of the points
        cell = self.point_to_cell(points.copy())
        R, C = np.array([]), np.array([])
        for r, c in cell.T:

            rr,cc = circle(r,c,self.robot_radius//self.map_settings_dict['resolution'], shape=self.map_shape)
            R = np.concatenate((R,rr))
            C = np.concatenate((C,cc))
        return R.astype(int), C.astype(int)

    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)

    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        #print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")

        dx = point_f[0] - node_i.point[0]
        dy = point_f[1] - node_i.point[1]

        # calculate rot_vel
        t = 1
        rot_vel = 2 * (np.arctan(dy/dx)) + node_i.point[2]
        if rot_vel > self.rot_vel_max:
            t = (rot_vel / self.rot_vel_max)[0]
            rot_vel = self.rot_vel_max

        vel = dx * rot_vel / np.sin(rot_vel * t)

        if vel >= self.vel_max:

            # if a larger linear vel is required, then no solution,
            # cannot connnect these two nodes
            return np.array([])

        # roll out
        nt = np.ceil(t * self.num_substeps).astype(int)
        trajectory_o = np.zeros((3, nt))
        tt = np.linspace(0, t, nt)

        # obtain the new point (x,y,theta) using the unicycle model
        trajectory_o[2, :] = rot_vel * tt
        trajectory_o[0, :] = np.multiply(np.divide(vel, rot_vel), np.sin(trajectory_o[2, :]))
        trajectory_o[1, :] = np.multiply(np.divide(vel, rot_vel), 1 - np.cos(trajectory_o[2, :]))
        trajectory_o[2, :] = [x - 2*np.pi if x > np.pi else x for x in trajectory_o[2, :]]
        trajectory_o[2, :] = [x + 2*np.pi if x < -np.pi else x for x in trajectory_o[2, :]]

        # line segment distance
        robot_traj = trajectory_o
        robot_dist = np.sqrt(np.square(robot_traj[0,:].copy()) + np.square(robot_traj[1,:].copy()))

        # line segment x, y in global frame
        robot_traj[1,:] = robot_dist * np.sin(robot_traj[2,:]+node_i.point[2])
        robot_traj[0,:] = robot_dist * np.cos(robot_traj[2,:]+node_i.point[2])

        global_traj = robot_traj + np.reshape(node_i.point,(3,1))

        if np.isnan(global_traj).any():
            return np.array([])

        return global_traj

    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle
        #print("TO DO: Implement a cost to come metric")

        # Euclidean distance
        cost = 0
        for i in range(len(trajectory_o)-1):
            cur_cost = np.linalg.norm(trajectory_o[0:2, i+1]-trajectory_o[0:2, i])
            cost += cur_cost

        return cost

    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        # print("TO DO: Update the costs of connected nodes after rewiring.")

        # ????
        self.nodes[node_id].cost += self.cost_to_come(self.nodes[node_id].traj_from_parent)
        for child_id in self.nodes[node_id].children_ids:
            self.update_children(child_id)

        return


    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work

        goal_reached = False
        n = 0
        threshold_iter = 69420

        lowest_d = 9999

        offx = [self.goal_point[0] - 7, self.goal_point[0] + 7 - self.bounds[0,1]]
        offy = [self.goal_point[1] - 7 - self.bounds[1,0], self.goal_point[1] + 7]

        if offx[0] < self.bounds[0,0]: offx[0] = self.bounds[0,0]
        if offx[1] > self.bounds[0,1]: offx[1] = self.bounds[0,1]
        if offy[0] < self.bounds[1,0]: offy[1] = self.bounds[1,0]
        if offy[1] > self.bounds[1,1]: offy[1] = self.bounds[1,1]

        while not goal_reached: #Most likely need more iterations than this to complete the map!

            #Sample map space
            point = self.sample_map_space()

            if n%200 == 0:
                point = self.goal_point
            
            elif n%205 == 0:
                point = np.array([[40.5], [-45.2]])
            
            elif n%210 == 0:
                point = np.array([[41], [-44]])

            elif lowest_d < 5 and n%50:
                point = self.sample_map_space(offx,offy)

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id], point)

            #Check for collisions
            #print("TO DO: Check for collisions and add safe points to list of nodes.")

            # get cell points of the robot circle along the simulated trajectory
            R, C = self.points_to_robot_circle(trajectory_o[0:2,:].copy())
            # print(R,C)

            n+=1

            if n%1000 == 0:
                print("#:",n)
                print("# of Nodes", len(self.nodes))

            # obstacles are False in the occupancy map
            # if there is collision
            if np.min(self.occupancy_map[R, C]) <= 0:
                continue

            # self.window.add_point(point.flatten().copy(),color=(0, 0, 255))

            # append this collision-free node to our list
            self.nodes.append(Node(trajectory_o[:,-1].reshape((3,1)).copy(),closest_node_id, trajectory_o, 0))
            self.nodes[closest_node_id].children_ids.append(len(self.nodes)-1)

            # self.window.add_point(trajectory_o[0:2,-1].copy(),color=(0, 255, 0))

            for i in range(10):
                self.window.add_point(trajectory_o[0:2,i].copy(),color=(0, 255, 0))

            if self.nodes[closest_node_id].opts.size == 0:
                print("Node is dead")
                self.nodes[closest_node_id].is_dead = True
                self.window.add_point(self.nodes[closest_node_id].point[0:2].copy().flatten(),color=(255, 0, 255))

            #Check if goal has been reached
            #print("TO DO: Check if at goal point.")
            d = np.linalg.norm(trajectory_o[0:2,-1].reshape((2,1))-self.goal_point)
            if d < lowest_d:
                lowest_d = d
                print('closest dist to goal: ', lowest_d)
            if np.linalg.norm(trajectory_o[0:2,-1].reshape((2,1))-self.goal_point) <= self.stopping_dist:
                goal_reached = True

            if n >= threshold_iter: break

        return self.nodes

    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot
        goal_reached = False
        n = 0
        threshold_iter = 69420

        lowest_d = 9999

        while not goal_reached: #Most likely need more iterations than this to complete the map!

            #Sample map space
            point = self.sample_map_space()

            if n%300 == 0:
                point = self.goal_point

            if lowest_d < 5 and n%100:
                    point = self.goal_point

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id], point)

            #Check for collisions
            #print("TO DO: Check for collisions and add safe points to list of nodes.")

            # get cell points of the robot circle along the simulated trajectory
            R, C = self.points_to_robot_circle(trajectory_o[0:2,:].copy())
            # print(R,C)

            n+=1

            if n%1000 == 0:
                print("#:",n)
                print("# of Nodes", len(self.nodes))

            # obstacles are False in the occupancy map
            # if there is collision
            if np.min(self.occupancy_map[R, C]) <= 0:
                continue

            self.window.add_point(point.flatten().copy(),color=(0, 0, 255))

            # append this collision-free node to our list
            self.nodes.append(Node(trajectory_o[:,-1].reshape((3,1)).copy(), closest_node_id, trajectory_o, self.nodes[closest_node_id].cost + self.cost_to_come(trajectory_o.copy())))
            self.nodes[closest_node_id].children_ids.append(len(self.nodes)-1)

            # find near nodes
            nn = self.near_node(len(self.nodes) -1)
            node_min_id = closest_node_id
            cost_min = self.nodes[closest_node_id].cost + self.cost_to_come(trajectory_o.copy())

            for node_id in nn:
                traj = self.connect_node_to_point(self.nodes[node_id], self.nodes[-1].point)
                if traj.size != 0:
                    # print(traj)
                    R, C = self.points_to_robot_circle(traj[0:2,:].copy())
                    if np.min(self.occupancy_map[R, C]) <= 0:
                        continue
                    if self.cost_to_come(traj.copy()) + self.nodes[node_id].cost < cost_min:
                        node_min_id = node_id
                        cost_min = self.cost_to_come(traj.copy()) + self.nodes[node_id].cost

            #Last node rewire
            # print("TO DO: Last node rewiring")
            self.nodes[-1].parent_id = node_min_id

            #Close node rewire
            # print("TO DO: Near point rewiring")
            for node_id in nn:
                traj = self.connect_node_to_point(self.nodes[-1], self.nodes[node_id].point)
                if traj.size != 0:
                    R, C = self.points_to_robot_circle(traj[0:2,:].copy())
                    if np.min(self.occupancy_map[R, C]) <= 0:
                        continue
                    if self.cost_to_come(traj.copy()) + self.nodes[-1].cost < self.nodes[node_id].cost:
                        self.nodes[node_id].parent_id = len(self.nodes) - 1
                        self.nodes[node_id].cost = self.cost_to_come(traj.copy()) + self.nodes[-1].cost
                        self.nodes[node_id].traj_from_parent = traj

            self.update_children(len(self.nodes) - 1)

            self.window.add_point(self.nodes[-1].point[0:2].copy().flatten(),color=(0, 255, 0))

            if self.nodes[closest_node_id].vels.size == 0 and self.nodes[closest_node_id].rot_vels.size == 0:
                print("Node is dead")
                self.nodes[closest_node_id].is_dead = True
                self.window.add_point(self.nodes[closest_node_id].point[0:2].copy().flatten(),color=(255, 0, 255))

            #Check if goal has been reached
            #print("TO DO: Check if at goal point.")
            d = np.linalg.norm(trajectory_o[0:2,-1].reshape((2,1))-self.goal_point)
            if d < lowest_d:
                lowest_d = d
                print('closest dist to goal: ', lowest_d)
            if np.linalg.norm(trajectory_o[0:2,-1].reshape((2,1))-self.goal_point) <= self.stopping_dist:
                goal_reached = True

            if n >= threshold_iter: break

        return self.nodes

    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
            self.window.add_point(self.nodes[current_node_id].point[0:2].copy().flatten(),color=(0, 0, 255))
        path.reverse()
        return path

def main():

    #Set map information
    map_filename = "willowgarageworld_05res.png"#"simple_map.png"#
    map_setings_filename = "willowgarageworld_05res.yaml"#"simple.yaml"#
    # map_filename = "simple_map.png"
    # map_setings_filename = "simple.yaml"
    #robot information
    goal_point = np.array([[41.7], [-44.2]]) #np.array([[30], [-20]])# np.array([[42], [-45]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)

    start = time.time()
    nodes = path_planner.rrt_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)
    
    end = time.time()
    print("Path Planning Time Elapsed:", end-start)
    time.sleep(5000)


if __name__ == '__main__':
    main()
