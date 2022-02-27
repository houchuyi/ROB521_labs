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
    def __init__(self, point, parent_id, cost):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
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
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[0] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[1] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.3 #m/s (Feel free to change!)
        self.rot_vel_max = 3.14 #0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]
        # self.nodes = [Node(np.array([[0], [20], [0]]), -1, 0)]

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
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        #print("TO DO: Sample point to drive towards")

        pt = np.zeros((2,1))
        pt[0] = np.random.rand(1)[0]
        pt[1] = np.random.rand(1)[0]

        pt[0] = pt[0] * (self.bounds[0,1] - self.bounds[0,0]) + self.bounds[0,0]
        pt[1] = pt[1] * (self.bounds[1,1] - self.bounds[1,0]) + self.bounds[1,0]
    
        while self.check_if_duplicate(pt):
            pt[0] = np.random.rand(1)[0]
            pt[1] = np.random.rand(1)[0]
            pt[0] = pt[0] * (self.bounds[0,1] - self.bounds[0,0]) + self.bounds[0,0]
            pt[1] = pt[1] * (self.bounds[1,1] - self.bounds[1,0]) + self.bounds[1,0]

        return pt
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        #print("TO DO: Check that nodes are not duplicates")
        
        for node in self.nodes:
            if np.linalg.norm(node.point[0:2]-point) < 0.2:
                return True
        
        return False
    
    def closest_node(self, point):
        #Returns the index of the closest node
        #print("TO DO: Implement a method to get the closest node to a sapled point")
        min_dist = float('inf')
        min_idx = -1
        d_list = []
        for i, node in enumerate(self.nodes):
            distance = np.linalg.norm(node.point[0:2]-point)
            if distance < min_dist:
                min_dist = distance
                min_idx = i

        return min_idx
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        # print("TO DO: Implment a method to simulate a trajectory given a sampled point")
        vel, rot_vel = self.robot_controller(node_i, point_s)

        robot_traj = self.trajectory_rollout(vel, rot_vel)
        global_traj = robot_traj + np.reshape(node_i,(3,1))

        global_traj[2, :] = [x - 2*np.pi if x > np.pi else x for x in global_traj[2, :]]
        global_traj[2, :] = [x + 2*np.pi if x < -np.pi else x for x in global_traj[2, :]]
        return global_traj
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        # print("TO DO: Implement a control scheme to drive you towards the sampled point")
        # return 0, 0

        # the idea is to make the robot turn towards the point using a rot_vel
        # while depending on the distance, we set a reasonable linear vel

        # depends on the angle differnce between the pose of the robot and the direction it needs
        # to head towards, set a reasonable rotation velocity
        
        dist = np.linalg.norm(point_s.reshape((2,1)) - node_i[0:2].reshape((2,1)))
        angle = np.arctan2(point_s[1], point_s[0])[0] - node_i[2, 0]
        if angle > np.pi:
            angle = angle - 2 * np.pi
        elif angle < -np.pi:
            angle = angle + 2 * np.pi
        dx = dist * np.cos(angle)
        dy = dist * np.sin(angle)
        # print(dist, angle, dx, dy)


        vel = self.vel_max
        rot_vel_t = 2 * np.arctan(dy/dx)
        
        t = dx * rot_vel_t / np.sin(rot_vel_t) / vel
        rot_vel = rot_vel_t / t
        
        
        # dx = point_s[0] - node_i[0][0]
        # dy = point_s[1] - node_i[1][0]
        # theta_i_s = np.arctan2(dy,dx)[0]

        # theta = node_i[2][0] # pose of the robot
        # angle_threshold = np.pi

        # angle_difference = theta - theta_i_s
        
        # # theta needs to be within -pi and pi for this to work. If not, we should change this
        # if angle_difference < 0:
        #     if np.abs(angle_difference) >= angle_threshold:
        #         angle_difference = (2 * np.pi + angle_difference)
        #     rot_vel = self.rot_vel_max * (angle_difference / angle_threshold)

        # elif angle_difference > 0:
        #     if np.abs(angle_difference) >= angle_threshold:
        #         angle_difference = -2 * np.pi + angle_difference
        #     rot_vel = - self.rot_vel_max * (angle_difference / angle_threshold)

        # elif angle_difference == 0:
        #     rot_vel = 0

        # # depends on the distance, we set a reasonable linear velocity
        # dist_threshold = self.vel_max # metres
        # dist = np.sqrt(np.square(dx)+np.square(dy))[0]
        # if dist >= dist_threshold:
        #     vel = self.vel_max
        # else:
        #     vel = self.vel_max * dist / dist_threshold

        # determine the direction of rotation
        # the idea is to try a positive rotation vel first and see
        # whether the result position is farther or closer
        # if farther, then we know the rot vel shold actually be negative
        # if closer, then we are good
        # theta = np.multiply(rot_vel, self.timestep)
        # x = np.multiply(np.divide(vel, rot_vel), np.sin(theta)) 
        # y = np.multiply(np.divide(vel, rot_vel), 1 - np.cos(theta))

        # dx = point_s[0] - x
        # dy = point_s[1] - y
        # new_dist = np.sqrt(np.square(dx)+np.square(dy))

        # if new_dist > dist:
        #     rot_vel = -rot_vel

        # note it is possible that new_dist = dist
        # because the vel and rot vel can result a certain curvature
        # basically, the simulated new position and the old position
        # are on the same arc. In this case, the direction of the rotation
        # is actually correct. Hence, we are good
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
        pt[0,:] = pt[0,:] - self.map_settings_dict["origin"][0]
        pt[1,:] = self.map_shape[1] * self.map_settings_dict["resolution"] - pt[1,:] + self.map_settings_dict["origin"][1]
        
        pt = pt // self.map_settings_dict['resolution']

        return pt

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        #print("TO DO: Implement a method to get the pixel locations of the robot path")

        # obtain pixel coordinates (cell) of the points
        cell = self.point_to_cell(points.copy())  
        R, C = np.array([]), np.array([])
        for r,c in cell.T:

            rr,cc = circle(r,c,self.robot_radius/self.map_settings_dict['resolution'], shape=self.map_shape)
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
        
        dx = point_f[0] - node_i[0]
        dy = point_f[1] - node_i[1]

        # calculate rot_vel
        t = 1
        rot_vel = 2 * np.arctan(dy/dx)
        if rot_vel > rot_vel_max:
            t = rot_vel / rot_vel_max
            rot_vel = rot_vel_max
        
        vel = dx * rot_vel / np.sin(rot_vel * t)

        # roll out
        nt = np.ceil(t * self.num_substeps)
        trajectory_o = np.zeros((3, nt))
        tt = np.linspace(0, t, nt)

        # obtain the new point (x,y,theta) using the unicycle model
        trajectory_o[2, :] = rot_vel * tt
        trajectory_o[0, :] = np.multiply(np.divide(vel, rot_vel), np.sin(trajectory_o[2, :])) 
        trajectory_o[1, :] = np.multiply(np.divide(vel, rot_vel), 1 - np.cos(trajectory_o[2, :]))
        trajectory_o[2, :] = [x - 2*np.pi if x > np.pi else x for x in trajectory_o[2, :]]
        trajectory_o[2, :] = [x + 2*np.pi if x < -np.pi else x for x in trajectory_o[2, :]]

        # check collision
        R, C = self.points_to_robot_circle(trajectory_o[0:2,:].copy())

        # if collision free
        if all(self.occupancy_map[R,C]):
            return trajectory_o
        else:
            return False
    
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



        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
    
        goal_reached = False
        n = 0
        threshold_iter = 69420

        for i in range(0, self.map_shape[0]):
            for j in range(0, self.map_shape[1]):
                if np.flipud(self.occupancy_map)[j,i] == 0:
                    x = self.map_settings_dict["origin"][0] + i * self.map_settings_dict["resolution"]
                    y = self.map_settings_dict["origin"][1] + j * self.map_settings_dict["resolution"]
                    self.window.add_point([x, y],color=(255, 0, 0))


        while not goal_reached: #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()
            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)
            # print(self.nodes[closest_node_id].point, point)
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

            # if min(np.flipud(self.occupancy_map.T)[R,C]) == 0: 
            if min(self.occupancy_map.T[R,C]) == 0: 
                continue 

            if self.check_if_duplicate(trajectory_o[0:2,-1].reshape((2,1))):
                continue

            # append this collision-free node to our list
            self.nodes.append(Node(trajectory_o[:,-1].reshape((3,1)),closest_node_id,0))
            # print(trajectory_o[0:3,-1])

            self.nodes[closest_node_id].children_ids.append(len(self.nodes)-1)
            
            self.window.add_point(trajectory_o[0:2,-1],color=(0, 255, 0))

            #Check if goal has been reached
            #print("TO DO: Check if at goal point.")
            # print(trajectory_o[0:2,-1].reshape((2,1)), self.goal_point)
            # print(np.linalg.norm(trajectory_o[0:2,-1].reshape((2,1))-self.goal_point))
            if np.linalg.norm(trajectory_o[0:2,-1].reshape((2,1))-self.goal_point) <= self.stopping_dist:
                goal_reached = True

            if n >= threshold_iter: break 

            
        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"#"simple_map.png"#
    map_setings_filename = "willowgarageworld_05res.yaml"#"simple.yaml"#
    # map_filename = "simple_map.png"
    # map_setings_filename = "simple.yaml"
    #robot information
    goal_point = np.array([[10], [10]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()