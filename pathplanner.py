import math
import numpy as np
from abc import abstractmethod, ABC
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim
import heapq
from io import StringIO
import random
import re
import time

class MathHelper:
    """
    An utility class without instances, just here to help with maths
    """
    @staticmethod
    def angle_diff(a1: float, a2: float) -> float:
        """
        Compute the difference between two angles.
        
        Parameters:
        ----------
        a1 (float): an angle in [-pi, pi]
        a2 (float): an angle in [-pi, pi]
        ----------
        
        Returns:
        ----------
        a2 - a1 (float): an angle in [-pi, pi]
        ----------
        """
        assert (a1 <= math.pi and -math.pi <= a1)
        assert (a2 <= math.pi and -math.pi <= a2)
        return (((a2 - a1) + math.pi) % (2 * math.pi)) - math.pi

    @staticmethod
    def circleSegmentIntersection(p1, p2, r: float) -> list:
        """
        Compute the list of points that belong to the intersection of circle centered at (0,0) and a segment.
        
        Parameters:
        ----------
        p1: the first extremity (x1,y1) of the segment
        p2: the second extremity (x2,y2) of the segment 
        r (float): the radius of a circle centered at (0,0)
        ----------
        
        Returns:
        ----------
        _ (list): a list that contains zero, one, or two points, corresponding to the intersection between the segment and the circle.
        ----------
        """
        assert r >= 0
        x1, x2 = p1[0], p2[0]
        y1, y2 = p1[1], p2[1]

        dx, dy = x2 - x1, y2 - y1
        dr2 = dx * dx + dy * dy
        D = x1 * y2 - x2 * y1

        # the first element is the point within segment
        d1 = x1 * x1 + y1 * y1
        d2 = x2 * x2 + y2 * y2
        dd = d2 - d1

        if r * r * dr2 - D * D >= 0:
            delta = math.sqrt(r * r * dr2 - D * D)
            if (delta == 0):
                return [(D * dy / dr2, -D * dx / dr2)]
            else:
                return [
                    ((D * dy + math.copysign(1.0, dd) * dx * delta) / dr2, (-D * dx + math.copysign(1.0, dd) * dy * delta) / dr2),
                    ((D * dy - math.copysign(1.0, dd) * dx * delta) / dr2, (-D * dx - math.copysign(1.0, dd) * dy * delta) / dr2)
                ]
        else:
            return []#problem here, no goal returned, will lead to an error
        
    @staticmethod
    def circleSegmentIntersectionInitial(p1, p2, c, r: float) -> list:
        """
        Compute the list of points that belong to the intersection of circle centered at (0,0) and a segment.
        
        Parameters:
        ----------
        p1: the first extremity (x1,y1) of the segment
        p2: the second extremity (x2,y2) of the segment 
        c: the center of the circle
        r (float): the radius of the circle
        ----------
        
        Returns:
        ----------
        _ (list): a list that contains zero, one, or two points, corresponding to the intersection between the segment and the circle.
        ----------
        """
        assert r >= 0
        q1 = np.array(p1) - np.array(c)
        q2 = np.array(p2) - np.array(c)
        sol = MathHelper.circleSegmentIntersection(q1,q2,r)
        intersections = []
        for i in sol:
            intersections.append(np.array(i) + np.array(c))
        return intersections        

    @staticmethod
    def get_cartesian_coord(v) -> np.ndarray:
        """
        Compute the cartesian coordinate of a vector given in polar coordinates.
        
        Parameters:
        ----------
        v: coordinates of the form (norm, angle)
        ----------
        
        Returns:
        ----------
        _ (np.ndarray): coordinates of the form (x, y).
        ----------
        """
        assert (v[0]>=0 and v[1]<= math.pi and -math.pi<= v[1])
        x = v[0]*math.cos(v[1])
        y = v[0]*math.sin(v[1])
        return np.array((x,y))

    @staticmethod
    def get_polar_coord(v) -> np.ndarray:
        """
        Compute the cartesian coordinate of a vector given in polar coordinates.
        
        Parameters:
        ----------
        v: coordinates of the form (x, y)
        ----------
        
        Returns:
        ----------
        _ (np.ndarray): coordinates of the form (norm, angle).
        ----------
        """
        v_norm = math.sqrt(v[0] ** 2 + v[1] ** 2)
        if v_norm == 0:
            return np.array((v_norm,0))
        v_angle = math.atan2(v[1], v[0])
        return np.array((v_norm,v_angle))
    
    @staticmethod
    def get_angle_points(p1s, p1e, p2s, p2e) -> float:
        """
        Compute the angle between two vectors given their extremities.
        
        Parameters:
        ----------
        p1s: starting of the first vector
        p1e: ending point of the first vector
        p2s: starting of the second vector
        p2e: ending point of the second vector
        ----------
        
        Returns:
        ----------
        angle (float): the angle (in [-pi,pi]) difference a2 - a1 between the two vectors, a2 and a1 corresponding to the global angle of the second and first vectors, respectively.
        ----------
        """
        v1 = p1e - p1s
        v2 = p2e - p2s
        angle = MathHelper.get_angle_vec_cart(v1,v2)
        return angle
    
    @staticmethod
    def get_angle_vec_cart(v1, v2) -> float:
        """
        Compute the angle between two vectors given in cartesian coordinate.
        
        Parameters:
        ----------
        v1: the first vector of the form (x, y)
        v2: the second vector of the form (x, y)
        ----------
        
        Returns:
        ----------
        angle (float): the angle (in [-pi,pi]) difference a2 - a1 between the two vectors, a2 and a1 corresponding to the global angle of the second and first vectors, respectively.
        ----------
        """
        v1p = MathHelper.get_polar_coord(v1)
        v2p = MathHelper.get_polar_coord(v2)
        angle = MathHelper.angle_diff(v1p[1],v2p[1])
        return angle
    
    @staticmethod
    def get_angle_vec_pol(v1, v2) -> float:
        """
        Compute the angle between two vectors given in polar coordinate.
        
        Parameters:
        ----------
        v1: the first vector of the form (norm, angle)
        v2: the second vector of the form (norm, angle)
        ----------
        
        Returns:
        ----------
        angle (float): the angle (in [-pi,pi]) difference a2 - a1 between the two vectors, a2 and a1 corresponding to the global angle of the second and first vectors, respectively.
        ----------
        """
        assert (v1[0]>=0 and v1[1]<=math.pi and -math.pi<=v1[1])
        assert (v2[0]>=0 and v2[1]<=math.pi and -math.pi<=v2[1])
        angle = MathHelper.angle_diff(v1[1],v2[1])
        return angle

class Obstacle:
    """
    A simple structure to save position and size of an obstacle.
    """
    def __init__(self, pos, size):
        self.pos = pos
        self.size = size
        
    def __eq__(self,obs):
        if self.pos[0]==obs.pos[0] and self.pos[1]==obs.pos[1] and self.size==obs.size:
            return True
        return False

    def __str__(self):
        return f'Obstacle({self.pos},{self.size})'

    def __repr__(self):
        return f'Obstacle({self.pos},{self.size})'

class BabyRobot(Obstacle):
    """
    A structure shared among Robot and Planner class, it encodes all necessary parameters of the robot which are independent of the planner used.
    """
    def __init__(self, pos: np.ndarray, theta: float = 0., speed_norm: float = 0., speed_angle: float = 0., max_speed: float = 10., size: float = 1., name: str = "default_name_robot"):
        """
        Create a Babyrobot.
        
        Parameters:
        ----------
        pos (np.ndarray): position of the babyrobot
        theta (float): globlal angle of the babyrobot
        speed_norm (float): the norm of the current velocity of the babyrobot
        speed_angle (float): the global angle of the current velocity of the robot
        max_speed (float): the maximal speed the robot can go
        size (float): the robot size, considering it is a circle
        name (str): name of the robot
        ----------

        Additionnal attributes:
        ----------
        velocity (np.ndarray): the velocity of the babyrobot in cartesian coordinates
        ----------
        """
        self.pos = pos
        self.theta = theta
        self.speed_norm = speed_norm
        self.speed_angle = speed_angle
        self.velocity = MathHelper.get_cartesian_coord((speed_norm,speed_angle))
        self.max_speed = max_speed
        self.history_pose = []
        self.size = size
        self.name = name

    def pos_update(self,pos: np.ndarray):
        """
        A method to update the mutable robot position.
        
        Parameters:
        ----------
        pos (np.ndarray): the new position of the robot
        ----------
        
        Returns:
        ----------
        A modified self.pos
        ----------
        """
        for i in range(len(pos)):
            self.pos[i]=pos[i]

class Agent(ABC):
    """
    A mother class for Robot, currently useless.
    """
    def __init__(self, robot: BabyRobot):
        """
        Create an agent.
        
        Parameters:
        ----------
        robot (BabyRobot): a Babyrobot instance
        ----------
        """
        self.robot = robot
        
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################    
#ENVIRONMENT BELOW
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################    

class Node:
    """
    A class used to attribute heuristics to a graph, especially useful for A* algorithm.
    """
    def __init__(self, current, parent=None, g: float=0, h: float=0) -> None:
        """
        Create a Node.
        
        Parameters:
        ----------
        current: the coordinates of the currently visited node/vertex
        parent: the coordinates of the parent node/vertex
        g (float): the distance traveled from the starting node to the current one (or path cost)
        h (float): the distance estimated (via heuristic such as euclidian distance) to reach the goal (or heuristic cost)
        ----------
        """
        self.current = current
        self.parent = parent
        self.g = g
        self.h = h
    
    def __add__(self, node):
        """
        Adding two nodes as a motion from the current node to another
        
        Parameters:
        ----------
        node (Node): the "motion" node
        ----------
        
        Returns:
        ----------
        _ (Node): A new node where the coordinates are the sum of both nodes, the parent is the same as the current instance, g is the sum of both g, h is the current one.
        ----------
        """
        return Node((self.x + node.x, self.y + node.y), self.parent, self.g + node.g, self.h)

    def __eq__(self, node) -> bool:
        """
        Check is two node have the same coordinates
        
        Parameters:
        ----------
        node (Node): the other node
        ----------
        
        Returns:
        ----------
        _ (bool): True if both coordinates are equals, else False.
        ----------
        """
        if isinstance(self.current, np.ndarray):
            return (self.current == node.current).all()
        if isinstance(node.current, np.ndarray):
            return (self.current == node.current).all()
        if isinstance(self.current, tuple):
            return self.current == node.current
        if isinstance(node.current, tuple):
            return self.current == node.current
        
    
    def __ne__(self, node) -> bool:
        """
        Check is two node does not have the same coordinates
        
        Parameters:
        ----------
        node (Node): the other node
        ----------
        
        Returns:
        ----------
        _ (bool): False if both coordinates are equals, else True.
        ----------
        """
        return not self.__eq__(node)

    def __lt__(self, node) -> bool:
        """
        Check if the current instance have a lower heuristic than the compared one
        
        Parameters:
        ----------
        node (Node): the other node
        ----------
        
        Return:
        ----------
        _ (bool): True if our instance g+h is lower than the compared node one, and in case of equality if our h is the lowest. 
        ----------
        """
        return self.g + self.h < node.g + node.h or \
                (self.g + self.h == node.g + node.h and self.h < node.h)

    def __hash__(self) -> int:
        """
        Hash the current node.
        """
        return hash(self.current)

    def __str__(self) -> str:
        """
        Return a string representing the Node.
        """
        return "----------\ncurrent:{}\nparent:{}\ng:{}\nh:{}\n----------" \
            .format(self.current, self.parent, self.g, self.h)
    
    @property
    def x(self) -> float:
        """
        Return the first coordinate of the current node.
        """
        return self.current[0]
    
    @property
    def y(self) -> float:
        """
        Return the second coordinate of the current node.
        """
        return self.current[1]

    @property
    def px(self) -> float:
        """
        Return the first coordinate of the parent node.
        """
        if self.parent:
            return self.parent[0]
        else:
            return None

    @property
    def py(self) -> float:
        """
        Return the second coordinate of the parent node.
        """
        if self.parent:
            return self.parent[1]
        else:
            return None

class Env(ABC):
    """
    A class that store static obstacles (in daughter classes), the size of the map, and a list of agents.
    """
    def __init__(self, x_range: int, y_range: int, agents = []):
        """
        Create an environment Env.
        
        Parameters:
        ----------
        x_range (int): the first coordinate for the size of the map, an int
        y_range (int): the second coordinate for the size of the map, an int
        agents (list): the list of movable agents
        ----------
        """
        self.x_range = x_range  
        self.y_range = y_range
        self.agents = agents

    @property
    def grid_map(self) -> set:
        """
        Return all points in Z^2 belonging to the map (from (0, 0) to (x_range, y_range)).
        """
        return {(i, j) for i in range(self.x_range) for j in range(self.y_range)}

    @abstractmethod
    def init(self):
        """
        An abstract method, used to generate the environment.
        """
        pass

class Grid(Env):
    """
    An Env that is a grid with 8 possible motions.
    TODO: consider how to face size (in file?), should we just let it to the map instead?
    """
    def __init__(self, x_range: int, y_range: int, file: str, agents = []):
        """
        Create a Grid.
        
        Parameters:
        ----------
        x_range (int): the first coordinate for the size of the map, an int
        y_range (int): the second coordinate for the size of the map, an int
        agents (list): the list of movable agents that should be considered as obstacles.
        ----------
        
        Additionnal attributes:
        ----------
        motions (list): A list of Nodes that represents possible movements from one point to another.
        obstacles (set/None): the set of obstacles
        ----------
        """
        super().__init__(x_range, y_range, agents)
        self.motions = [Node((-1, 0), None, 1, None), Node((-1, 1),  None, math.sqrt(2), None),
                        Node((0, 1),  None, 1, None), Node((1, 1),   None, math.sqrt(2), None),
                        Node((1, 0),  None, 1, None), Node((1, -1),  None, math.sqrt(2), None),
                        Node((0, -1), None, 1, None), Node((-1, -1), None, math.sqrt(2), None)]
        #self.obstacles = set()
        self.obstacles = []
        self.init(file)
    
    @staticmethod
    def write_random_env(file: str, x, y, nb_obs, rng= np.random.RandomState(0)):
        f = open(file, 'w', encoding = "utf-8")
        f.write("Environment:\n")
        my_obs = rng.random_sample((nb_obs,2))
        for i in range(nb_obs):
            f.write("(" + str(x*my_obs[i][0]) + "," + str(y*my_obs[i][1]) + ")\n")
        f.close()

    def init(self, file: str):
        """
        Initialize the map with the given file of the form "Environment:\n (x,y)\n ..."
        x and y are supposed to be integers.
        For each such line, a new tuple is added in the set of obstacles.
        
        Parameters:
        ----------
        file (str): the path where the file is stored
        ----------
        """
        #obstacles = set()
        f = open(file, 'r')
        l = f.readlines()
        obsl = False
        for i in range(len(l)):
            if re.search("Environment:",l[i]):
                obsl = True
                continue
            if obsl:
                nobs = re.search("\((-?\d+.?\d*),\s?(-?\d+.?\d*)\)",l[i])
                if nobs:
                    self.obstacles.append((float(nobs.group(1)),float(nobs.group(2))))
                else:
                    break
        f.close()
        #self.obstacles = obstacles

    def update(self, obstacles: list):
        """
        Update/reset the map with the given obstacles.
        
        Parameters:
        ----------
        obstacles (list): a list of tuples that represents all static obstacles
        ----------
        """
        self.obstacles = obstacles 

class Map(Env):
    """
    An Env that is represented with static circle or rectangular obstacles, used for local or non-graph planning.
    TODO: from grid to map, consider circle or rectangles? Should we allow non-horizotal rectangles? update infos.
    """
    def __init__(self, x_range: int, y_range: int, agents = []):
        """
        Create a Map.
        
        Parameters:
        ----------
        x_range (int): the first coordinate for the size of the map, an int
        y_range (int): the second coordinate for the size of the map, an int
        agents (list): the list of movable agents that should be considered as obstacles.
        ----------
        
        Additionnal attributes:
        ----------
        boundary: walls representing the boundary of the map, encoded as a list of rectangular obstacles
        obs_circle (Obstacle): the list of circular obstacles, each circle is an Obstacle
        obs_rect: the list of rectangular obstacles, each rectangle is a list of the shape [x_min,y_min,x_max,y_max] (the rectangles are not oriented, they always have a zero angle with the ground)
        ----------
        """
        super().__init__(x_range, y_range, agents)
        #self.boundary = None
        self.obs_circ = []
        self.obs_rect = []
        #self.init()

    def init(self):
        """
        Initialize the map.
        
        NEED TO BE CHANGED TO A FILE READER
        """
        x, y = self.x_range, self.y_range

        self.boundary = [
            [0, 0, 1, y],
            [0, y, x, 1],
            [1, 0, x, 1],
            [x, 1, 1, y]
        ]

        self.obs_rect = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2]
        ]

        self.obs_circ = [
            [7, 12, 3],
            [46, 20, 2],
            [15, 5, 2],
            [37, 7, 3],
            [37, 23, 3]
        ]

    def from_grid_to_map(self, grid, size = 0.5):
        for i in grid.obstacles:
            self.obs_circ.append(Obstacle(np.array((i[0],i[1])),size))#considering that grid have tuple (x,y) as obstacles.

    def avoid_overlap(self):
        overlap = True
        move_buffer = []
        while overlap:
            overlap=False
            for i in self.obs_circ:
                for j in self.obs_circ:
                    if np.linalg.norm(i.pos-j.pos)<(i.size+j.size) and i!=j:
                        move_buffer.append((i,np.array(i.pos-j.pos)*(1/np.linalg.norm(i.pos-j.pos))*(((i.size+j.size)-np.linalg.norm(i.pos-j.pos))/(i.size+j.size))*j.size))
                        overlap=True
            for k in move_buffer:
                k[0].pos[0]=k[0].pos[0]+k[1][0]
                k[0].pos[1]=k[0].pos[1]+k[1][1]

    def update(self, boundary, obs_circ, obs_rect):
        """
        Update/reset the map with the given obstacles.
        
        Parameters:
        ----------
        boundary: walls representing the boundary of the map, encoded as a list of rectangular obstacles
        obs_circle: the set of circular obstacles, each circle is a list of the shape [x,y,radius]
        obs_rect: the set of rectangular obstacles, each rectangle is a list of the shape [x_min,y_min,x_max,y_max] (the rectangles are not oriented, they always have a zero angle with the ground)
        ----------
        """
        self.boundary = boundary if boundary else self.boundary
        self.obs_circ = obs_circ if obs_circ else self.obs_circ
        self.obs_rect = obs_rect if obs_rect else self.obs_rect

#########################################################################################################################################################################################################
#########################################################################################################################################################################################################    
#ROBOT BELOW, AND MOTHER PLANNER
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################   

class Planner(ABC):
    """
    A mother class for both global and local planners.
    """
    def __init__(self, start: np.ndarray, goal: np.ndarray, env: Env):
        """
        Create a Planner.
        
        Parameters:
        ----------
        start (np.ndarray -> Node): the starting point of the path
        goal (np.ndarray -> Node): the ending point of the path
        env (Env): the environment
        ----------
        """
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)
        self.env = env

    def dist(self, node1: Node, node2: Node) -> float:
        """
        Give the euclidean distance between two Node
        
        Parameters:
        ----------
        node1 (Node): the first Node
        node2 (Node): the second Node
        ----------
        """
        return math.hypot(node2.x - node1.x, node2.y - node1.y)
    
    def angle(self, node1: Node, node2: Node) -> float:
        """
        Give the global angle/orientation of the vector starting from node1 and ending at node2
        
        Parameters:
        ----------
        node1 (Node): the first Node
        node2 (Node): the second Node
        ----------
        """
        return math.atan2(node2.y - node1.y, node2.x - node1.x)

    @abstractmethod
    def run(self):
        """
        An abstract method, used to plan the whole path for global path planners, and play a single step in the Simulator for local path planners.
        """
        pass

class Robot(Agent):
    """
    A class used to associate a babyrobot with a planner and compute some data.
    """
    def __init__(self, robot: BabyRobot):
        """
        Create a Robot.
        
        Parameters:
        ----------
        robot (BabyRobot): a BabyRobot instance
        ----------
        
        Additionnal attributes:
        ----------
        path_length (float): the path length traveled by the BabyRobot, set to 0
        sum_angle (tuple): the sum of angles made by the BabyRobot between two time steps since the start, encoded as (total in absolute value, left angles, right angles)
        time_to_reach_goal (float): the time needed in the simulation for the BabyRobot to reach the goal
        max_angle (float): the maximal angle (between two time steps) made by the BabyRobot during its travel
        ----------
        """
        self.robot = robot #BabyRobot(pos, theta, speed_norm, speed_angle, max_speed, size, name)
        self.robot.history_pose.append([self.robot.pos.copy(),self.robot.theta,self.robot.speed_norm,self.robot.speed_angle,self.robot.velocity])
        self.path_length = 0.
        self.sum_angle = (0.,0.,0.)
        self.time_to_reach_goal = 0.
        self.max_angle = 0.
        self.planner_list = []

    def set_planner(self, planner: Planner):
        """
        Add a planner to the robot.
        
        Parameters:
        ----------
        planner (Planner): a Planner instance that must be linked to the same BabyRobot instance.
        ----------
        """
        self.planner = planner
        
    def compute_path_length(self):
        """
        Return the path length traveled by the BabyRobot.
        """
        length = 0.
        for i in range(len(self.robot.history_pose)-1):
            #length += np.linalg.norm(self.history_pose[i][0],self.history_pose[i+1][0])
            speed = self.robot.history_pose[i+1][0] - self.robot.history_pose[i][0]
            length += math.sqrt(speed[0] ** 2 + speed[1] ** 2)
        self.path_length = length

    def compute_sum_angle(self):
        """
        Return the sum of angles made by the BabyRobot between two time steps since the start, encoded as (total in absolute value, left angles, right angles)
        """
        sumt = 0.
        suml = 0.
        sumr = 0.
        for i in range(len(self.robot.history_pose)-1):
            sdiff = MathHelper.angle_diff(self.robot.history_pose[i][1],self.robot.history_pose[i+1][1])
            diff = abs(sdiff)
            sumt += diff
            if sdiff<0:
                sumr += diff
            else:
                suml += diff
        self.sum_angle = (sumt, suml, sumr)

    def compute_time_to_reach_goal(self):
        """
        Return the time needed in the simulation for the BabyRobot to reach the goal.
        """
        self.time_to_reach_goal = self.planner.reached[1] - self.planner.t_start
        
    def compute_max_angle(self):
        """
        Return the maximal angle (between two time steps) made by the BabyRobot during its travel.
        """
        maxa = 0.
        for i in range(len(self.robot.history_pose)-1):
            sdiff = MathHelper.angle_diff(self.robot.history_pose[i][1],self.robot.history_pose[i+1][1])
            diff = abs(sdiff)
            if diff > abs(maxa):
                maxa = sdiff
        self.max_angle = maxa
        
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################        
#PLOT BELOW
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################        

class Plot:
    """
    A class dedicated to plot precomputed data
    """
    def __init__(self, env: Env):
        """
        Create a Plot.
        
        Parameters:
        ----------
        env (Env): the environment
        SHOULD WE PUT THE SIMULATOR INSTEAD?
        ----------
        
        Additionnal attributes:
        ----------
        agent_list (list): a list of agents that need to be drawn with their path
        ----------
        """
        self.env = env
        self.fig, self.ax = plt.subplots()
        self.agent_list = []
        self.ims = []
        self.ani = None

    def add_agent(self, agent: Robot):
        """
        Add an agent to be drawn.
        
        Parameters:
        ----------
        agent (Robot): a Robot that we want to show.
        ----------
        """
        self.agent_list.append(agent)

    def animation(self, name: str) -> None:
        """
        DEPRECATED, TO BE REMOVED, BUT FIRST LOOK AT A_STAR AND UNLINK IT.
        """
        self.plotEnv(name)
        print("plot env finished")
        for i in self.agent_list:
            self.plotPath(i.planner, path_color='r')
        self.plotHistoryPose()    
        plt.show()

    def animate(self, name: str) -> None:
        """
        Show an animation of the simulation.
        
        Parameters:
        ----------
        name (str): the name of the window that can be algorithm name or some other information
        ----------
        """
        self.plotEnv(name)
        print("plot env finished")
        for i in self.agent_list:
            self.plotPath(i.planner, path_color='r')
        self.plotHistoryPose2()
        self.ani = matplotlib.animation.ArtistAnimation(self.fig, self.ims, interval = 10, blit=True, repeat_delay=2000)
        plt.show()

    def plotEnv(self, name: str) -> None:
        """
        Plot environment with static obstacles.

        Parameters:
        ----------
        name (str): the name of the window that can be algorithm name or some other information
        ----------
        """
        for i in self.agent_list:
            plt.plot(i.planner.start[0], i.planner.start[1], marker="s", color="#ff0000")
            plt.plot(i.planner.goal[0], i.planner.goal[1], marker="s", color="#1155cc")
        if isinstance(self.env, Grid):
            obs_x = [x[0] for x in self.env.obstacles]
            obs_y = [x[1] for x in self.env.obstacles]
            plt.plot(obs_x, obs_y, "sk")
        if isinstance(self.env, Map):
            #ax = self.fig.add_subplot()
            ## boundary
            #for (ox, oy, w, h) in self.env.boundary:
            #    ax.add_patch(patches.Rectangle((ox, oy), w, h, edgecolor='black', facecolor='black', fill=True))
            # rectangle obstacles
            #for (ox, oy, w, h) in self.env.obs_rect:
            #    ax.add_patch(patches.Rectangle((ox, oy), w, h, edgecolor='black', facecolor='gray', fill=True))
            # circle obstacles
            obs_x = [x.pos[0] for x in self.env.obs_circ]
            obs_y = [x.pos[1] for x in self.env.obs_circ]
            plt.plot(obs_x, obs_y, "sk")
            for i in self.env.obs_circ:
                circle = self.ax.add_patch(patches.Circle(i.pos, i.size, edgecolor='black', facecolor='gray', fill=True))
        plt.title(name)
        plt.axis("equal")

    def plotPath(self, planner: Planner, path_color="#13ae00", path_style: str="-") -> None:
        """
        Plot the path made by the global planning phase for one agent.

        Parameters:
        ----------
        planner (Planner): the local planner used, that contains the path made by the global planner.
        path_color: the color used to plot the path
        path_style (str): the type of lines to be plotted
        ----------
        """
        path_x = [planner.path[i][0] for i in range(len(planner.path))]
        path_y = [planner.path[i][1] for i in range(len(planner.path))]
        plt.plot(path_x, path_y, path_style, linewidth='2', color=path_color, alpha=0.2)
        plt.plot(planner.start[0], planner.start[1], marker="s", color="#ff0000", alpha=0.2)
        plt.plot(planner.goal[0], planner.goal[1], marker="s", color="#1155cc", alpha=0.2)

    def plotAgent(self, pose, radius: float=1) -> None:
        """
        Plot agent with specifical pose.

        Parameters:
        ----------
        pose: pose of agent
        radius (float): radius of agent
        ----------
        
        DEPRECATED, TO BE REMOVED WITH ANIMATION
        """
        x = pose[0][0]
        y = pose[0][1]
        theta = pose[1]
        ref_vec = np.array([[radius / 2], [0]])
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
        end_pt = rot_mat @ ref_vec + np.array([[x], [y]])
        self.ax.arrow(x, y, float(end_pt[0]) - x, float(end_pt[1]) - y,
                width=0.1, head_width=0.40, color="r")
        circle = plt.Circle((x, y), radius, color="r", fill=False)
        self.ax.add_artist(circle)

    def plotAgent2(self, pos, radius: float=1):
        """
        Plot an agent given its position, orientation, and size.

        Parameters:
        ----------
        pos: the position and orientation of the agent
        radius (float): radius of the agent
        ----------
        
        Returns:
        ----------
        circle (patches.Circle): a circle patch
        ----------
        """
        x = pos[0][0]
        y = pos[0][1]
        #theta = pos[1]
        #theta is currently currently unused, lets see later if adding an arrow is necessary
        circle = self.ax.add_patch(patches.Circle((x, y), radius, edgecolor='black', fill=False))
        return circle

    def plotHistoryPose(self) -> None:
        """
        DEPRECATED, TO BE REMOVED WITH ANIMATION
        """
        maxlen = max(len(s.robot.history_pose) for s in self.agent_list)
        for i in range(maxlen):    
            for j in self.agent_list:
                if i>= len(j.robot.history_pose):
                    continue
                else:
                    if i < len(j.robot.history_pose) -1:
                        plt.plot([j.robot.history_pose[i][0][0], j.robot.history_pose[i + 1][0][0]], [j.robot.history_pose[i][0][1], j.robot.history_pose[i + 1][0][1]], color=j.planner.color_trace)
                        self.plotAgent(j.robot.history_pose[i])
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            plt.pause(0.001)
            for art in self.ax.get_children():
                if isinstance(art, matplotlib.patches.FancyArrow):
                    art.remove()
                if isinstance(art, matplotlib.patches.Circle):
                    art.remove()

    def plotHistoryPose2(self) -> None:
        """
        Plot the whole path of each agent, time step by time step.
        """
        maxlen = max(len(s.robot.history_pose) for s in self.agent_list)
        currentframe = []
        for i in range(maxlen):
            currentframe = []
            for j in self.agent_list:
                if i>= len(j.robot.history_pose):
                    continue
                else:
                    if i < len(j.robot.history_pose) -1:
                        x = []
                        y = []
                        for k in range(i+2):
                            x.append(j.robot.history_pose[k][0][0])
                            y.append(j.robot.history_pose[k][0][1])
                        tr = plt.plot(x, y, color=j.planner.color_trace)
                        currentframe.append(tr[0])
                        #plt.plot([j.robot.history_pose[i][0][0], j.robot.history_pose[i + 1][0][0]], [j.robot.history_pose[i][0][1], j.robot.history_pose[i + 1][0][1]], color=j.planner.color_trace)
                        #ag = self.plotAgent2(j.robot.history_pose[i])
                        #print(j.robot.size)
                        circle = self.ax.add_patch(patches.Circle((j.robot.history_pose[i][0][0], j.robot.history_pose[i][0][1]), j.robot.size, edgecolor=j.planner.color_trace, fill=False))
                        currentframe.append(circle)
            #plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            self.ims.append(currentframe.copy())

    def connect(self, name: str, func):
        """
        Used to handle event (e.g. space for pause, escape for quit) during ploting
        
        Parameters:
        ----------
        name (str): the event name
        func: the function to be applied
        ----------
        """
        self.fig.canvas.mpl_connect(name, func)

    def update(self):
        """
        Redrawing the scene
        
        IS IT USED SOMEWHERE?
        """
        self.fig.canvas.draw_idle()

    @staticmethod
    def plotArrow(x: float, y: float, theta: float, length: float, color):
        """
        Plot an arrow.
        
        Parameters:
        ----------
        x (float): the starting x-coordinate
        y (float): the starting y-coordinate
        theta (float): orientation of the arrow / global angle
        lenght (float): the length of the arrow
        color: the color of the plotted arrow
        ----------
        """
        angle = np.deg2rad(30)
        d = 0.5 * length
        w = 2
        x_start, y_start = x, y
        x_end = x + length * np.cos(theta)
        y_end = y + length * np.sin(theta)
        theta_hat_L = theta + np.pi - angle
        theta_hat_R = theta + np.pi + angle
        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)
        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)
        plt.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_L], [y_hat_start, y_hat_end_L], color=color, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_R], [y_hat_start, y_hat_end_R], color=color, linewidth=w)

    @staticmethod
    def plotCar(x: float, y: float, theta: float, width: float, length: float, color):
        """
        Plot a car as rectangular shape with an arrow inside.
        
        Parameters:
        ----------
        x (float): the starting x-coordinate
        y (float): the starting y-coordinate
        theta (float): orientation of the car / global angle
        width (float): width of the car
        lenght (float): length of the car
        color: color of the plotted car
        ----------
        """
        theta_B = np.pi + theta
        xB = x + length / 4 * np.cos(theta_B)
        yB = y + length / 4 * np.sin(theta_B)
        theta_BL = theta_B + np.pi / 2
        theta_BR = theta_B - np.pi / 2
        # Bottom-Left vertex
        x_BL = xB + width / 2 * np.cos(theta_BL)        
        y_BL = yB + width / 2 * np.sin(theta_BL)
        # Bottom-Right vertex
        x_BR = xB + width / 2 * np.cos(theta_BR)        
        y_BR = yB + width / 2 * np.sin(theta_BR)
        # Front-Left vertex
        x_FL = x_BL + length * np.cos(theta)               
        y_FL = y_BL + length * np.sin(theta)
        # Front-Right vertex
        x_FR = x_BR + length * np.cos(theta)               
        y_FR = y_BR + length * np.sin(theta)
        plt.plot([x_BL, x_BR, x_FR, x_FL, x_BL],
                 [y_BL, y_BR, y_FR, y_FL, y_BL],
                 linewidth=1, color=color)
        Plot.plotArrow(x, y, theta, length / 2, color)

#########################################################################################################################################################################################################
#########################################################################################################################################################################################################        
#GLOBAL PLANNERS BELOW
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################        

class GlobalStupid(Planner):
    """
    A global planner that does nothing. Used when we want to use a local planner alone.
    """
    def __init__(self, start: np.ndarray, goal: np.ndarray, env: Env):
        """
        Create a GlobalStupid.
        
        Parameters:
        ----------
        start (np.ndarray): the starting point of the path
        goal (np.ndarray): the ending point of the path
        env (Env): the environment
        ----------
        """
        self.start = start
        self.goal = goal
        self.env = env
        
    def run(self):
        """
        Return the path as just a straight line without middle point.
        """
        return [self.start,self.goal]

class GraphSearcher(Planner):
    """
    Base class for planner based on graph searching.
    """
    def __init__(self, start: np.ndarray, goal: np.ndarray, env: Env, heuristic_type: str="euclidean"):
        """
        Create a GraphSearcher.
        
        Parameters:
        ----------
        start (np.ndarray): start point coordinates
        goal (np.ndarray): goal point coordinates
        env (Env): environment
        heuristic_type (str): heuristic function type, can be euclidean or manhattan distances.
        ----------
        """
        super().__init__(start, goal, env)
        # heuristic type
        self.heuristic_type = heuristic_type
        # allowed motions
        self.motions = self.env.motions
        # obstacles
        self.obstacles = self.env.obstacles

    def h(self, node: Node, goal: Node) -> float:
        """
        Calculate heuristic.

        Parameters:
        ----------
        node (Node): current node
        goal (Node): goal node
        ----------

        Returns:
        ----------
        h (float): heuristic function value of node
        ----------
        """
        if self.heuristic_type == "manhattan":
            return abs(goal.x - node.x) + abs(goal.y - node.y)
        elif self.heuristic_type == "euclidean":
            return math.hypot(goal.x - node.x, goal.y - node.y)

    def cost(self, node1: Node, node2: Node) -> float:
        """
        Calculate cost for this motion.

        Parameters:
        ----------
        node1 (Node): node 1
        node2 (Node): node 2
        ----------

        Returns:
        ----------
        cost (float): cost of this motion
        ----------
        """
        if self.isCollision(node1, node2):
            return float("inf")
        return self.dist(node1, node2)

    def isCollision(self, node1: Node, node2: Node) -> bool:
        """
        Judge collision when moving from node1 to node2.

        Parameters:
        ----------
        node1 (Node): node 1
        node2 (Node): node 2
        ----------

        Returns:
        ----------
        collision (bool): True if collision exists else False
        ----------
        """
        if tuple(node1.current) in self.obstacles or tuple(node2.current) in self.obstacles:
            return True

        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y

        if x1 != x2 and y1 != y2:
            if x2 - x1 == y1 - y2:
                s1 = (min(x1, x2), min(y1, y2))
                s2 = (max(x1, x2), max(y1, y2))
            else:
                s1 = (min(x1, x2), max(y1, y2))
                s2 = (max(x1, x2), min(y1, y2))
            if s1 in self.obstacles or s2 in self.obstacles:
                return True
        return False

class AStar(GraphSearcher):
    """
    A class for the global planner A*.
    """
    def __init__(self, start: np.ndarray, goal: np.ndarray, env: Env, heuristic_type: str = "euclidean"):
        """
        Create an AStar.
        
        Parameters:
        ----------
        start (np.ndarray): start point coordinate
        goal (np.ndarray): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type
        ----------
        """
        super().__init__(start, goal, env, heuristic_type)

    def __str__(self) -> str:
        """
        Return the "A*" name.
        """
        return "A*"

    def plan(self):
        """
        A* motion plan function.

        Returns:
        ----------
        cost (float): path cost
        path (list): planning path
        visited_nodes (list): all nodes that planner has searched
        ----------
        
        SHOULD BE CHANGED TO RUN INSTEAD OF PLAN TO BE COHERENT WITH GLOBALSTUPPID AND LOCALPLANNER, OR CHANGE RUN? BUT CHECK THE OUTPUT THEN
        """
        print("starting A-star computation")
        open_nodes = []
        heapq.heappush(open_nodes, self.start)
        visited_nodes = []
        while open_nodes:
            node = heapq.heappop(open_nodes)
            if node in visited_nodes:
                continue
            if node == self.goal:
                visited_nodes.append(node)
                return self.extractPath(visited_nodes), visited_nodes
            # add non-visited neighbors to the open_nodes list
            for node_n in self.getNeighbor(node):                
                if node_n in visited_nodes:
                    continue
                node_n.parent = node.current
                node_n.h = self.h(node_n, self.goal)
                if node_n == self.goal:
                    heapq.heappush(open_nodes, node_n)
                    break
                heapq.heappush(open_nodes, node_n)
            visited_nodes.append(node)
        return ([], []), []

    def getNeighbor(self, node: Node) -> list:
        """
        Find neighbors of the current node.

        Parameters:
        ----------
        node (Node): current node
        ----------
        
        Returns:
        ----------
        _ (list): neighbors of current node
        ----------
        """
        return [node + motion for motion in self.motions if not self.isCollision(node, node + motion)]

    def extractPath(self, closed_list):
        """
        Extract the path based on the visited_nodes set.

        Parameters:
        ----------
        closed_list (list): visited_nodes set
        ----------
        
        Returns:
        ----------
        cost (float): the cost of planning path
        path (list): the planning path
        ----------
        """
        cost = 0
        node = closed_list[closed_list.index(self.goal)]
        path = [node.current]
        while node != self.start:
            node_parent = closed_list[closed_list.index(Node(node.parent, None, None, None))]
            cost += self.dist(node, node_parent)
            node = node_parent
            path.insert(0,node.current)
        return cost, path

    def run(self):
        """
        Running both planning and animation.
        
        DEPRECATED SINCE SIMULATOR AND LOCAL/AGENT ORIENTED PROJECT.
        """
        (cost, path), expand = self.plan()
        self.plot.animation(path, str(self), cost, expand)

#########################################################################################################################################################################################################
#########################################################################################################################################################################################################        
#LOCAL PLANNERS BELOW
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################        

class LocalPlanner(Planner):
    """
    The mother class for all local planners.
    """
    def __init__(self, robot: BabyRobot, start: np.ndarray, goal: np.ndarray, env: Env, obstacles, **kwargs):
        """
        Create a LocalPlanner.
        
        Parameters:
        ----------
        robot (BabyRobot): the BabyRobot that needs to follow this local planner
        start (np.ndarray): the starting point of the path
        goal (np.ndarray): the end point of the path
        env (Env): the environment, containing static obstacles
        obstacles: the obstacles this agent have to consider
        ----------

        Additionnal Parameters:
        ----------
        color_trace: the color of the path when plotted, red by default
        t_start (float): the time, in the simulator, when the agent start to move, 0 by default
        lookahead_dist (float): a distance used to setup temporary goals based on the global path, 10 by default
        nb_collision (tuple): encodes each collision already encountered, of the form (n, [t_1, ..., t_n]), (0,[]) by default
        ----------
        """
        self.start = start
        self.goal = goal
        self.env = env
        self.robot = robot
        self.reached = (False,0)
        self.g_planner = None
        self.path = None
        self.goal_current = goal
        self.obstacles = obstacles
        self.color_trace = kwargs["color_trace"] if "color_trace" in kwargs.keys() else (1,0,0)
        self.t_start = kwargs["t_start"] if "t_start" in kwargs.keys() else 0.
        self.lookahead_dist = kwargs["lookahead_dist"] if "lookahead_dist" in kwargs.keys() else 10.
        self.nb_collision = kwargs["nb_collision"] if "nb_collision" in kwargs.keys() else [0,[]]#else (n,[time_1,...,time_n])
        self.comp_time = 0.
        self.run_count = 0
        self.obs_lists = [self.obstacles]#a list of list of obstacles.
    
    def dist(self, start: np.ndarray, end: np.ndarray) -> float:
        """
        Compute the euclidean distance between end and start.

        Parameters:
        ----------
        start (np.ndarray): starting point
        end (np.ndarray): ending point
        ----------

        Returns:
        ----------
        _ (float): the euclidean distance between start and end
        ----------
        """
        return math.hypot(end[0] - start[0], end[1] - start[1])
    
    def update_goal(self, goal: np.ndarray):
        """
        Used to change the final goal

        Parameters:
        ----------
        goal (np.ndarray): the new final goal
        ----------
        """
        self.goal = goal

    def update_obstacles(self, obstacles):
        """
        Used to change the obstacles to be considered

        Parameters:
        ----------
        obstacles: the new set/list of obstacles
        ----------
        """
        self.obstacles = obstacles

    def add_obstacles(self, obstacle):
        """
        Used to change the obstacles to be considered

        Parameters:
        ----------
        obstacles: the new set/list of obstacles
        ----------
        """
        if isinstance(self.obstacles, set):
            self.obstacles.add(obstacle)
        elif isinstance(self.obstacles, list):
            self.obstacles.append(obstacle)
        else:
            assert isinstance(self.obstacles, (set,list))

    def obs_detection_circle(self, r):
        obs = []
        for i in self.obs_lists:
            for j in i:
                if np.linalg.norm(self.linalg.robot.pos - j.pos) - self.robot.size - j.size < r:
                    obs.append(j)
        self.update_obstacles(obs)
    
    def obs_detection_rays(self, rays):
        obs = []
        for i in self.obs_lists:
            for j in i:
                for k in rays:
                    if MathHelper.circleSegmentIntersectionInitial(k[0],k[1],j.pos,j.size)!=[]:
                        obs.append(j)
        self.update_obstacles(obs)

    def getLookaheadPoint(self):
        """
        Compute the point on the global path that is exactly at distance self.lookahead_dist from the BabyRobot, close to the goal.
        If there is no such point, return the after closest point defined in path.
        If we arre close enough to the goal, return the goal instead.

        Returns:
        ----------
        _ (np.ndarray): the lookahead point corresponding to lookahead_dist
        ----------
        """
        if self.path is None:
            assert RuntimeError("Please plan the path using g_planner!")
        dist_to_robot = [self.dist(p, (self.robot.pos[0], self.robot.pos[1])) for p in self.path[1:]]
        idx_closest = dist_to_robot.index(min(dist_to_robot))
        idx_goal = len(self.path) - 1
        idx_prev = idx_goal - 1
        # Looking at the semi-curve starting at the closest point (in path) to the BabyRobot and ending at the goal, we get the first point's index ahead from lookahead_dist.
        for i in range(idx_closest, len(self.path)):
            if self.dist(self.path[i], (self.robot.pos[0], self.robot.pos[1])) >= self.lookahead_dist:
                idx_goal = i
                break
        pt_x, pt_y = None, None
        if idx_goal == len(self.path) - 1:
            # If the BabyRobot is close enough from the goal, then return the goal.
            pt_x = self.path[idx_goal][0]
            pt_y = self.path[idx_goal][1]
        else:
            if idx_goal == 0:
                idx_goal = idx_goal + 1
            idx_prev = idx_goal - 1
            px, py = self.path[idx_prev][0], self.path[idx_prev][1]
            gx, gy = self.path[idx_goal][0], self.path[idx_goal][1]
            # transform to the robot frame so that the circle centers at (0,0)
            prev_p = (px - self.robot.pos[0], py - self.robot.pos[1])
            goal_p = (gx - self.robot.pos[0], gy - self.robot.pos[1])
            i_points = MathHelper.circleSegmentIntersection(prev_p, goal_p, self.lookahead_dist)
            # if a point is found (note: as it is possible to get two points, how do we ensure that we are not turning back?)
            if (not(i_points==[])): 
                pt_x = i_points[0][0] + self.robot.pos[0]
                pt_y = i_points[0][1] + self.robot.pos[1]
            # else, we return not the closest point, but the one right after it (to avoid returning to the start)
            else:
                pt_x = self.path[idx_goal][0]
                pt_y = self.path[idx_goal][1]
        return np.array((pt_x,pt_y))

    def local_minima_detection(self, vrange: float, turning: float, time: float, dt: float, prec: float, t: float) -> (bool, np.ndarray):
        """
        IN CONSTRUCTION
        Detect if the robot currently locked in a local minima. 

        Parameters:
        ----------
        vrange (float): a distance, to check if the robot position and the mean position for x seconds are less than range
        turning (float): a boundary, to check if the robot turn in circle
        time (float): the time interval used to check the previous parameters
        dt (float): the simulator time step
        prec (float): precision for position comparison
        ----------

        Returns:
        ----------
        in_local_minima (bool): True if the robot is in a local minima, else False
        ----------
        """
        #warning: range and turning are highly dependant on the speed of the vehicle.
        sumt = 0.
        mean_pose = np.array((0.,0.))
        nb_last_pose = int(time/dt)
        # if self.robot.speed_norm < self.robot.max_speed/10:
        #     #in_local_minima = True
        #     print(nb_last_pose, self.robot.pos.copy())
        #     return (True, self.robot.pos.copy())
        if len(self.robot.history_pose)>nb_last_pose+1:
            pos_start = len(self.robot.history_pose)-nb_last_pose-1
            pos_end = len(self.robot.history_pose)-2
            for i in range(pos_start, pos_end):
                if np.linalg.norm(self.robot.pos - self.robot.history_pose[i][0])<prec and i<pos_end/2:
                    #print("cross back, time-step:", pos_start+i, ", pos: ", self.robot.history_pose[i][0].copy(), ", time: ", t)
                    return(True, Obstacle(self.robot.history_pose[i][0].copy(),0.))
                diff = abs(MathHelper.angle_diff(self.robot.history_pose[i][1],self.robot.history_pose[i+1][1]))
                sumt += diff
                mean_pose += self.robot.history_pose[i+1][0]
            mean_pose = 1/nb_last_pose * mean_pose
            if np.linalg.norm(self.robot.pos - mean_pose)<vrange and sumt>turning:
                #in_local_minima = True
                #print("area detected: ", nb_last_pose, pos_start, pos_end, mean_pose, ", time: ", t)
                return (True, Obstacle(mean_pose,0.))
        #it remains one case when the local env in the robot perspective does not change, this can happens in case of moving obstacles. (circle centered on the goal)
        return (False, mean_pose)

    def collision_detection(self):
        for i in self.obs_lists:
            for j in i:
                if np.linalg.norm(self.robot.pos - j.pos) - self.robot.size - j.size <= 0:
                    self.nb_collision[0]+=1
                    self.nb_collision[1].append(j)
                    #return (True, j)
            if self.nb_collision!=[0,[]]:
                print(self.nb_collision)
        #return (False, None)

class LocalStupid(LocalPlanner):
    """
    A class used for moving obstacles that don't need to avoid anything.
    """
    def __init__(self, robot: BabyRobot, start: np.ndarray, goal: np.ndarray, env: Env, obstacles, **kwargs):
        """
        Create a LocalStupid planner.

        Parameters:
        ----------
        robot (BabyRobot): The BabyRobot that need to move
        start (np.ndarray): starting point
        goal (np.ndarray): end point
        env (Env): the environment (actually useless for this planner)
        obstacles: obstacles to avoid (actually useless for this planner)
        ----------
        """
        super().__init__(robot, start, goal, env, obstacles, **kwargs)
        self.g_planner = GlobalStupid(start, goal, env)
        self.path = self.g_planner.run()
        
    def run(self, dt: float, t: float):
        """
        Move the BabyRobot in a straight line to the goal.

        Parameters:
        ----------
        dt: the time step of the simulator
        t: the current time in the simulator
        ----------
        """
        if t>=self.t_start and self.reached[0] == False:
            vect_dir = np.array(self.goal) - np.array(self.start)
            nv = (vect_dir/np.linalg.norm(vect_dir)) * self.robot.max_speed
            self.robot.velocity = nv
            new_pos = self.robot.pos + dt * nv
            self.robot.pos_update(new_pos)
            polar_v = MathHelper.get_polar_coord(nv)
            self.robot.speed_norm = polar_v[0]
            self.robot.speed_angle = polar_v[1]
            self.robot.theta = self.robot.speed_angle
        self.robot.history_pose.append([self.robot.pos.copy(),self.robot.theta,self.robot.speed_norm,self.robot.speed_angle,self.robot.velocity])

class InputPlanner(LocalPlanner):
    """
    A class used for moving manually a BabyRobot.
    """
    def __init__(self, robot: BabyRobot, start: np.ndarray, goal: np.ndarray, env: Env, obstacles, input_x, input_y, **kwargs):
        """
        Create a InputPlanner, a planner based on user input instead of planning autonomously a future path.

        Parameters:
        ----------
        robot (BabyRobot): The BabyRobot that need to move
        start (np.ndarray): starting point
        goal (np.ndarray): end point
        env (Env): the environment (actually useless for this planner)
        obstacles: obstacles to avoid (actually useless for this planner)
        ----------
        """
        super().__init__(robot, start, goal, env, obstacles, **kwargs)
        self.g_planner = GlobalStupid(start, goal, env)
        self.path = self.g_planner.run()
        self.input_x = input_x
        self.input_y = input_y
        self.acts_on_speed = kwargs["acts_on_speed"] if "acts_on_speed" in kwargs.keys() else False
        self.rng = kwargs["rng"] if "rng" in kwargs.keys() else np.random.RandomState(0)

    def set_input(self):
        rinput = self.rng.random_sample((1,2))
        self.input_x = 2*rinput[0][0]-1
        self.input_y = 2*rinput[0][1]-1

    def run(self, dt: float, t: float):
        """
        Move the BabyRobot accordingly to the user input.

        Parameters:
        ----------
        dt: the time step of the simulator
        t: the current time in the simulator
        ----------
        """
        if t>=self.t_start and self.reached[0] == False:
            self.set_input()
            vect_dir = np.array((self.input_x,self.input_y))
            if self.acts_on_speed:
                nv = vect_dir * self.robot.max_speed
            else:
                #v = MathHelper.get_cartesian_coord([self.robot.speed_norm,self.robot.speed_angle])
                v = self.robot.velocity
                nv = dt * vect_dir * self.robot.max_speed + v
            if np.linalg.norm(nv)>self.robot.max_speed:
                    nv = nv/np.linalg.norm(nv) * self.robot.max_speed
            #nvp = MathHelper.get_polar_coord(nv)
            #if nvp[0] > self.robot.max_speed:
            #    nvp[0] = self.robot.max_speed
            #    nv = MathHelper.get_cartesian_coord(nvp)
            self.robot.velocity = nv
            new_pos = self.robot.pos + dt * nv
            self.robot.pos_update(new_pos)
            polar_v = MathHelper.get_polar_coord(nv)
            self.robot.speed_norm = polar_v[0]
            self.robot.speed_angle = polar_v[1]
            self.robot.theta = self.robot.speed_angle
        self.robot.history_pose.append([self.robot.pos.copy(),self.robot.theta,self.robot.speed_norm,self.robot.speed_angle, self.robot.velocity])

class FAPF(LocalPlanner):
    """
    A class designed for encoding a Flexible Artificial Potential Field (FAPF).
    Flexible in the sens that many different versions of APF are available.
    """
    def __init__(self, robot: BabyRobot, start: np.ndarray, goal: np.ndarray, env: Env, obstacles, k_rep: float, k_attr: float, k_dist: float, **kwargs):
        """
        Create a FAPF.

        Parameters:
        ----------
        robot (BabyRobot): the BabyRobot that needs to follow this local planner
        start (np.ndarray): the starting point of the path
        goal (np.ndarray): the end point of the path
        env (Env): the environment, containing static obstacles
        obstacles: the obstacles this agent have to consider
        k_rep (float): a constant scaling the repulsive forces
        k_attr (float): a constant scaling the attractive force
        k_dist (float): the distance from which obstacles are considered or not.
        ----------

        Additionnal parameters:
        ----------
        angle_detection (float): based on the current orientation of the BabyRobot, the angle of vision for both left and right side, pi by default (full vision)
        goal_method (string): choose the what should be the current goal, can be "goal" or "lookahead", "goal" by default
        tangential (bool): enable tangential force, False by default
        inertia (bool): enable Inertia/momemtum, False by default
        t_inertia (float): the time looking for inertia, 1 by default
        k_inertia (float): a constant scaling the inertia force, 1 by default
        linear_repulsion (bool): enable linear repulsion instead of quadratic one, False by default
        acts_on_speed (bool): choose if the force acts on speed or on acceleration, False by default (meaning acts on acceleration)
        attr_look (bool): enable a maximum to the attractive force, True by default
        g_planner (Planner): global planner to follow, GlobalStupid by default (means no global planner)
        path (list): the path made by the globalplanner
        ----------
        """
        super().__init__(robot, start, goal, env, obstacles, **kwargs)
        self.k_rep = k_rep
        self.k_attr = k_attr
        self.k_dist = k_dist
        self.angle_detection = kwargs["angle_detection"] if "angle_detection" in kwargs.keys() else math.pi
        self.goal_method = kwargs["goal_method"] if "goal_method" in kwargs.keys() else "goal"
        self.tangential = kwargs["tangential"] if "tangential" in kwargs.keys() else False
        self.inertia = kwargs["inertia"] if "inertia" in kwargs.keys() else False
        self.t_inertia = kwargs["t_inertia"] if "t_inertia" in kwargs.keys() else 1.
        self.k_inertia = kwargs["k_inertia"] if "k_inertia" in kwargs.keys() else 1.
        self.linear_repulsion = kwargs["linear_repulsion"] if "linear_repulsion" in kwargs.keys() else False
        self.acts_on_speed = kwargs["acts_on_speed"] if "acts_on_speed" in kwargs.keys() else False
        self.attr_look = kwargs["attr_look"] if "attr_look" in kwargs.keys() else True
        self.g_planner = GlobalStupid(start, goal, env)
        self.path = self.g_planner.run()

    def get_repulsive_force(self, obstacles) -> (np.ndarray, np.ndarray):
        """
        Return the sum of all repulsive and tangential forces, weighted by the repulsive constant.
        
        Parameters:
        ----------
        obstacles: the set of obstacles to be avoided
        ----------
        
        Returns:
        ----------
        rep_force (np.ndarray): the weighted sum of repulsive forces
        tan_force (np.ndarray): the weighted sum of tangential forces
        ----------
        """
        rep_force = np.array((0.,0.))
        tan_force = np.array((0.,0.))
        for i in obstacles:
            d = np.linalg.norm(self.robot.pos - i.pos)
            c_dir = MathHelper.get_cartesian_coord(np.array((1,self.robot.theta)))
            obs_angle = MathHelper.get_angle_vec_cart(c_dir, i.pos - self.robot.pos)
            #obs_angle>0 means the obstacle is to the left of the robot
            if d<self.k_dist+self.robot.size+i.size and (obs_angle>=-self.angle_detection and obs_angle<=self.angle_detection):
                if self.linear_repulsion:#TODO change to a true selection
                    nf = np.array(self.k_rep * ((1/d) - (1/self.k_dist)) * (self.robot.pos - i.pos))
                else:
                    #nf = np.array(self.k_rep * ((1/d) - (1/self.k_dist)) * ((1/d) ** 2) * (np.array(self.robot.pos) - np.array(i)))
                    #nf = np.array(self.k_rep * ((self.k_dist/d) - 1) * ((self.k_dist/d) ** 2) * (np.array(self.robot.pos) - np.array(i))) #equivalent profiles, when the repulsive force is scaled and then multiplied by kdist (mid dist =2*kdist)
                    #nf = np.array((self.k_rep/self.k_dist) * ((self.k_dist/d) - 1) * ((self.k_dist/d) ** 2) * (np.array(self.robot.pos) - np.array(i))) #flat dist indep, it is scaled to k_dist=1 and then rescaled (mid dist =2)
                    #classic
                    nf = np.array(self.k_rep * ((1/(d-self.robot.size-i.size))- (1/self.k_dist)) * ((1/(d- self.robot.size- i.size)) ** 2) * ((d-self.robot.size-i.size)/d) * (self.robot.pos - i.pos))
                #elif x:
                    #nf = np.array(self.k_rep * ((self.k_dist/(d-self.robot.size-i.size)) - 1) * ((self.k_dist/(d-self.robot.size-i.size)) ** 2) * ((d-self.robot.size-i.size)/d) * (self.robot.pos - i.pos)) 
                    ##equivalent profiles, when the repulsive force is scaled and then multiplied by kdist (mid dist =2*kdist)
                #elif y:
                    #nf = np.array((self.k_rep/self.k_dist) * ((self.k_dist/(d-self.robot.size-i.size)) - 1) * ((self.k_dist/(d-self.robot.size-i.size)) ** 2) * ((d-self.robot.size-i.size)/d) * (self.robot.pos - i.pos)) 
                    ##flat dist indep, it is scaled to k_dist=1 and then rescaled (mid dist =2)
                if self.tangential == True:
                    if obs_angle < 0:
                        nt = np.array((nf[1],-nf[0]))
                        # the angle between the robot orientation and the tangential force is always less than 90 degree.
                    else:
                        nt = np.array((-nf[1],nf[0]))
                    tan_force = tan_force + nt
                rep_force = rep_force + nf
        return (rep_force, tan_force)

    def get_attractive_force(self, goal) -> np.ndarray:
        """
        Return the weighted attractive force.
        
        Parameters:
        ----------
        goal: the current goal
        ----------
        
        Returns:
        ----------
        attr_force (np.ndarray): the weighted attractive force
        ----------
        """
        attr_force = np.array((0.,0.))
        attr_force = self.k_attr * (np.array(goal) - self.robot.pos)
        if self.attr_look and np.linalg.norm(attr_force)>self.lookahead_dist:
                attr_force = attr_force/np.linalg.norm(attr_force) * self.lookahead_dist
        #afp = MathHelper.get_polar_coord(attr_force)
        #if self.attr_look and afp[0] > self.lookahead_dist:
        #    afp[0] = self.lookahead_dist
        #    attr_force = MathHelper.get_cartesian_coord(afp)
        return attr_force

    def get_inertia(self, dt: float) -> np.ndarray:
        """
        Return the weighted mean of previous speeds, considering t_inertia.
        If we are too close to the goal, the inertia is scaled such that it decreases.
        
        Parameters:
        ----------
        dt (float): the simulator time step
        ----------
        
        Returns:
        ----------
        _ (np.ndarray): the weighted mean of previous speed
        ----------
        """
        inertia = np.array((0.,0.))
        nb_iner = int(self.t_inertia/dt)
        if len(self.robot.history_pose)>nb_iner:
            for i in self.robot.history_pose[-nb_iner:-1]:
                #inertia += MathHelper.get_cartesian_coord((i[2],i[3]))
                inertia += i[4]
            inertia = inertia/nb_iner
        else:
            for i in self.robot.history_pose:
                #inertia += MathHelper.get_cartesian_coord((i[2],i[3]))
                inertia += i[4]
            inertia = inertia/len(self.robot.history_pose)
        dist_to_goal = MathHelper.get_polar_coord(np.array(self.goal) - self.robot.pos)[0]
        if dist_to_goal<self.lookahead_dist:
            return self.k_inertia * inertia * (dist_to_goal/self.lookahead_dist)
        return self.k_inertia * inertia

    def run(self, dt: float, t: float):
        """
        Run one step of APF, moving the BabyRobot and saving its data.
        
        Parameters:
        ----------
        dt (float): the time step of the simulator
        t (float): the current time in the simulator
        ----------
        """
        if t>=self.t_start and self.reached[0] == False:
            s_t = time.process_time_ns()
            (rep_force, tan_force) = self.get_repulsive_force(self.obstacles)
            if self.goal_method == "lookahead":
                self.goal_current = self.getLookaheadPoint()
            else:
                self.goal_current = self.goal
            attr_force = self.get_attractive_force(self.goal_current)
            if self.inertia:
                inertia = self.get_inertia(dt)
            else:
                inertia = np.array((0.,0.))
            net_force = attr_force + rep_force + tan_force + inertia
            if self.acts_on_speed:
                nv = net_force
            else:
                #v = MathHelper.get_cartesian_coord([self.robot.speed_norm,self.robot.speed_angle])
                v = self.robot.velocity
                nv = dt * net_force + v
            if np.linalg.norm(nv)>self.robot.max_speed:
                nv = nv/np.linalg.norm(nv) * self.robot.max_speed
            #nvp = MathHelper.get_polar_coord(nv)
            #if nvp[0] > self.robot.max_speed:
            #    nvp[0] = self.robot.max_speed
            #    nv = MathHelper.get_cartesian_coord(nvp)
            self.robot.velocity = nv
            new_pos = self.robot.pos + dt * nv
            self.robot.pos_update(new_pos)
            polar_v = MathHelper.get_polar_coord(nv)
            self.robot.speed_norm = polar_v[0]
            self.robot.speed_angle = polar_v[1]
            self.robot.theta = self.robot.speed_angle
            e_t = time.process_time_ns()
            self.comp_time += e_t - s_t
            self.run_count += 1
        self.robot.history_pose.append([self.robot.pos.copy(),self.robot.theta,self.robot.speed_norm,self.robot.speed_angle, self.robot.velocity])

class FDAPF(LocalPlanner):
    """
    A class designed for encoding a Flexible Dynamic Artificial Potential Field (FDAPF).
    Flexible in the sens that many different versions of APF are available.
    Dynamic since the constants weighting each force can change over time to avoid local minima.
    """
    def __init__(self, robot: BabyRobot, start: np.ndarray, goal: np.ndarray, env: Env, obstacles, k_rep_min: float, k_rep_max: float, k_attr_min: float, k_attr_max: float, k_dist_min: float, k_dist_max: float, **kwargs):
        """
        Create a FDAPF.

        Parameters:
        ----------
        robot (BabyRobot): the BabyRobot that needs to follow this local planner
        start (np.ndarray): the starting point of the path
        goal (np.ndarray): the end point of the path
        env (Env): the environment, containing static obstacles
        obstacles: the obstacles this agent have to consider
        k_rep_min (float): the minimal value of the constant scaling the repulsive forces
        k_rep_max (float): the maximal value of the constant scaling the repulsive forces
        k_attr_min (float): the minimal value of the constant scaling the attractive force
        k_attr_max (float): the maximal value of the constant scaling the attractive force
        k_dist_min (float): the minimal value of the distance from which obstacles are considered or not.
        k_dist_max (float): the maximal value of the distance from which obstacles are considered or not.
        ----------

        Additionnal parameters:
        ----------
        angle_detection (float): based on the current orientation of the BabyRobot, the angle of vision for both left and right side, pi by default (full vision)
        goal_method (string): choose the what should be the current goal, can be "goal" or "lookahead", "goal" by default
        tangential (bool): enable tangential force, False by default
        inertia (bool): enable Inertia/momemtum, False by default
        t_inertia (float): the time looking for inertia, 1 by default
        k_inertia_min (float): the minimal value of the constant scaling the inertia force, 0 by default
        k_inertia_max (float): the maximal value of the  constant scaling the inertia force, 1 by default
        k_inertia (float): a constant scaling the inertia force, (k_inertia_min+k_inertia_max)/2 by default
        linear_repulsion (bool): enable linear repulsion instead of quadratic one, False by default
        acts_on_speed (bool): choose if the force acts on speed or on acceleration, False by default (meaning acts on acceleration)
        k_inertia_change_obs (float): the modification percentage of k_inertia in the presence of obstacles, 0.02 by default
        k_attr_change_obs (float): the modification percentage of k_attr in the presence of obstacles, 0.1 by default
        k_rep_change_obs (float): the modification percentage of k_rep in the presence of obstacles, 0.5 by default
        k_dist_change_obs (float): the modification percentage of k_dist in the presence of obstacles, 0.1 by default
        k_inertia_change_no_obs (float): the modification percentage of k_inertia in the absence of obstacles, 0.02 by default
        k_attr_change_no_obs (float): the modification percentage of k_attr in the absence of obstacles, 0.5 by default
        k_rep_change_no_obs (float): the modification percentage of k_rep in the absence of obstacles, 0.5 by default
        k_dist_change_no_obs (float): the modification percentage of k_dist in the absence of obstacles, 1 by default
        attr_look (bool): enable a maximum to the attractive force, True by default
        g_planner (Planner): global planner to follow, GlobalStupid by default (means no global planner)
        path (list): the path made by the globalplanner
        ----------
        """
        super().__init__(robot, start, goal, env, obstacles, **kwargs)
        self.k_rep_min = k_rep_min
        self.k_rep_max = k_rep_max
        self.k_rep = k_rep_min
        self.k_attr_min = k_attr_min
        self.k_attr_max = k_attr_max
        self.k_attr = k_attr_max
        self.k_dist = k_dist_min
        self.k_dist_min = k_dist_min
        self.k_dist_max = k_dist_max
        self.recent_lm = 0.
        self.n_vobs = []
        self.angle_detection = kwargs["angle_detection"] if "angle_detection" in kwargs.keys() else math.pi
        self.goal_method = kwargs["goal_method"] if "goal_method" in kwargs.keys() else "goal"
        self.tangential = kwargs["tangential"] if "tangential" in kwargs.keys() else False
        self.inertia = kwargs["inertia"] if "inertia" in kwargs.keys() else False
        self.t_inertia = kwargs["t_inertia"] if "t_inertia" in kwargs.keys() else 1.
        self.k_inertia_min = kwargs["k_inertia_min"] if "k_inertia_min" in kwargs.keys() else 0.
        self.k_inertia_max = kwargs["k_inertia_max"] if "k_inertia_max" in kwargs.keys() else 1.
        self.k_inertia = kwargs["k_inertia"] if "k_inertia" in kwargs.keys() else (self.k_inertia_min + self.k_inertia_max)/2.
        self.linear_repulsion = kwargs["linear_repulsion"] if "linear_repulsion" in kwargs.keys() else False
        self.acts_on_speed = kwargs["acts_on_speed"] if "acts_on_speed" in kwargs.keys() else False
        self.k_inertia_change_obs = kwargs["k_inertia_change_obs"] if "k_inertia_change_obs" in kwargs.keys() else 0.02
        self.k_attr_change_obs = kwargs["k_attr_change_obs"] if "k_attr_change_obs" in kwargs.keys() else 0.1
        self.k_rep_change_obs = kwargs["k_rep_change_obs"] if "k_rep_change_obs" in kwargs.keys() else 0.5
        self.k_dist_change_obs = kwargs["k_dist_change_obs"] if "k_dist_change_obs" in kwargs.keys() else 0.1
        self.k_inertia_change_no_obs = kwargs["k_inertia_change_no_obs"] if "k_inertia_change_no_obs" in kwargs.keys() else 0.02
        self.k_attr_change_no_obs = kwargs["k_attr_change_no_obs"] if "k_attr_change_no_obs" in kwargs.keys() else 0.5
        self.k_rep_change_no_obs = kwargs["k_rep_change_no_obs"] if "k_rep_change_no_obs" in kwargs.keys() else 0.5
        self.k_dist_change_no_obs = kwargs["k_dist_change_no_obs"] if "k_dist_change_no_obs" in kwargs.keys() else 1.
        self.attr_look = kwargs["attr_look"] if "attr_look" in kwargs.keys() else True
        self.g_planner = GlobalStupid(start, goal, env)
        self.path = self.g_planner.run()

    def set_dynamic_constant_forces(self, in_lm, dt: float):
        """
        The function responsible of changing the dynamic constants.
        
        Parameters:
        ----------
        obstacles: the set of obstacles to avoid
        dt (float): the time step of the simulator
        ----------
        """
        if in_lm:
            self.k_rep = min(self.k_rep*(1+self.k_rep_change_obs*dt),self.k_rep_max)
            self.k_attr = max(self.k_attr*(1-self.k_attr_change_obs*dt), self.k_attr_min)
            self.k_inertia = max(self.k_inertia*(1-self.k_inertia_change_obs*dt), self.k_inertia_min)
            self.k_dist = min(self.k_dist*(1+self.k_dist_change_obs*dt),self.k_dist_max)
        else:
            self.k_rep = max(self.k_rep*(1-self.k_rep_change_no_obs*dt),self.k_rep_min)
            self.k_attr = min(self.k_attr*(1+self.k_attr_change_no_obs*dt),self.k_attr_max)
            self.k_inertia = max(self.k_inertia*(1+self.k_inertia_change_no_obs*dt), self.k_inertia_max)
            self.k_dist = max(self.k_dist*(1-self.k_dist_change_no_obs*dt),self.k_dist_min)
        dist_to_goal = np.linalg.norm(np.array(self.robot.pos) - np.array(self.goal))
        if dist_to_goal < self.k_dist:
            self.k_rep = self.k_rep * (dist_to_goal/self.k_dist)**2

    def get_repulsive_force(self, obstacles) -> (np.ndarray, np.ndarray):
        """
        Return the sum of all repulsive and tangential forces, weighted by the repulsive constant.
        
        Parameters:
        ----------
        obstacles: the set of obstacles to be avoided
        ----------
        
        Returns:
        ----------
        rep_force (np.ndarray): the weighted sum of repulsive forces
        tan_force (np.ndarray): the weighted sum of tangential forces
        ----------
        """
        rep_force = np.array((0.,0.))
        tan_force = np.array((0.,0.))
        for i in obstacles:
            d = np.linalg.norm(self.robot.pos - i.pos)
            c_dir = MathHelper.get_cartesian_coord(np.array((1,self.robot.theta)))
            obs_angle = MathHelper.get_angle_vec_cart(c_dir, i.pos - self.robot.pos)
            if d<self.k_dist+self.robot.size+i.size and (obs_angle>=-self.angle_detection and obs_angle<=self.angle_detection):
                if self.linear_repulsion:#TODO change to a true selection
                    nf = np.array(self.k_rep * ((1/d) - (1/self.k_dist)) * (self.robot.pos - i.pos))
                else:
                    #nf = np.array(self.k_rep * ((1/d) - (1/self.k_dist)) * ((1/d) ** 2) * (np.array(self.robot.pos) - np.array(i)))
                    #nf = np.array(self.k_rep * ((self.k_dist/d) - 1) * ((self.k_dist/d) ** 2) * (np.array(self.robot.pos) - np.array(i))) #equivalent profiles, when the repulsive force is scaled and then multiplied by kdist (mid dist =2*kdist)
                    #nf = np.array((self.k_rep/self.k_dist) * ((self.k_dist/d) - 1) * ((self.k_dist/d) ** 2) * (np.array(self.robot.pos) - np.array(i))) #flat dist indep, it is scaled to k_dist=1 and then rescaled (mid dist =2)
                    #classic
                    nf = np.array(self.k_rep * ((1/(d-self.robot.size-i.size))- (1/self.k_dist)) * ((1/(d- self.robot.size- i.size)) ** 2) * ((d-self.robot.size-i.size)/d) * (self.robot.pos - i.pos))
                #elif x:
                    #nf = np.array(self.k_rep * ((self.k_dist/(d-self.robot.size-i.size)) - 1) * ((self.k_dist/(d-self.robot.size-i.size)) ** 2) * ((d-self.robot.size-i.size)/d) * (self.robot.pos - i.pos)) 
                    ##equivalent profiles, when the repulsive force is scaled and then multiplied by kdist (mid dist =2*kdist)
                #elif y:
                    #nf = np.array((self.k_rep/self.k_dist) * ((self.k_dist/(d-self.robot.size-i.size)) - 1) * ((self.k_dist/(d-self.robot.size-i.size)) ** 2) * ((d-self.robot.size-i.size)/d) * (self.robot.pos - i.pos)) 
                    ##flat dist indep, it is scaled to k_dist=1 and then rescaled (mid dist =2)
                if self.tangential:
                    if obs_angle < 0:
                        nt = np.array((nf[1],-nf[0]))
                        #if the obstacle is to the right of the robot, the tangential force goes to the right of the repulsive force, so that, cutting the plan with the line made by the repulsive force vector, the tangential force lives in the same half-space as the attractive force.
                    else:
                        nt = np.array((-nf[1],nf[0]))
                    tan_force = tan_force + nt
                rep_force = rep_force + nf
        return (rep_force, tan_force)

    def get_attractive_force(self, goal) -> np.ndarray:
        """
        Return the weighted attractive force.
        
        Parameters:
        ----------
        goal: the current goal
        ----------
        
        Returns:
        ----------
        attr_force (np.ndarray): the weighted attractive force
        ----------
        """
        attr_force = np.array((0.,0.))
        attr_force = self.k_attr * (np.array(goal) - self.robot.pos)
        if self.attr_look and np.linalg.norm(attr_force)>self.lookahead_dist:
                attr_force = attr_force/np.linalg.norm(attr_force) * self.lookahead_dist
        #afp = MathHelper.get_polar_coord(attr_force)
        #if self.attr_look and afp[0] > self.lookahead_dist:
        #    afp[0] = self.lookahead_dist
        #    attr_force = MathHelper.get_cartesian_coord(afp)
        return attr_force

    def get_inertia(self, dt: float) -> np.ndarray:
        """
        Return the weighted mean of previous speeds, considering t_inertia.
        If we are too close to the goal, the inertia is scaled such that it decreases.
        
        Parameters:
        ----------
        dt (float): the simulator time step
        ----------
        
        Returns:
        ----------
        _ (np.ndarray): the weighted mean of previous speed
        ----------
        """
        inertia = np.array((0.,0.))
        nb_iner = int(self.t_inertia/dt)
        if len(self.robot.history_pose)>nb_iner:
            for i in self.robot.history_pose[-nb_iner:-1]:
                #inertia += MathHelper.get_cartesian_coord((i[2],i[3]))
                inertia += i[4]
            inertia = inertia/nb_iner
        else:
            for i in self.robot.history_pose:
                #inertia += MathHelper.get_cartesian_coord((i[2],i[3]))
                inertia += i[4]
            inertia = inertia/len(self.robot.history_pose)
        dist_to_goal = MathHelper.get_polar_coord(np.array(self.goal) - self.robot.pos)[0]
        if dist_to_goal<self.lookahead_dist:
            return self.k_inertia * inertia * (dist_to_goal/self.lookahead_dist)
        return self.k_inertia * inertia

    def run(self, dt: float, t: float):
        """
        Run one step of DAPF, moving the BabyRobot and saving its data.
        
        Parameters:
        ----------
        dt (float): the time step of the simulator
        t (float): the current time in the simulator
        ----------
        """
        if t>=self.t_start and self.reached[0] == False:
            s_t = time.process_time_ns()
            #need to study the possible realistic parameters for local minima detection
            in_lm = False
            if abs(self.recent_lm) < dt:
                while len(self.n_vobs) >= 1:
                    #print("self.n_vobs: ", self.n_vobs)
                    self.obstacles.append(self.n_vobs.pop())
                (in_lm, n_pos) = self.local_minima_detection(self.robot.max_speed*2, 2*math.pi, 6., dt, 0.1, t)
                if in_lm:
                    self.recent_lm = 2.
                    self.n_vobs.append(n_pos)
            else:
                self.recent_lm -= dt
            self.set_dynamic_constant_forces(dt<self.recent_lm<=2., dt)
            (rep_force, tan_force) = self.get_repulsive_force(self.obstacles)
            if self.goal_method == "lookahead":
                self.goal_current = self.getLookaheadPoint()
            else:
                self.goal_current = self.goal
            attr_force = self.get_attractive_force(self.goal_current)
            if self.inertia:
                inertia = self.get_inertia(dt)
            else:
                inertia = np.array((0.,0.))
            net_force = attr_force + rep_force + tan_force + inertia
            if self.acts_on_speed:
                nv = net_force
            else:
                v = MathHelper.get_cartesian_coord([self.robot.speed_norm,self.robot.speed_angle])
                nv = dt * net_force + v
            if np.linalg.norm(nv)>self.robot.max_speed:
                nv = nv/np.linalg.norm(nv) * self.robot.max_speed
            #nvp = MathHelper.get_polar_coord(nv)
            #if nvp[0] > self.robot.max_speed:
            #    nvp[0] = self.robot.max_speed
            #    nv = MathHelper.get_cartesian_coord(nvp)
            self.robot.velocity = nv
            new_pos = self.robot.pos + dt * nv
            self.robot.pos_update(new_pos)
            polar_v = MathHelper.get_polar_coord(nv)
            self.robot.speed_norm = polar_v[0]
            self.robot.speed_angle = polar_v[1]
            self.robot.theta = self.robot.speed_angle
            e_t = time.process_time_ns()
            self.comp_time += e_t - s_t
            self.run_count += 1
        self.robot.history_pose.append([self.robot.pos.copy(),self.robot.theta,self.robot.speed_norm,self.robot.speed_angle,self.robot.velocity])

#########################################################################################################################################################################################################
#########################################################################################################################################################################################################        
#SIMULATOR BELOW
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################        

class Simulator:
    """
    The main class that groups every agents and environment, and run the simulation.
    """
    def __init__(self, env: Env, t_step: float, t_limit: float, prec: float):
        """
        Create a Simulator.
        
        Parameters:
        ----------
        env (Env): the environment, that comprises static obstacles
        t_step (float): the time step of the simulation
        t_limit (float): the maximal simulated time
        prec (float): a precision criterion for stopping agents when reach their goal
        ----------
        
        Additionnal attributes:
        ----------
        agent_list (list): a list of agents moving in the simulator.
        plot (Plot): the class responsible of plotting the simulation
        t (float): the current time inside the simulator
        ----------
        """
        self.agent_list = []
        self.env = env
        self.t_step = t_step
        self.plot = Plot(env)
        #TEMPORARY COMMENT TO RUN THE BENCHMARK
        self.t = 0.
        self.t_limit = t_limit
        self.prec = prec
    
    def add_agent(self, agent: Robot):
        """
        Adding an agent to be simulated.
        
        Parameters:
        ----------
        agent (Robot): the new agent to be added to the simulator
        ----------
        """
        self.agent_list.append(agent)
        self.plot.add_agent(agent)
        #TEMPORARY COMMENT TO RUN THE BENCHMARK

    def __call__(self, *planner_list):
        """
        DEPRECATED
        """
        for i in planner_list:
            i.run()

    def run(self):
        """
        The main simulation loop.
        """
        dt = self.t_step
        all_reach = True
        for i in self.agent_list:
            if np.linalg.norm(np.array(i.planner.robot.pos) - np.array(i.planner.goal)) < self.prec:
                i.planner.reached = (True, self.t)
            if i.planner.reached == (False,0):
                all_reach = False
        while (not(all_reach) and self.t < self.t_limit):
            all_reach = True
            for i in self.agent_list:
                i.planner.run(dt, self.t)
                if not(i.planner.reached[0]):
                    if np.linalg.norm(np.array(i.planner.robot.pos) - np.array(i.planner.goal)) < self.prec:
                        i.planner.reached = (True, self.t)
                    if i.planner.reached == (False,0):
                        all_reach = False
            self.t += dt

    def run2(self, agent):
        """
        The main simulation loop.
        """
        dt = self.t_step
        #all_reach = True
        for i in self.agent_list:
            if np.linalg.norm(np.array(i.planner.robot.pos) - np.array(i.planner.goal)) < self.prec:
                i.planner.reached = (True, self.t)
            if i.planner.reached == (False,0):
                all_reach = False
        while (not(agent.planner.reached[0]) and (agent.planner.nb_collision == [0,[]]) and self.t < self.t_limit):
            all_reach = True
            for i in self.agent_list:
                i.planner.run(dt, self.t)
                i.planner.collision_detection()
                if not(i.planner.reached[0]):
                    if np.linalg.norm(np.array(i.planner.robot.pos) - np.array(i.planner.goal)) < self.prec:
                        i.planner.reached = (True, self.t)
                    #if i.planner.reached == (False,0):
                    #    all_reach = False
            self.t += dt
            
    def avoid_overlap(self,obs_lists):
        overlap = True
        move_buffer = []
        obss = []
        for i in obs_lists:
            for j in i:
                obss.append(j)
        iter = 0
        while overlap:
            move_buffer = []
            overlap=False
            for i in obss:
                for j in obss:
                    if np.linalg.norm(i.pos-j.pos)<(i.size+j.size)-0.001 and i!=j:
                        move_buffer.append((i,(np.array(i.pos-j.pos)*(1/np.linalg.norm(i.pos-j.pos))*(((i.size+j.size)-np.linalg.norm(i.pos-j.pos))/(i.size+j.size))*j.size),j,np.linalg.norm(i.pos-j.pos)))
                        overlap=True
            #print("nb buffer: ", len(move_buffer))
            for k in move_buffer:
                k[0].pos[0]=k[0].pos[0]+k[1][0]
                k[0].pos[1]=k[0].pos[1]+k[1][1]
            iter+=1
            #print("iter: ",iter)
        for l in obss:
            l.pos[0]=l.pos[0]*1.01
            l.pos[1]=l.pos[1]*1.01

    def compute_data(self):
        """
        For each agent in the simulator, compute several data such as path length.
        """
        for i in self.agent_list:
            i.compute_path_length()
            i.compute_sum_angle()
            i.compute_time_to_reach_goal()
            i.compute_max_angle()
    
    def print_data(self):
        """
        Print all the computed data.
        """
        for i in self.agent_list:
            print(i.robot.name, " path length:", i.path_length)
            print(i.robot.name, " sum angles:", i.sum_angle)
            print(i.robot.name, " time to reach goal:", i.time_to_reach_goal)
            print(i.robot.name, " max angle:", i.max_angle)
            print(i.robot.name, " computation time:", i.planner.comp_time)#to be changed if a list of planners.
            if i.planner.run_count!=0:
                print(i.robot.name, " average computation time:", i.planner.comp_time/i.planner.run_count)

    def save_data(self, file: str):
        """
        Save the computed data into a file.
        
        Parameters:
        ----------
        file (str): the path location/name of the file 
        ----------
        """
        f = open(file, 'w', encoding = "utf-8")
        f.write("Env:\n\n")
        for i in self.env.obs_circ:#tacles:
            f.write(str(i) + '\n')
        f.write('\n' + "simulation:\n\n")
        hps = len(self.agent_list[0].robot.history_pose)
        for j in range(hps):
            for i in self.agent_list:
                f.write(str(j*self.t_step) + ' ' + i.robot.name + ' ' + str(i.robot.history_pose[j]) + '\n')
        f.write('\n' + "data:\n\n")
        for i in self.agent_list:
            f.write(i.robot.name + " path length:" + str(i.path_length) + '\n')
            f.write(i.robot.name + " sum angles:" + str(i.sum_angle) + '\n')
            f.write(i.robot.name + " time to reach goal:" + str(i.time_to_reach_goal) + '\n')
            f.write(i.robot.name + " max angle:" + str(i.max_angle) + '\n')
            f.write(i.robot.name + " computation time:" + str(i.planner.comp_time) + '\n')
            f.write(i.robot.name + " number of collision:" + str(i.planner.nb_collision[0]) + '\n')
            if i.planner.run_count!=0:
                f.write(i.robot.name + " average computation time:" + str(i.planner.comp_time/i.planner.run_count) + '\n')
        f.close()

    def save_data_latex(self, file: str):
        """
        Save the computed data into a file as a latex tikzpicture format with caption.
        
        Parameters:
        ----------
        file (str): the path location/name of the file 
        ----------
        """
        f = open(file, 'w', encoding = "utf-8")
        hps = len(self.agent_list[0].robot.history_pose)
        f.write("\\begin{figure}[H]\n \\centering\n \\begin{tikzpicture}\n") 
        for i in self.agent_list:
            f.write("\\definecolor{color" + i.robot.name + "}{rgb}{" + str(i.planner.color_trace[0]) + "," + str(i.planner.color_trace[1]) + "," + str(i.planner.color_trace[2]) + "}\n")
            f.write("\\draw[line width = 1pt, color=color" + i.robot.name + "] " + "(" + str(i.robot.history_pose[0][0][0]) + "," + str(i.robot.history_pose[0][0][1]) + ")")
            for j in range(1, hps):
                f.write(" -- (" + str(i.robot.history_pose[j][0][0]) + "," + str(i.robot.history_pose[j][0][1]) + ")")
            f.write(';\n\n')
        for i in self.env.obs_circ:#tacles:
            f.write("\\filldraw[black] (" + str(i.pos[0]) + ", " + str(i.pos[1]) + ") circle (1pt);\n")
        f.write("\\end{tikzpicture}\n \\caption{Environment:\\\\ \n")
        for i in self.env.obs_circ:#tacles:
            f.write(str(i) + '       \n')
        for i in self.agent_list:
            f.write(i.robot.name + ": start " + str(i.planner.start) + ", goal " + str(i.planner.goal) + ", parameters: " + 
                    str(i.robot.max_speed) + ", " + str(i.planner.k_attr) + ", " + str(i.planner.k_rep) + ", " + str(i.planner.k_dist)
                    + "\\\\ \n" + "path length: " + str(i.path_length) + "\\\\ \n" + "sum angles: " + str(i.sum_angle) + "\\\\ \n" + "time to reach goal: " + str(i.time_to_reach_goal) + "\\\\ \n" + "additional obstacles: ")
        for i in self.agent_list:
            pobs=i.planner.obstacles.copy()
            #print("pobs",pobs)
            eobs=self.env.obs_circ.copy()
            #print("eobs",eobs)
            diff=[]
            for j in pobs:
                b=True
                for k in eobs:
                    if j==k:
                        #pobs.remove(j)
                        eobs.remove(k)
                        b=False
                        break
                if b:
                    diff.append(j)
            #print('diff',diff)
            #s = set(i.planner.obstacles).difference(set(self.env.obs_circ))
            f.write(i.robot.name + ": " + str(diff) + "\\\\ \n")
        f.write("}\n \\label{fig:enter-label}\n \\end{figure}")
        f.close()

    def get_init(self, file: str):
        pass

class Map_Designer:
    @staticmethod
    def make_u(length, width, pos, density, file):
        f = open(file, 'w', encoding = "utf-8")
        f.write("Environment:\n")
        my_obs = np.array(pos)
        c_l = 0
        while c_l <= length:
            f.write("(" + str(my_obs[0]+c_l) + "," + str(my_obs[1]) + ")\n")
            f.write("(" + str(my_obs[0]+c_l) + "," + str(my_obs[1]+width) + ")\n")
            c_l+=density
        c_w = 0
        while c_w <= width:
            f.write("(" + str(my_obs[0]+length) + "," + str(my_obs[1]+c_w) + ")\n")
            c_w += density
        f.close()

    @staticmethod
    def make_v(length, width, pos, density, file):
        f = open(file, 'w', encoding = "utf-8")
        f.write("Environment:\n")
        my_obs = np.array(pos)
        h=math.sqrt(length**2 + (width/2)**2)
        alpha = math.acos(length/h)
        stop = 0
        i=0
        while stop < length:
            f.write("(" + str(my_obs[0]+i*density*math.cos(alpha)) + "," + str(my_obs[1]+i*density*math.sin(alpha)) + ")\n")
            f.write("(" + str(my_obs[0]+i*density*math.cos(alpha)) + "," + str(my_obs[1]+width-i*density*math.sin(alpha)) + ")\n")
            stop+=density*math.cos(alpha)
            i+=1
        f.close()

    @staticmethod
    def make_poly_line(pos_list,density,file,theta = 0):
        f = open(file, 'w', encoding = "utf-8")
        f.write("Environment:\n")
        points = []
        meanpoint = np.array((0.,0.))
        for i in range(len(pos_list)-1):
            p = np.array(pos_list[i])
            points.append(p.copy())
            meanpoint+=points[-1]
            q = np.array(pos_list[i+1])
            (d,alpha) = MathHelper.get_polar_coord(q-p)
            j = 0
            stop = d/density
            while j < stop:
                points.append(np.array((p[0]+j*density*math.cos(alpha),p[1]+j*density*math.sin(alpha))))
                meanpoint+=points[-1]
                #f.write("(" + str(p[0]+j*density*math.cos(alpha)) + "," + str(p[1]+j*density*math.sin(alpha)) + ")\n")
                j+=1
        p = np.array(pos_list[-1])
        points.append(p.copy())
        meanpoint+=points[-1]
        meanpoint=meanpoint/len(points)
        #f.write("(" + str(pos_list[-1][0]) + "," + str(pos_list[-1][1]) + ")\n")
        rtheta = np.array(((math.cos(theta),-math.sin(theta)),(math.sin(theta),math.cos(theta))))
        npoints = []
        if theta!=0:
            for i in points:
                npoints.append(rtheta@(i-meanpoint)+meanpoint)
            for i in npoints:
                f.write("(" + str(i[0]) + "," + str(i[1]) + ")\n")
        else:
            for i in points:
                f.write("(" + str(i[0]) + "," + str(i[1]) + ")\n")
        f.close()

    @staticmethod
    def make_ellipse(length, width, center, paramt, file, theta = 0):
        f = open(file, 'w', encoding = "utf-8")
        f.write("Environment:\n")
        my_obs = np.array(center)
        points = []
        i=0
        while i < math.pi/2:
            points.append(np.array((my_obs[0]+length*math.cos(i),my_obs[1]+width*math.sin(i))))
            points.append(np.array((my_obs[0]+length*math.cos(i),my_obs[1]-width*math.sin(i))))
            #f.write("(" + str(my_obs[0]+length*math.cos(i)) + "," + str(my_obs[1]+width*math.sin(i)) + ")\n")
            #f.write("(" + str(my_obs[0]+length*math.cos(i))+ "," + str(my_obs[1]-width*math.sin(i)) + ")\n")
            i+=paramt
        rtheta = np.array(((math.cos(theta),-math.sin(theta)),(math.sin(theta),math.cos(theta))))
        npoints = []
        if theta!=0:
            for i in points:
                npoints.append(rtheta@(i-center)+center)
            for i in npoints:
                f.write("(" + str(i[0]) + "," + str(i[1]) + ")\n")
        else:
            for i in points:
                f.write("(" + str(i[0]) + "," + str(i[1]) + ")\n")
        f.close()


class GAparams:
    """
    A class used to see what could be good parameters for FAPF in the current environement.
    """
    def __init__(self, list_of_list_of_genes: list, pop_size: int):
        """
        Create a GAparams.
        
        Parameters:
        ----------
        list_of_list_of_genes (list): a list of list, each list being possiblilities for one parameter.
        pop_size: the total population for the genetic algorithm
        ----------
        
        Additionnal attributes:
        ----------
        my_cromozomes (list): the list of parametrised APF in the current generation
        ----------
        """
        #the list_of_list_gene determine the size of the chromosome, and the possibilities for each gene
        self.list_of_list_of_genes = list_of_list_of_genes
        self.pop_size = pop_size
        self.my_chromosomes = []

    def mutated_gene(self, list_of_genes: list):
        """
        Randomly selecting one value for a parameter.
        
        Paramters:
        ----------
        list_of_genes (list): a list inside list_of_list_of_genes, this list correspond to the possibles values for one FAPF parameter
        ----------
        
        Returns:
        ----------
        _: a random element of the list
        ----------
        """
        return random.choice(list_of_genes)

    def make_chromosome(self) -> list:
        """
        Make one parametrised FAPF (the chromosome).
        
        Returns:
        ----------
        chr (list): a list of values, one value corresponding to one FAPF parameter
        ----------
        """
        chr_size = len(self.list_of_list_of_genes)
        chr = []
        for i in range(chr_size):
            chr.append(self.mutated_gene(self.list_of_list_of_genes[i]))
        return chr

    def mate(self, chr_1: list, chr_2: list) -> list:
        """
        Make an new chromosome based one two parents.
        
        Parameters:
        ----------
        chr_1 (list): a list of values, one value corresponding to one FAPF parameter, the first parent
        chr_2 (list): a list of values, one value corresponding to one FAPF parameter, the second parent
        ----------
        
        Returns:
        ----------
        child_chr (list): a list of values, one value corresponding to one FAPF parameter, the child chromosome
        ----------
        """
        child_chr = []
        for g1, g2 in zip(chr_1, chr_2):
            prob = random.random()
            if prob < 0.45:
                child_chr.append(g1)
            elif prob < 0.9:
                child_chr.append(g2)
            else:
                child_chr.append(self.mutated_gene(self.list_of_list_of_genes[chr_1.index(g1)]))
        return child_chr

    def run(self):
        """
        The main loop for the genetic algorithm. It runs the simulator at each generation with the given population and select the best ones.
        
        Returns:
        ----------
        fitness_list (list): a list of tuple of the form (generation, best chromosome for this generation, its fitness value), the fitness value considered is the sum of the path lenght and the distance of the BabyRobot from the goal at the end of the simulation
        ----------
        """
        fitness = 1e14
        fitness_list = [] # store the best outcome from each generation
        generation = 1
        env = Grid(400,200)
        start = np.array((0.,0.))
        goal = np.array((350.,150.))
        obstacles = env.obstacles
        robot_list = []
        planner_list = []
        simulator = Simulator(env,0.1,80)
        for i in range(self.pop_size):
            robot_list.append(Robot(start.copy(),0.,0.,0.,10.,0.,str(i)))
            chr = self.make_chromosome()
            planner_list.append(FDAPF(robot_list[i].robot, robot_list[i].robot.pos.copy(), goal.copy(), env, obstacles, 
                                chr[0], chr[1], chr[2], chr[3], chr[4], chr[5],
                                linear_repulsion = True, lookahead_dist = 50., inertia = True, tangential = True
                                ))
            robot_list[i].set_planner(planner_list[i])
            simulator.add_agent(robot_list[i])
            self.my_chromosomes.append((chr,robot_list[i]))
        print("simulation run, generation:", generation)
        simulator.run()
        simulator.compute_data()
        b = True
        self.my_chromosomes.sort(key = lambda x: (np.linalg.norm(goal-x[1].robot.history_pose[-1][0]), x[1].path_length))
        fitness = (np.linalg.norm(goal-self.my_chromosomes[0][1].robot.history_pose[-1][0]), self.my_chromosomes[0][1].path_length)
        fitness_list.append((generation, self.my_chromosomes[0], fitness))
        print(fitness_list[-1])
        while b:#fitness > 600 and (len(fitness_list) < 5 or abs(sorted(fitness_list[-6])-sorted(fitness_list[-1]))< 20):
            new_chr =[]
            robot_list = []
            planner_list = []
            simulator = Simulator(env,0.1,80)
            s = int((10*self.pop_size)/100)
            for i in range(s):
                robot_list.append(Robot(start.copy(),0.,0.,0.,10.,0.,str(i)))
                chr = self.my_chromosomes[i][0]
                planner_list.append(FDAPF(robot_list[i].robot, robot_list[i].robot.pos.copy(), goal.copy(), env, obstacles, 
                                    chr[0], chr[1], chr[2], chr[3], chr[4], chr[5],
                                    linear_repulsion = True, lookahead_dist = 50., inertia = True, tangential = True
                                    ))
                robot_list[i].set_planner(planner_list[i])
                simulator.add_agent(robot_list[i])
                new_chr.append((chr,robot_list[i]))
            s2 = self.pop_size-s
            for i in range(s):
                robot_list.append(Robot(start.copy(),0.,0.,0.,10.,0.,str(i)))
                fpc=int(self.pop_size/2)
                chr = self.mate(random.choice(self.my_chromosomes[:fpc])[0],random.choice(self.my_chromosomes[:fpc])[0])
                planner_list.append(FDAPF(robot_list[i].robot, robot_list[i].robot.pos.copy(), goal.copy(), env, obstacles, 
                                    chr[0], chr[1], chr[2], chr[3], chr[4], chr[5],
                                    linear_repulsion = True, lookahead_dist = 50., inertia = True, tangential = True
                                    ))
                robot_list[i].set_planner(planner_list[i])
                simulator.add_agent(robot_list[i])
                new_chr.append((chr,robot_list[i]))
            self.my_chromosomes = new_chr.copy()
            generation+=1
            print("simulation run, generation:", generation)
            simulator.run()
            simulator.compute_data()
            self.my_chromosomes.sort(key = lambda x: (np.linalg.norm(goal-x[1].robot.history_pose[-1][0]), x[1].path_length))
            fitness = (np.linalg.norm(goal-self.my_chromosomes[0][1].robot.history_pose[-1][0]), self.my_chromosomes[0][1].path_length)
            fitness_list.append((generation, self.my_chromosomes[0], fitness))
            print(fitness_list[-1])
            if fitness[0]<1:
                b = False
        return fitness_list    
                

#########################################################################################################################################################################################################
#########################################################################################################################################################################################################            
#MAIN BELOW
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################            

def main():
    """
    The main function for running the simulator, save and plot the results.
    """
    start_computation_time = time.process_time_ns()
    #file = "tests/dt01_s_ushape_5.txt"
    file = "F:/pathplanner1/pathplanner/tests/benchmarku8w/u8w00000001.txt"
    file_split = file.split('.')
    env_init = Grid(20,5,file)
    start = np.array((0.,0.))
    #start0 = np.array((0.,0.5))
    #startl = np.array((-5.,0.))
    #startb = np.array((0.,-5.))
    goal = np.array((20.,0.))
    #goal0 = np.array((10.,0.5))
    #goaltop = np.array((10.,3.))
    env = Map(20,5)
    env.from_grid_to_map(env_init, 0.)

    obstacles = env.obs_circ#tacles

    #classic1r = BabyRobot(start.copy(),0,1,0,1,0.,"classic1")
    #classic1 = Robot(classic1r)
    #classic1plan = FAPF(classic1.robot, classic1.robot.pos.copy(), goal.copy(), env, obstacles, 1., 1., 2., color_trace = (1,0,0), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
    #classic1.set_planner(classic1plan)
    #tangential1r = BabyRobot(start.copy(),0,1,0,1,0.,"tangential1")
    #tangential1 = Robot(tangential1r)
    #tangential1plan = FAPF(tangential1.robot, tangential1.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (0,1,0), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
    #tangential1.set_planner(tangential1plan)
    #iner1r = BabyRobot(start.copy(),0,1,0,1,0.,"inertial1")
    #iner1 = Robot(iner1r)
    #iner1plan = FAPF(iner1.robot, iner1.robot.pos.copy(), goal.copy(), env, obstacles, 1., 1., 2., color_trace = (0,0,1), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
    #iner1.set_planner(iner1plan)
    #taniner1r = BabyRobot(start.copy(),0,1,0,1,0.,"tangentialinertial1")
    #taniner1 = Robot(taniner1r)
    #taniner1plan = FAPF(taniner1.robot, taniner1.robot.pos.copy(), goal.copy(), env, obstacles, 1., 1., 2., color_trace = (0,0,0), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
    #taniner1.set_planner(taniner1plan)
    #dtangential1r = BabyRobot(start.copy(),0,1,0,1,0.1,"dtangential1")
    #dtangential1 = Robot(dtangential1r)
    #dtangential1plan = FDAPF(dtangential1.robot, dtangential1.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (1,0,1), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
    ##dtangential1plan = FAPF(dtangential1.robot, dtangential1.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (1,0,1), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
    #dtangential1.set_planner(dtangential1plan)
    ditbr = BabyRobot(start.copy(),0,1,0,1,0.1,"dit")
    ditr = Robot(ditbr)
    ditrplan = FDAPF(ditr.robot, ditr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (0.3,0.3,0.3), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
    ditr.set_planner(ditrplan)

    simulator = Simulator(env,0.1,60,0.1)
    #simulator.add_agent(classic1)
    #simulator.add_agent(tangential1)
    #simulator.add_agent(dtangential1)
    simulator.add_agent(ditr)
    #simulator.add_agent(iner1)
    #simulator.add_agent(taniner1)

    #simulator.run2(dtangential1)
    simulator.run2(ditr)
    simulator.compute_data()
    simulator.print_data()
    simulator.save_data(file_split[0]+"_out."+file_split[1])
    simulator.save_data_latex(file_split[0]+"_out.tex")
    end_computation_time = time.process_time_ns()
    print("Computation time: ", end_computation_time - start_computation_time)
    print(str(file) + ';' + str(ditr.path_length) + ';' + str(ditr.sum_angle[0]) + ';' + str(ditr.sum_angle[1]) + ';' + str(ditr.sum_angle[2]) 
          + ';' + str(ditr.max_angle) + ';' + str(ditr.time_to_reach_goal) + ';' + str(ditr.planner.comp_time) + ';' + str(ditr.planner.nb_collision[0]) + '\n')
    simulator.plot.animate("test")
    

def main2():
    """
    The main function for running the geneatic algorithm.
    """
    dic=[]
    subdic = []
    print("start")
    for i in range(20,100):
        subdic.append(i)
    dic.append(subdic.copy())
    subdic = []
    for j in range(100,300):
        subdic.append(j)
    dic.append(subdic.copy())
    subdic = []
    for k in range(1,10):
        subdic.append(k/10.)
    dic.append(subdic.copy())
    subdic = []
    for l in range(1,2):
        subdic.append(l)
    dic.append(subdic.copy())
    subdic = []
    for m in range(5,15):
        subdic.append(m)
    dic.append(subdic.copy())
    subdic = []
    for n in range(15,30):
        subdic.append(n)
    dic.append(subdic.copy())
    subdic = []
    gatest = GAparams(dic,20)
    print("start running")
    results = gatest.run()
    print("results: ", results)

def main3():
    start_computation_time = time.process_time_ns()
    rnggen = np.random.RandomState(849)
    rngsim = np.random.RandomState(564564)
    file = "tests/benchmark/0001.txt"
    file_split = file.split('.')
    x = 20
    y = 5
    Grid.write_random_env(file,x,y,20,rnggen)
    env = Grid(x,y,file)
    start = np.array((0.,1.))
    goal = np.array((10.,1.))
    random_robots_list = []
    random_agents_list = []
    random_planner_list = []
    apf_robots_list = []
    apf_agents_list = []
    apf_planner_list = []
    starts_ends = rnggen.random_sample((30,2))
    for i in range(30):
        starts_ends[i][0] = x*starts_ends[i][0]
        starts_ends[i][1] = y*starts_ends[i][1]
    obstacles = env.obstacles.copy()
    simulator = Simulator(env,0.05,40,0.1)
    for i in range(0,10):
        random_robots_list.append(BabyRobot(starts_ends[i].copy(),0,0,0,1,0,"r"+str(i)))
        random_agents_list.append(Robot(random_robots_list[i]))
        random_planner_list.append(InputPlanner(random_agents_list[i].robot,starts_ends[i].copy(),starts_ends[i+15].copy(),env,env.obstacles.copy(),0,0,acts_on_speed=True, rng=rngsim))
        random_agents_list[i].set_planner(random_planner_list[i])
        obstacles.append(random_robots_list[i].pos)
        simulator.add_agent(random_agents_list[i])
    for i in range(10,15):
        apf_robots_list.append(BabyRobot(starts_ends[i].copy(),0,0,0,1,0,"a"+str(i-10)))
        apf_agents_list.append(Robot(apf_robots_list[i-10]))
        apf_planner_list.append(FDAPF(apf_agents_list[i-10].robot, starts_ends[i].copy(), starts_ends[i+15].copy(), env, env.obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (1,0,1), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi))
        apf_agents_list[i-10].set_planner(apf_planner_list[i-10])
        obstacles.append(apf_robots_list[i-10].pos)
        simulator.add_agent(apf_agents_list[i-10])
    dtangential1r = BabyRobot(start.copy(),0,1,0,1,0.,"dtangential1")
    dtangential1 = Robot(dtangential1r)
    dtangential1plan = FDAPF(dtangential1.robot, dtangential1.robot.pos.copy(), goal.copy(), env, obstacles, 1., 4., 0.2, 1., 2., 4., color_trace = (0,1,0), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
    dtangential1.set_planner(dtangential1plan)
    simulator.add_agent(dtangential1)

    simulator.run()
    simulator.compute_data()
    simulator.print_data()
    end_computation_time = time.process_time_ns()
    print("Computation time: ", end_computation_time - start_computation_time)
    #simulator.save_data(file_split[0]+"_out."+file_split[1])
    #simulator.save_data_latex(file_split[0]+"_out.tex")
    simulator.plot.animate("test")

def main4():
    start_computation_time = time.process_time_ns()
    rnggen = np.random.RandomState(849)
    rngsim = np.random.RandomState(564564)
    file = "tests/benchmark/0001.txt"
    file_split = file.split('.')
    x = 20
    y = 5
    Grid.write_random_env(file,x,y,20,rnggen)
    env_init = Grid(x,y,file)
    env = Map(x,y)
    env.from_grid_to_map(env_init, 0.5)
    env.avoid_overlap()
    start = np.array((0.,1.))
    goal = np.array((10.,1.))
    sg_list = [Obstacle(start,1), Obstacle(goal,1)]
    random_robots_list = []
    random_agents_list = []
    random_planner_list = []
    apf_robots_list = []
    apf_agents_list = []
    apf_planner_list = []
    starts_ends = rnggen.random_sample((30,2))
    for i in range(30):
        starts_ends[i][0] = x*starts_ends[i][0]
        starts_ends[i][1] = y*starts_ends[i][1]
        sg_list.append(Obstacle(starts_ends[i],1))
    #obstacles = env.obstacles.copy()
    simulator = Simulator(env,0.05,80,0.1)
    obs_lists = [env.obs_circ,sg_list]
    simulator.avoid_overlap(obs_lists)
    obstacles = env.obs_circ.copy()
    #for i in range(0,10):
    #    random_robots_list.append(BabyRobot(starts_ends[i].copy(),0,0,0,1,1.,"r"+str(i)))
    #    random_agents_list.append(Robot(random_robots_list[i]))
    #    random_planner_list.append(InputPlanner(random_agents_list[i].robot,starts_ends[i].copy(),starts_ends[i+15].copy(),env,env.obs_circ.copy(),0,0,acts_on_speed=True, rng=rngsim))
    #    random_agents_list[i].set_planner(random_planner_list[i])
    #    obstacles.append(random_robots_list[i])
    #    simulator.add_agent(random_agents_list[i])
    #for i in range(10,15):
    #    apf_robots_list.append(BabyRobot(starts_ends[i].copy(),0,0,0,1,1.,"a"+str(i-10)))
    #    apf_agents_list.append(Robot(apf_robots_list[i-10]))
    #    apf_planner_list.append(FDAPF(apf_agents_list[i-10].robot, starts_ends[i].copy(), starts_ends[i+15].copy(), env, env.obs_circ.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (1,0,1), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi))
    #    apf_agents_list[i-10].set_planner(apf_planner_list[i-10])
    #    obstacles.append(apf_robots_list[i-10])
    #    simulator.add_agent(apf_agents_list[i-10])
    dtangential1r = BabyRobot(start.copy(),0,1,0,1,1.,"dtangential1")
    dtangential1 = Robot(dtangential1r)
    dtangential1plan = FDAPF(dtangential1.robot, dtangential1.robot.pos.copy(), goal.copy(), env, obstacles, 1., 4., 0.2, 1., 2., 4., color_trace = (0,1,0), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
    dtangential1.set_planner(dtangential1plan)
    simulator.add_agent(dtangential1)

    simulator.run2(dtangential1)
    simulator.compute_data()
    simulator.print_data()
    end_computation_time = time.process_time_ns()
    print("Computation time: ", end_computation_time - start_computation_time)
    #simulator.save_data(file_split[0]+"_out."+file_split[1])
    #simulator.save_data_latex(file_split[0]+"_out.tex")
    simulator.plot.animate("test")

def main5():
    lrpos = [0.05,0.2,0.5,0.8,0.95] #obstacle on left-right percentage
    btpos = [0.,-0.25,0.25] #obstacle on bot-top percentage
    dec = [0.,0.05] #broke symetry from on point in 0.1 to one in 0.05 and 0.15 (point to door)
    theta = [0.,math.pi/6,math.pi/4,math.pi/3,-math.pi/6,-math.pi/4,-math.pi/3,math.pi] #obstacle angle
    ls = [8.,6.,4.,3.,2.,1.] #length
    ws = [8.,6.,4.,3.,2.,1.] #width
    sap = [0.25,0.33,0.5,0.67,0.75,1.] #second arm size percentage
    lota = [-1,1] #who is the second arm (bot or top)
    test = 1 #test number
    start = np.array((0.,0.))
    goal = np.array((20.,0.))
    prec = 0.1
    t_lim = 60
    t_step = 0.1

    filec = "tests/benchmark/wallc.txt"
    fc = open(filec, 'w', encoding = "utf-8")
    fc.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fc.close()

    filet = "tests/benchmark/wallt.txt"
    ft = open(filet, 'w', encoding = "utf-8")
    ft.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    ft.close()

    filei = "tests/benchmark/walli.txt"
    fi = open(filei, 'w', encoding = "utf-8")
    fi.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fi.close()

    fileit = "tests/benchmark/wallit.txt"
    fit = open(fileit, 'w', encoding = "utf-8")
    fit.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fit.close()

    filedc = "tests/benchmark/walldc.txt"
    fdc = open(filedc, 'w', encoding = "utf-8")
    fdc.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdc.close()

    filedt = "tests/benchmark/walldt.txt"
    fdt = open(filedt, 'w', encoding = "utf-8")
    fdt.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdt.close()

    filedi = "tests/benchmark/walldi.txt"
    fdi = open(filedi, 'w', encoding = "utf-8")
    fdi.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdi.close()

    filedit = "tests/benchmark/walldit.txt"
    fdit = open(filedit, 'w', encoding = "utf-8")
    fdit.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdit.close()

    #walls
    for a in range(len(ls)):
        for b in range(len(lrpos)):
            for c in range(len(btpos)):
                for d in range(len(theta)):
                    for e in range(len(dec)):
                        file = "tests/benchmark/"
                        file = file + "wall" + f"{test:08}" + ".txt"
                        #testlist.append(file)
                        print(file)
                        file_split = file.split('.')
                        p1 = np.array((20.*lrpos[b],0.-ls[a]/2+ls[a]*btpos[c]-dec[e]))
                        p2 = np.array((20.*lrpos[b],0.+ls[a]/2+ls[a]*btpos[c]+dec[e]))
                        p = [p1,p2]
                        Map_Designer.make_poly_line(p,0.1,file,theta[d])
                        env_init = Grid(20,5,file)
                        env = Map(20,5)
                        env.from_grid_to_map(env_init, 0.)
                        obstacles = env.obs_circ

                        cbr = BabyRobot(start.copy(),0,1,0,1,0.1,"cr")
                        cr = Robot(cbr)
                        crplan = FAPF(cr.robot, cr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (1,0,0), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        cr.set_planner(crplan)
                        sc = Simulator(env,t_step,t_lim,prec)
                        sc.add_agent(cr)
                        sc.run2(cr)
                        sc.compute_data()
                        fc = open(filec, 'a', encoding = "utf-8")
                        fc.write(str(file) + ';' + str(cr.path_length) + ';' + str(cr.sum_angle[0]) + ';' + str(cr.sum_angle[1]) + ';' + str(cr.sum_angle[2]) 
                                 + ';' + str(cr.max_angle) + ';' + str(cr.time_to_reach_goal) + ';' + str(cr.planner.comp_time) + ';' + str(cr.planner.nb_collision[0]) + '\n')
                        fc.close()

                        tbr = BabyRobot(start.copy(),0,1,0,1,0.1,"tr")
                        tr = Robot(tbr)
                        trplan = FAPF(tr.robot, tr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (0,1,0), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        tr.set_planner(trplan)
                        st = Simulator(env,t_step,t_lim,prec)
                        st.add_agent(tr)
                        st.run2(tr)
                        st.compute_data()
                        ft = open(filet, 'a', encoding = "utf-8")
                        ft.write(str(file) + ';' + str(tr.path_length) + ';' + str(tr.sum_angle[0]) + ';' + str(tr.sum_angle[1]) + ';' + str(tr.sum_angle[2]) 
                                 + ';' + str(tr.max_angle) + ';' + str(tr.time_to_reach_goal) + ';' + str(tr.planner.comp_time) + ';' + str(tr.planner.nb_collision[0]) + '\n')
                        ft.close()

                        ibr = BabyRobot(start.copy(),0,1,0,1,0.1,"ir")
                        ir = Robot(ibr)
                        irplan = FAPF(ir.robot, ir.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (0,0,1), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        ir.set_planner(irplan)
                        si = Simulator(env,t_step,t_lim,prec)
                        si.add_agent(ir)
                        si.run2(ir)
                        si.compute_data()
                        fi = open(filei, 'a', encoding = "utf-8")
                        fi.write(str(file) + ';' + str(ir.path_length) + ';' + str(ir.sum_angle[0]) + ';' + str(ir.sum_angle[1]) + ';' + str(ir.sum_angle[2]) 
                                 + ';' + str(ir.max_angle) + ';' + str(ir.time_to_reach_goal) + ';' + str(ir.planner.comp_time) + ';' + str(ir.planner.nb_collision[0]) + '\n')
                        fi.close()

                        itbr = BabyRobot(start.copy(),0,1,0,1,0.1,"itr")
                        itr = Robot(itbr)
                        itrplan = FAPF(itr.robot, itr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (1,1,1), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        itr.set_planner(itrplan)
                        sit = Simulator(env,t_step,t_lim,prec)
                        sit.add_agent(itr)
                        sit.run2(itr)
                        sit.compute_data()
                        fit = open(fileit, 'a', encoding = "utf-8")
                        fit.write(str(file) + ';' + str(itr.path_length) + ';' + str(itr.sum_angle[0]) + ';' + str(itr.sum_angle[1]) + ';' + str(itr.sum_angle[2]) 
                                 + ';' + str(itr.max_angle) + ';' + str(itr.time_to_reach_goal) + ';' + str(itr.planner.comp_time) + ';' + str(itr.planner.nb_collision[0]) + '\n')
                        fit.close()

                        dcbr = BabyRobot(start.copy(),0,1,0,1,0.1,"dc")
                        dcr = Robot(dcbr)
                        dcrplan = FDAPF(dcr.robot, dcr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (0,1,1), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        dcr.set_planner(dcrplan)
                        sdc = Simulator(env,t_step,t_lim,prec)
                        sdc.add_agent(dcr)
                        sdc.run2(dcr)
                        sdc.compute_data()
                        fdc = open(filedc, 'a', encoding = "utf-8")
                        fdc.write(str(file) + ';' + str(dcr.path_length) + ';' + str(dcr.sum_angle[0]) + ';' + str(dcr.sum_angle[1]) + ';' + str(dcr.sum_angle[2]) 
                                 + ';' + str(dcr.max_angle) + ';' + str(dcr.time_to_reach_goal) + ';' + str(dcr.planner.comp_time) + ';' + str(dcr.planner.nb_collision[0]) + '\n')
                        fdc.close()

                        dtbr = BabyRobot(start.copy(),0,1,0,1,0.1,"dt")
                        dtr = Robot(dtbr)
                        dtrplan = FDAPF(dtr.robot, dtr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (1,0,1), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        dtr.set_planner(dtrplan)
                        sdt = Simulator(env,t_step,t_lim,prec)
                        sdt.add_agent(dtr)
                        sdt.run2(dtr)
                        sdt.compute_data()
                        fdt = open(filedt, 'a', encoding = "utf-8")
                        fdt.write(str(file) + ';' + str(dtr.path_length) + ';' + str(dtr.sum_angle[0]) + ';' + str(dtr.sum_angle[1]) + ';' + str(dtr.sum_angle[2]) 
                                 + ';' + str(dtr.max_angle) + ';' + str(dtr.time_to_reach_goal) + ';' + str(dtr.planner.comp_time) + ';' + str(dtr.planner.nb_collision[0]) + '\n')
                        fdt.close()

                        dibr = BabyRobot(start.copy(),0,1,0,1,0.1,"di")
                        dir = Robot(dibr)
                        dirplan = FDAPF(dir.robot, dir.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (1,1,0), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        dir.set_planner(dirplan)
                        sdi = Simulator(env,t_step,t_lim,prec)
                        sdi.add_agent(dir)
                        sdi.run2(dir)
                        sdi.compute_data()
                        fdi = open(filedi, 'a', encoding = "utf-8")
                        fdi.write(str(file) + ';' + str(dir.path_length) + ';' + str(dir.sum_angle[0]) + ';' + str(dir.sum_angle[1]) + ';' + str(dir.sum_angle[2]) 
                                 + ';' + str(dir.max_angle) + ';' + str(dir.time_to_reach_goal) + ';' + str(dir.planner.comp_time) + ';' + str(dir.planner.nb_collision[0]) + '\n')
                        fdi.close()

                        ditbr = BabyRobot(start.copy(),0,1,0,1,0.1,"dit")
                        ditr = Robot(ditbr)
                        ditrplan = FDAPF(ditr.robot, ditr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (0.3,0.3,0.3), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        ditr.set_planner(ditrplan)
                        sdit = Simulator(env,t_step,t_lim,prec)
                        sdit.add_agent(ditr)
                        sdit.run2(ditr)
                        sdit.compute_data()
                        fdit = open(filedit, 'a', encoding = "utf-8")
                        fdit.write(str(file) + ';' + str(ditr.path_length) + ';' + str(ditr.sum_angle[0]) + ';' + str(ditr.sum_angle[1]) + ';' + str(ditr.sum_angle[2]) 
                                 + ';' + str(ditr.max_angle) + ';' + str(ditr.time_to_reach_goal) + ';' + str(ditr.planner.comp_time) + ';' + str(ditr.planner.nb_collision[0]) + '\n')
                        fdit.close()

                        test += 1

def main6():
    lrpos = [0.05,0.2,0.5,0.8,0.95] #obstacle on left-right percentage
    btpos = [0.,-0.25,0.25] #obstacle on bot-top percentage
    dec = [0.,0.05] #broke symetry from on point in 0.1 to one in 0.05 and 0.15 (point to door)
    theta = [0.,math.pi/6,math.pi/4,math.pi/3,-math.pi/6,-math.pi/4,-math.pi/3,math.pi] #obstacle angle
    ls = [8.,6.,4.,3.,2.,1.] #length
    ws = [8.,6.,4.,3.,2.,1.] #width
    sap = [0.25,0.33,0.5,0.67,0.75,1.] #second arm size percentage
    lota = [0,1] #who is the second arm (bot or top)
    test = 1 #test number
    start = np.array((0.,0.))
    goal = np.array((20.,0.))
    prec = 0.1
    t_lim = 60
    t_step = 0.1

    filec = "tests/benchmarku/uc.txt"
    fc = open(filec, 'w', encoding = "utf-8")
    fc.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fc.close()

    filet = "tests/benchmarku/ut.txt"
    ft = open(filet, 'w', encoding = "utf-8")
    ft.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    ft.close()

    filei = "tests/benchmarku/ui.txt"
    fi = open(filei, 'w', encoding = "utf-8")
    fi.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fi.close()

    fileit = "tests/benchmarku/uit.txt"
    fit = open(fileit, 'w', encoding = "utf-8")
    fit.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fit.close()

    filedc = "tests/benchmarku/udc.txt"
    fdc = open(filedc, 'w', encoding = "utf-8")
    fdc.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdc.close()

    filedt = "tests/benchmarku/udt.txt"
    fdt = open(filedt, 'w', encoding = "utf-8")
    fdt.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdt.close()

    filedi = "tests/benchmarku/udi.txt"
    fdi = open(filedi, 'w', encoding = "utf-8")
    fdi.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdi.close()

    filedit = "tests/benchmarku/udit.txt"
    fdit = open(filedit, 'w', encoding = "utf-8")
    fdit.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdit.close()

    #walls
    for a in range(len(ls)):
        for b in range(len(ws)):
            for c in range(len(sap)):
                for d in range(len(lota)):
                    for e in range(len(lrpos)):
                        for f in range(len(btpos)):
                            for g in range(len(theta)):
                                for h in range(len(dec)):
                                    file = "tests/benchmarku/"
                                    file = file + "u" + f"{test:08}" + ".txt"
                                    #testlist.append(file)
                                    print(file)
                                    file_split = file.split('.')
                                    p1 = np.array((20.*lrpos[e]+ls[a]/2-ls[a] if lota[d]==1 else 20.*lrpos[e]+ls[a]/2-ls[a]*sap[c], 0.-ws[b]/2+ws[b]*btpos[f]-dec[h])) #bot left
                                    p2 = np.array((20.*lrpos[e]+ls[a]/2, 0.-ws[b]/2+ws[b]*btpos[f]-dec[h])) #bot right
                                    p3 = np.array((20.*lrpos[e]+ls[a]/2, 0.+ws[b]/2+ws[b]*btpos[f]+dec[h])) #top right
                                    p4 = np.array((20.*lrpos[e]+ls[a]/2-ls[a] if lota[d]==0 else 20.*lrpos[e]+ls[a]/2-ls[a]*sap[c], 0.+ws[b]/2+ws[b]*btpos[f]+dec[h])) #top left
                                    p = [p1,p2,p3,p4]
                                    Map_Designer.make_poly_line(p,0.1,file,theta[g])
                                    env_init = Grid(20,5,file)
                                    env = Map(20,5)
                                    env.from_grid_to_map(env_init, 0.)
                                    obstacles = env.obs_circ

                                    cbr = BabyRobot(start.copy(),0,1,0,1,0.1,"cr")
                                    cr = Robot(cbr)
                                    crplan = FAPF(cr.robot, cr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (1,0,0), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    cr.set_planner(crplan)

                                    tbr = BabyRobot(start.copy(),0,1,0,1,0.1,"tr")
                                    tr = Robot(tbr)
                                    trplan = FAPF(tr.robot, tr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (0,1,0), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    tr.set_planner(trplan)

                                    ibr = BabyRobot(start.copy(),0,1,0,1,0.1,"ir")
                                    ir = Robot(ibr)
                                    irplan = FAPF(ir.robot, ir.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (0,0,1), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    ir.set_planner(irplan)

                                    itbr = BabyRobot(start.copy(),0,1,0,1,0.1,"itr")
                                    itr = Robot(itbr)
                                    itrplan = FAPF(itr.robot, itr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (1,1,1), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    itr.set_planner(itrplan)

                                    dcbr = BabyRobot(start.copy(),0,1,0,1,0.1,"dc")
                                    dcr = Robot(dcbr)
                                    dcrplan = FDAPF(dcr.robot, dcr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (0,1,1), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    dcr.set_planner(dcrplan)

                                    dtbr = BabyRobot(start.copy(),0,1,0,1,0.1,"dt")
                                    dtr = Robot(dtbr)
                                    dtrplan = FDAPF(dtr.robot, dtr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (1,0,1), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    dtr.set_planner(dtrplan)

                                    dibr = BabyRobot(start.copy(),0,1,0,1,0.1,"di")
                                    dir = Robot(dibr)
                                    dirplan = FDAPF(dir.robot, dir.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (1,1,0), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    dir.set_planner(dirplan)

                                    ditbr = BabyRobot(start.copy(),0,1,0,1,0.1,"dit")
                                    ditr = Robot(ditbr)
                                    ditrplan = FDAPF(ditr.robot, ditr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (0.3,0.3,0.3), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    ditr.set_planner(ditrplan)

                                    sc = Simulator(env,t_step,t_lim,prec)
                                    sc.add_agent(cr)
                                    sc.run2(cr)
                                    sc.compute_data()
                                    fc = open(filec, 'a', encoding = "utf-8")
                                    fc.write(str(file) + ';' + str(cr.path_length) + ';' + str(cr.sum_angle[0]) + ';' + str(cr.sum_angle[1]) + ';' + str(cr.sum_angle[2]) 
                                             + ';' + str(cr.max_angle) + ';' + str(cr.time_to_reach_goal) + ';' + str(cr.planner.comp_time) + ';' + str(cr.planner.nb_collision[0]) + '\n')
                                    fc.close()

                                    st = Simulator(env,t_step,t_lim,prec)
                                    st.add_agent(tr)
                                    st.run2(tr)
                                    st.compute_data()
                                    ft = open(filet, 'a', encoding = "utf-8")
                                    ft.write(str(file) + ';' + str(tr.path_length) + ';' + str(tr.sum_angle[0]) + ';' + str(tr.sum_angle[1]) + ';' + str(tr.sum_angle[2]) 
                                             + ';' + str(tr.max_angle) + ';' + str(tr.time_to_reach_goal) + ';' + str(tr.planner.comp_time) + ';' + str(tr.planner.nb_collision[0]) + '\n')
                                    ft.close()

                                    si = Simulator(env,t_step,t_lim,prec)
                                    si.add_agent(ir)
                                    si.run2(ir)
                                    si.compute_data()
                                    fi = open(filei, 'a', encoding = "utf-8")
                                    fi.write(str(file) + ';' + str(ir.path_length) + ';' + str(ir.sum_angle[0]) + ';' + str(ir.sum_angle[1]) + ';' + str(ir.sum_angle[2]) 
                                             + ';' + str(ir.max_angle) + ';' + str(ir.time_to_reach_goal) + ';' + str(ir.planner.comp_time) + ';' + str(ir.planner.nb_collision[0]) + '\n')
                                    fi.close()

                                    sit = Simulator(env,t_step,t_lim,prec)
                                    sit.add_agent(itr)
                                    sit.run2(itr)
                                    sit.compute_data()
                                    fit = open(fileit, 'a', encoding = "utf-8")
                                    fit.write(str(file) + ';' + str(itr.path_length) + ';' + str(itr.sum_angle[0]) + ';' + str(itr.sum_angle[1]) + ';' + str(itr.sum_angle[2]) 
                                             + ';' + str(itr.max_angle) + ';' + str(itr.time_to_reach_goal) + ';' + str(itr.planner.comp_time) + ';' + str(itr.planner.nb_collision[0]) + '\n')
                                    fit.close()

                                    sdc = Simulator(env,t_step,t_lim,prec)
                                    sdc.add_agent(dcr)
                                    sdc.run2(dcr)
                                    sdc.compute_data()
                                    fdc = open(filedc, 'a', encoding = "utf-8")
                                    fdc.write(str(file) + ';' + str(dcr.path_length) + ';' + str(dcr.sum_angle[0]) + ';' + str(dcr.sum_angle[1]) + ';' + str(dcr.sum_angle[2]) 
                                             + ';' + str(dcr.max_angle) + ';' + str(dcr.time_to_reach_goal) + ';' + str(dcr.planner.comp_time) + ';' + str(dcr.planner.nb_collision[0]) + '\n')
                                    fdc.close()

                                    sdt = Simulator(env,t_step,t_lim,prec)
                                    sdt.add_agent(dtr)
                                    sdt.run2(dtr)
                                    sdt.compute_data()
                                    fdt = open(filedt, 'a', encoding = "utf-8")
                                    fdt.write(str(file) + ';' + str(dtr.path_length) + ';' + str(dtr.sum_angle[0]) + ';' + str(dtr.sum_angle[1]) + ';' + str(dtr.sum_angle[2]) 
                                             + ';' + str(dtr.max_angle) + ';' + str(dtr.time_to_reach_goal) + ';' + str(dtr.planner.comp_time) + ';' + str(dtr.planner.nb_collision[0]) + '\n')
                                    fdt.close()

                                    sdi = Simulator(env,t_step,t_lim,prec)
                                    sdi.add_agent(dir)
                                    sdi.run2(dir)
                                    sdi.compute_data()
                                    fdi = open(filedi, 'a', encoding = "utf-8")
                                    fdi.write(str(file) + ';' + str(dir.path_length) + ';' + str(dir.sum_angle[0]) + ';' + str(dir.sum_angle[1]) + ';' + str(dir.sum_angle[2]) 
                                             + ';' + str(dir.max_angle) + ';' + str(dir.time_to_reach_goal) + ';' + str(dir.planner.comp_time) + ';' + str(dir.planner.nb_collision[0]) + '\n')
                                    fdi.close()

                                    sdit = Simulator(env,t_step,t_lim,prec)
                                    sdit.add_agent(ditr)
                                    sdit.run2(ditr)
                                    sdit.compute_data()
                                    fdit = open(filedit, 'a', encoding = "utf-8")
                                    fdit.write(str(file) + ';' + str(ditr.path_length) + ';' + str(ditr.sum_angle[0]) + ';' + str(ditr.sum_angle[1]) + ';' + str(ditr.sum_angle[2]) 
                                             + ';' + str(ditr.max_angle) + ';' + str(ditr.time_to_reach_goal) + ';' + str(ditr.planner.comp_time) + ';' + str(ditr.planner.nb_collision[0]) + '\n')
                                    fdit.close()

                                    test += 1


def main8():
    lrpos = [0.05,0.2,0.5,0.8,0.95] #obstacle on left-right percentage
    btpos = [0.,-0.25,0.25] #obstacle on bot-top percentage
    dec = [0.,0.05] #broke symetry from on point in 0.1 to one in 0.05 and 0.15 (point to door)
    theta = [0.,math.pi/6,math.pi/4,math.pi/3,-math.pi/6,-math.pi/4,-math.pi/3,math.pi] #obstacle angle
    ls = [8.,6.,4.,3.,2.,1.] #length
    ws = [8.,6.,4.,3.,2.,1.] #width
    sap = [0.25,0.33,0.5,0.67,0.75,1.] #second arm size percentage
    lota = [0,1] #who is the second arm (bot or top)
    test = 1 #test number
    start = np.array((0.,0.))
    goal = np.array((20.,0.))
    prec = 0.1
    t_lim = 60
    t_step = 0.1

    filec = "tests/benchmarkv/uc.txt"
    fc = open(filec, 'w', encoding = "utf-8")
    fc.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fc.close()

    filet = "tests/benchmarkv/ut.txt"
    ft = open(filet, 'w', encoding = "utf-8")
    ft.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    ft.close()

    filei = "tests/benchmarkv/ui.txt"
    fi = open(filei, 'w', encoding = "utf-8")
    fi.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fi.close()

    fileit = "tests/benchmarkv/uit.txt"
    fit = open(fileit, 'w', encoding = "utf-8")
    fit.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fit.close()

    filedc = "tests/benchmarkv/udc.txt"
    fdc = open(filedc, 'w', encoding = "utf-8")
    fdc.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdc.close()

    filedt = "tests/benchmarkv/udt.txt"
    fdt = open(filedt, 'w', encoding = "utf-8")
    fdt.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdt.close()

    filedi = "tests/benchmarkv/udi.txt"
    fdi = open(filedi, 'w', encoding = "utf-8")
    fdi.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdi.close()

    filedit = "tests/benchmarkv/udit.txt"
    fdit = open(filedit, 'w', encoding = "utf-8")
    fdit.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdit.close()

    #walls
    for a in range(len(ls)):
        for b in range(len(ws)):
            for c in range(len(sap)):
                for d in range(len(lota)):
                    for e in range(len(lrpos)):
                        for f in range(len(btpos)):
                            for g in range(len(theta)):
                                for h in range(len(dec)):
                                    file = "tests/benchmarkv/"
                                    file = file + "v" + f"{test:08}" + ".txt"
                                    #testlist.append(file)
                                    print(file)
                                    file_split = file.split('.')
                                    p1 = np.array((20.*lrpos[e]+ls[a]/2-ls[a] if lota[d]==1 else 20.*lrpos[e]+ls[a]/2-ls[a]*sap[c], 0.-ws[b]/2+ws[b]*btpos[f]-dec[h])) #bot left
                                    p2 = np.array((20.*lrpos[e]+ls[a]/2, 0.+ws[b]*btpos[f]-dec[h])) #bot right
                                    p3 = np.array((20.*lrpos[e]+ls[a]/2, 0.+ws[b]*btpos[f]+dec[h])) #top right
                                    p4 = np.array((20.*lrpos[e]+ls[a]/2-ls[a] if lota[d]==0 else 20.*lrpos[e]+ls[a]/2-ls[a]*sap[c], 0.+ws[b]/2+ws[b]*btpos[f]+dec[h])) #top left
                                    p = [p1,p2,p3,p4]
                                    Map_Designer.make_poly_line(p,0.1,file,theta[g])
                                    env_init = Grid(20,5,file)
                                    env = Map(20,5)
                                    env.from_grid_to_map(env_init, 0.)
                                    obstacles = env.obs_circ

                                    cbr = BabyRobot(start.copy(),0,1,0,1,0.1,"cr")
                                    cr = Robot(cbr)
                                    crplan = FAPF(cr.robot, cr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (1,0,0), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    cr.set_planner(crplan)

                                    tbr = BabyRobot(start.copy(),0,1,0,1,0.1,"tr")
                                    tr = Robot(tbr)
                                    trplan = FAPF(tr.robot, tr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (0,1,0), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    tr.set_planner(trplan)

                                    ibr = BabyRobot(start.copy(),0,1,0,1,0.1,"ir")
                                    ir = Robot(ibr)
                                    irplan = FAPF(ir.robot, ir.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (0,0,1), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    ir.set_planner(irplan)

                                    itbr = BabyRobot(start.copy(),0,1,0,1,0.1,"itr")
                                    itr = Robot(itbr)
                                    itrplan = FAPF(itr.robot, itr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (1,1,1), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    itr.set_planner(itrplan)

                                    dcbr = BabyRobot(start.copy(),0,1,0,1,0.1,"dc")
                                    dcr = Robot(dcbr)
                                    dcrplan = FDAPF(dcr.robot, dcr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (0,1,1), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    dcr.set_planner(dcrplan)

                                    dtbr = BabyRobot(start.copy(),0,1,0,1,0.1,"dt")
                                    dtr = Robot(dtbr)
                                    dtrplan = FDAPF(dtr.robot, dtr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (1,0,1), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    dtr.set_planner(dtrplan)

                                    dibr = BabyRobot(start.copy(),0,1,0,1,0.1,"di")
                                    dir = Robot(dibr)
                                    dirplan = FDAPF(dir.robot, dir.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (1,1,0), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    dir.set_planner(dirplan)

                                    ditbr = BabyRobot(start.copy(),0,1,0,1,0.1,"dit")
                                    ditr = Robot(ditbr)
                                    ditrplan = FDAPF(ditr.robot, ditr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (0.3,0.3,0.3), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                                    ditr.set_planner(ditrplan)

                                    sc = Simulator(env,t_step,t_lim,prec)
                                    sc.add_agent(cr)
                                    sc.run2(cr)
                                    sc.compute_data()
                                    fc = open(filec, 'a', encoding = "utf-8")
                                    fc.write(str(file) + ';' + str(cr.path_length) + ';' + str(cr.sum_angle[0]) + ';' + str(cr.sum_angle[1]) + ';' + str(cr.sum_angle[2]) 
                                             + ';' + str(cr.max_angle) + ';' + str(cr.time_to_reach_goal) + ';' + str(cr.planner.comp_time) + ';' + str(cr.planner.nb_collision[0]) + '\n')
                                    fc.close()

                                    st = Simulator(env,t_step,t_lim,prec)
                                    st.add_agent(tr)
                                    st.run2(tr)
                                    st.compute_data()
                                    ft = open(filet, 'a', encoding = "utf-8")
                                    ft.write(str(file) + ';' + str(tr.path_length) + ';' + str(tr.sum_angle[0]) + ';' + str(tr.sum_angle[1]) + ';' + str(tr.sum_angle[2]) 
                                             + ';' + str(tr.max_angle) + ';' + str(tr.time_to_reach_goal) + ';' + str(tr.planner.comp_time) + ';' + str(tr.planner.nb_collision[0]) + '\n')
                                    ft.close()

                                    si = Simulator(env,t_step,t_lim,prec)
                                    si.add_agent(ir)
                                    si.run2(ir)
                                    si.compute_data()
                                    fi = open(filei, 'a', encoding = "utf-8")
                                    fi.write(str(file) + ';' + str(ir.path_length) + ';' + str(ir.sum_angle[0]) + ';' + str(ir.sum_angle[1]) + ';' + str(ir.sum_angle[2]) 
                                             + ';' + str(ir.max_angle) + ';' + str(ir.time_to_reach_goal) + ';' + str(ir.planner.comp_time) + ';' + str(ir.planner.nb_collision[0]) + '\n')
                                    fi.close()

                                    sit = Simulator(env,t_step,t_lim,prec)
                                    sit.add_agent(itr)
                                    sit.run2(itr)
                                    sit.compute_data()
                                    fit = open(fileit, 'a', encoding = "utf-8")
                                    fit.write(str(file) + ';' + str(itr.path_length) + ';' + str(itr.sum_angle[0]) + ';' + str(itr.sum_angle[1]) + ';' + str(itr.sum_angle[2]) 
                                             + ';' + str(itr.max_angle) + ';' + str(itr.time_to_reach_goal) + ';' + str(itr.planner.comp_time) + ';' + str(itr.planner.nb_collision[0]) + '\n')
                                    fit.close()

                                    sdc = Simulator(env,t_step,t_lim,prec)
                                    sdc.add_agent(dcr)
                                    sdc.run2(dcr)
                                    sdc.compute_data()
                                    fdc = open(filedc, 'a', encoding = "utf-8")
                                    fdc.write(str(file) + ';' + str(dcr.path_length) + ';' + str(dcr.sum_angle[0]) + ';' + str(dcr.sum_angle[1]) + ';' + str(dcr.sum_angle[2]) 
                                             + ';' + str(dcr.max_angle) + ';' + str(dcr.time_to_reach_goal) + ';' + str(dcr.planner.comp_time) + ';' + str(dcr.planner.nb_collision[0]) + '\n')
                                    fdc.close()

                                    sdt = Simulator(env,t_step,t_lim,prec)
                                    sdt.add_agent(dtr)
                                    sdt.run2(dtr)
                                    sdt.compute_data()
                                    fdt = open(filedt, 'a', encoding = "utf-8")
                                    fdt.write(str(file) + ';' + str(dtr.path_length) + ';' + str(dtr.sum_angle[0]) + ';' + str(dtr.sum_angle[1]) + ';' + str(dtr.sum_angle[2]) 
                                             + ';' + str(dtr.max_angle) + ';' + str(dtr.time_to_reach_goal) + ';' + str(dtr.planner.comp_time) + ';' + str(dtr.planner.nb_collision[0]) + '\n')
                                    fdt.close()

                                    sdi = Simulator(env,t_step,t_lim,prec)
                                    sdi.add_agent(dir)
                                    sdi.run2(dir)
                                    sdi.compute_data()
                                    fdi = open(filedi, 'a', encoding = "utf-8")
                                    fdi.write(str(file) + ';' + str(dir.path_length) + ';' + str(dir.sum_angle[0]) + ';' + str(dir.sum_angle[1]) + ';' + str(dir.sum_angle[2]) 
                                             + ';' + str(dir.max_angle) + ';' + str(dir.time_to_reach_goal) + ';' + str(dir.planner.comp_time) + ';' + str(dir.planner.nb_collision[0]) + '\n')
                                    fdi.close()

                                    sdit = Simulator(env,t_step,t_lim,prec)
                                    sdit.add_agent(ditr)
                                    sdit.run2(ditr)
                                    sdit.compute_data()
                                    fdit = open(filedit, 'a', encoding = "utf-8")
                                    fdit.write(str(file) + ';' + str(ditr.path_length) + ';' + str(ditr.sum_angle[0]) + ';' + str(ditr.sum_angle[1]) + ';' + str(ditr.sum_angle[2]) 
                                             + ';' + str(ditr.max_angle) + ';' + str(ditr.time_to_reach_goal) + ';' + str(ditr.planner.comp_time) + ';' + str(ditr.planner.nb_collision[0]) + '\n')
                                    fdit.close()

                                    test += 1

def main9():
    lrpos = [0.05,0.2,0.5,0.8,0.95] #obstacle on left-right percentage
    btpos = [0.,-0.25,0.25] #obstacle on bot-top percentage
    dec = [0.,0.05] #broke symetry from on point in 0.1 to one in 0.05 and 0.15 (point to door)
    theta = [0.,math.pi/6,math.pi/4,math.pi/3,-math.pi/6,-math.pi/4,-math.pi/3,math.pi] #obstacle angle
    ls = [8.,6.,4.,3.,2.,1.] #length
    ws = [8.,6.,4.,3.,2.,1.] #width
    sap = [0.25,0.33,0.5,0.67,0.75,1.] #second arm size percentage
    lota = [0,1] #who is the second arm (bot or top)
    test = 1 #test number
    start = np.array((0.,0.))
    goal = np.array((20.,0.))
    prec = 0.1
    t_lim = 60
    t_step = 0.1

    filec = "tests/benchmarke/uc.txt"
    fc = open(filec, 'w', encoding = "utf-8")
    fc.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fc.close()

    filet = "tests/benchmarke/ut.txt"
    ft = open(filet, 'w', encoding = "utf-8")
    ft.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    ft.close()

    filei = "tests/benchmarke/ui.txt"
    fi = open(filei, 'w', encoding = "utf-8")
    fi.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fi.close()

    fileit = "tests/benchmarke/uit.txt"
    fit = open(fileit, 'w', encoding = "utf-8")
    fit.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fit.close()

    filedc = "tests/benchmarke/udc.txt"
    fdc = open(filedc, 'w', encoding = "utf-8")
    fdc.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdc.close()

    filedt = "tests/benchmarke/udt.txt"
    fdt = open(filedt, 'w', encoding = "utf-8")
    fdt.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdt.close()

    filedi = "tests/benchmarke/udi.txt"
    fdi = open(filedi, 'w', encoding = "utf-8")
    fdi.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdi.close()

    filedit = "tests/benchmarke/udit.txt"
    fdit = open(filedit, 'w', encoding = "utf-8")
    fdit.write("name;path length;sum angles tot;sum angles left;sum angles right;max angle;time to reach the goal;computation time;number of collisions\n")
    fdit.close()

    #walls
    for a in range(len(ls)):
        for b in range(len(ws)):
            for e in range(len(lrpos)):
                for f in range(len(btpos)):
                    for g in range(len(theta)):
                        file = "tests/benchmarke/"
                        file = file + "e" + f"{test:08}" + ".txt"
                        #testlist.append(file)
                        print(file)
                        file_split = file.split('.')
                        p = np.array((lrpos[e]*20,btpos[f]*ws[b]))
                        paramt =(0.1/0.124998728438125)*(ls[a]+ws[b])/(16*(ls[a]*ws[b]))
                        Map_Designer.make_ellipse(ls[a],ws[b]/2,p,paramt,file,theta[g])
                        env_init = Grid(20,5,file)
                        env = Map(20,5)
                        env.from_grid_to_map(env_init, 0.)
                        obstacles = env.obs_circ

                        cbr = BabyRobot(start.copy(),0,1,0,1,0.1,"cr")
                        cr = Robot(cbr)
                        crplan = FAPF(cr.robot, cr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (1,0,0), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        cr.set_planner(crplan)

                        tbr = BabyRobot(start.copy(),0,1,0,1,0.1,"tr")
                        tr = Robot(tbr)
                        trplan = FAPF(tr.robot, tr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (0,1,0), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        tr.set_planner(trplan)

                        ibr = BabyRobot(start.copy(),0,1,0,1,0.1,"ir")
                        ir = Robot(ibr)
                        irplan = FAPF(ir.robot, ir.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (0,0,1), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        ir.set_planner(irplan)

                        itbr = BabyRobot(start.copy(),0,1,0,1,0.1,"itr")
                        itr = Robot(itbr)
                        itrplan = FAPF(itr.robot, itr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 1., 2., color_trace = (1,1,1), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        itr.set_planner(itrplan)

                        dcbr = BabyRobot(start.copy(),0,1,0,1,0.1,"dc")
                        dcr = Robot(dcbr)
                        dcrplan = FDAPF(dcr.robot, dcr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (0,1,1), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        dcr.set_planner(dcrplan)

                        dtbr = BabyRobot(start.copy(),0,1,0,1,0.1,"dt")
                        dtr = Robot(dtbr)
                        dtrplan = FDAPF(dtr.robot, dtr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (1,0,1), linear_repulsion = False, lookahead_dist = 4., inertia = False, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        dtr.set_planner(dtrplan)

                        dibr = BabyRobot(start.copy(),0,1,0,1,0.1,"di")
                        dir = Robot(dibr)
                        dirplan = FDAPF(dir.robot, dir.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (1,1,0), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = False, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        dir.set_planner(dirplan)

                        ditbr = BabyRobot(start.copy(),0,1,0,1,0.1,"dit")
                        ditr = Robot(ditbr)
                        ditrplan = FDAPF(ditr.robot, ditr.robot.pos.copy(), goal.copy(), env, obstacles.copy(), 1., 4., 0.2, 1., 2., 4., color_trace = (0.3,0.3,0.3), linear_repulsion = False, lookahead_dist = 4., inertia = True, tangential = True, attr_look = True, acts_on_speed = True, angle_detection = math.pi)
                        ditr.set_planner(ditrplan)

                        sc = Simulator(env,t_step,t_lim,prec)
                        sc.add_agent(cr)
                        sc.run2(cr)
                        sc.compute_data()
                        fc = open(filec, 'a', encoding = "utf-8")
                        fc.write(str(file) + ';' + str(cr.path_length) + ';' + str(cr.sum_angle[0]) + ';' + str(cr.sum_angle[1]) + ';' + str(cr.sum_angle[2]) 
                                    + ';' + str(cr.max_angle) + ';' + str(cr.time_to_reach_goal) + ';' + str(cr.planner.comp_time) + ';' + str(cr.planner.nb_collision[0]) + '\n')
                        fc.close()

                        st = Simulator(env,t_step,t_lim,prec)
                        st.add_agent(tr)
                        st.run2(tr)
                        st.compute_data()
                        ft = open(filet, 'a', encoding = "utf-8")
                        ft.write(str(file) + ';' + str(tr.path_length) + ';' + str(tr.sum_angle[0]) + ';' + str(tr.sum_angle[1]) + ';' + str(tr.sum_angle[2]) 
                                    + ';' + str(tr.max_angle) + ';' + str(tr.time_to_reach_goal) + ';' + str(tr.planner.comp_time) + ';' + str(tr.planner.nb_collision[0]) + '\n')
                        ft.close()

                        si = Simulator(env,t_step,t_lim,prec)
                        si.add_agent(ir)
                        si.run2(ir)
                        si.compute_data()
                        fi = open(filei, 'a', encoding = "utf-8")
                        fi.write(str(file) + ';' + str(ir.path_length) + ';' + str(ir.sum_angle[0]) + ';' + str(ir.sum_angle[1]) + ';' + str(ir.sum_angle[2]) 
                                    + ';' + str(ir.max_angle) + ';' + str(ir.time_to_reach_goal) + ';' + str(ir.planner.comp_time) + ';' + str(ir.planner.nb_collision[0]) + '\n')
                        fi.close()

                        sit = Simulator(env,t_step,t_lim,prec)
                        sit.add_agent(itr)
                        sit.run2(itr)
                        sit.compute_data()
                        fit = open(fileit, 'a', encoding = "utf-8")
                        fit.write(str(file) + ';' + str(itr.path_length) + ';' + str(itr.sum_angle[0]) + ';' + str(itr.sum_angle[1]) + ';' + str(itr.sum_angle[2]) 
                                    + ';' + str(itr.max_angle) + ';' + str(itr.time_to_reach_goal) + ';' + str(itr.planner.comp_time) + ';' + str(itr.planner.nb_collision[0]) + '\n')
                        fit.close()

                        sdc = Simulator(env,t_step,t_lim,prec)
                        sdc.add_agent(dcr)
                        sdc.run2(dcr)
                        sdc.compute_data()
                        fdc = open(filedc, 'a', encoding = "utf-8")
                        fdc.write(str(file) + ';' + str(dcr.path_length) + ';' + str(dcr.sum_angle[0]) + ';' + str(dcr.sum_angle[1]) + ';' + str(dcr.sum_angle[2]) 
                                    + ';' + str(dcr.max_angle) + ';' + str(dcr.time_to_reach_goal) + ';' + str(dcr.planner.comp_time) + ';' + str(dcr.planner.nb_collision[0]) + '\n')
                        fdc.close()

                        sdt = Simulator(env,t_step,t_lim,prec)
                        sdt.add_agent(dtr)
                        sdt.run2(dtr)
                        sdt.compute_data()
                        fdt = open(filedt, 'a', encoding = "utf-8")
                        fdt.write(str(file) + ';' + str(dtr.path_length) + ';' + str(dtr.sum_angle[0]) + ';' + str(dtr.sum_angle[1]) + ';' + str(dtr.sum_angle[2]) 
                                    + ';' + str(dtr.max_angle) + ';' + str(dtr.time_to_reach_goal) + ';' + str(dtr.planner.comp_time) + ';' + str(dtr.planner.nb_collision[0]) + '\n')
                        fdt.close()

                        sdi = Simulator(env,t_step,t_lim,prec)
                        sdi.add_agent(dir)
                        sdi.run2(dir)
                        sdi.compute_data()
                        fdi = open(filedi, 'a', encoding = "utf-8")
                        fdi.write(str(file) + ';' + str(dir.path_length) + ';' + str(dir.sum_angle[0]) + ';' + str(dir.sum_angle[1]) + ';' + str(dir.sum_angle[2]) 
                                    + ';' + str(dir.max_angle) + ';' + str(dir.time_to_reach_goal) + ';' + str(dir.planner.comp_time) + ';' + str(dir.planner.nb_collision[0]) + '\n')
                        fdi.close()

                        sdit = Simulator(env,t_step,t_lim,prec)
                        sdit.add_agent(ditr)
                        sdit.run2(ditr)
                        sdit.compute_data()
                        fdit = open(filedit, 'a', encoding = "utf-8")
                        fdit.write(str(file) + ';' + str(ditr.path_length) + ';' + str(ditr.sum_angle[0]) + ';' + str(ditr.sum_angle[1]) + ';' + str(ditr.sum_angle[2]) 
                                    + ';' + str(ditr.max_angle) + ';' + str(ditr.time_to_reach_goal) + ';' + str(ditr.planner.comp_time) + ';' + str(ditr.planner.nb_collision[0]) + '\n')
                        fdit.close()

                        test += 1

if __name__ == '__main__':
    main()