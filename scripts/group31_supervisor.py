#!/usr/bin/env python3

# Run this file by:
# python3 group31_supervisor.py grep -v TF_REPEATED_DATA buffer_core

import rospy
from aa274_final_project.msg import DetectedObject
from nav_msgs.msg import OccupancyGrid, MapMetaData
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from visualization_msgs.msg import Marker
import tf
import time
from enum import Enum
import numpy as np
from utils.grids import StochOccupancyGrid2D
from planners import solve_tsp_from_map
from std_msgs.msg import Bool
from utils import wrapToPi

class Mode(Enum):
    """State machine modes. Feel free to change."""
    IDLE = 1
    NAV = 2
    MANUAL= 3
    SELECT_WAYPOINT = 4
    STOP = 5

class SupervisorParams:

    def __init__(self, verbose=False):
        # If sim is True (i.e. using gazebo), we want to subscribe to
        # /gazebo/model_states. Otherwise, we will use a TF lookup.
        self.use_gazebo = rospy.get_param("sim")

        # How is nav_cmd being decided -- human manually setting it, or rviz
        self.rviz = rospy.get_param("rviz")

        # If using gmapping, we will have a map frame. Otherwise, it will be odom frame.
        self.mapping = rospy.get_param("map")

        # Threshold at which we consider the robot at a location
        # TODO: Make sure these match with the navigator epsilons
        self.pos_eps = rospy.get_param("~pos_eps", 0.25)
        self.theta_eps = rospy.get_param("~theta_eps", 0.4)

        # Time to stop at a stop sign
        self.stop_time = rospy.get_param("~stop_time", 2.)

        # Minimum distance from a stop sign to obey it
        self.stop_min_dist = rospy.get_param("~stop_min_dist", 0.5) # was formerly 0.3m

        # Time taken to cross an intersection
        self.crossing_time = rospy.get_param("~crossing_time", 3.)

        if verbose:
            print("SupervisorParams:")
            print("    use_gazebo = {}".format(self.use_gazebo))
            print("    rviz = {}".format(self.rviz))
            print("    mapping = {}".format(self.mapping))
            print("    pos_eps, theta_eps = {}, {}".format(self.pos_eps, self.theta_eps))
            print("    stop_time, stop_min_dist, crossing_time = {}, {}, {}".format(self.stop_time, self.stop_min_dist, self.crossing_time))

class Supervisor:
    def __init__(self):

        ########## VARIABLES ##########

        # Initialize ROS node
        rospy.init_node('turtlebot_supervisor', anonymous=True)
        self.params = SupervisorParams(verbose=True)

        # Home state
        self.x_home = 3
        self.y_home = 2
        self.theta_home = np.pi/2

        # Current state
        self.x = self.x_home
        self.y = self.y_home
        self.theta = self.theta_home

        # Current phase
        self.rescuing  = False
        self.mode = Mode.IDLE
        self.prev_mode = None

        # Waypoints
        # self.waypoints = [[3.5,2.6,0],[2.3,2.4,0],[2.3,0.2,0],[0.5,0.1,0],[0.3,2,0]]
        # self.waypoints = [[3.5,2.6,0],[3.5,0.2,0]]
        self.waypoints = [[3.5, 2.8, np.pi/2.], # Giraffe
                          [2.45,2.8,np.pi],
                          [2.45,2.0,-np.pi/2], 
                          [2.45,1.0,np.pi], # Elephant
                          [2.45,0.2,-np.pi/2],
                          [3.4, 0.1,-3*np.pi/4],
                          [3.4,-1.1,np.pi],
                          [2,-1.2,-np.pi/2],    # bear
                          [.4,-1.0,np.pi/4], # Exiting weird area
                          [.3,0.32,np.pi], # Zebra
                          [.3,2.0,np.pi]] # cow

        # Objects
        self.valid_objects = ['cow','zebra','giraffe','elephant', 'bear']
        self.detected_objects = {}

        # Objects selected by TAs
        self.selected_objects = {}

        # Map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0, 0]
        self.map_probs = []
        self.occupancy = None

        self.should_stop = False

        ########## PUBLISHERS ##########

        # Command navigator for controller
        self.nav_goal_publisher = rospy.Publisher('/cmd_nav', Pose2D, queue_size=10)

        # Command vel (used for idling)
        self.cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # For people stuff
        self.person_pub = rospy.Publisher('/person_pub',Pose2D,queue_size=10)

        # Markers
        # self.animal_marker_pubs = {}
        self.animal_marker_pub = rospy.Publisher('/animal_marker', Marker, queue_size=10)

        ########## SUBSCRIBERS ##########
        # Map
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)

        # Camera
        rospy.Subscriber('/detector/person', DetectedObject, self.person_callback)
        rospy.Subscriber('/collides',Bool,self.collide_cb)

        rospy.Subscriber('/detector/giraffe', DetectedObject, self.add_object_to_dict)
        rospy.Subscriber('/detector/elephant', DetectedObject, self.add_object_to_dict)
        rospy.Subscriber('/detector/cow', DetectedObject, self.add_object_to_dict)
        rospy.Subscriber('/detector/zebra', DetectedObject, self.add_object_to_dict)
        rospy.Subscriber('/detector/bear', DetectedObject, self.add_object_to_dict)


        # High-level navigation pose
        rospy.Subscriber('/nav_pose', Pose2D, self.nav_pose_callback)

        # If using gazebo, we have access to perfect state
        if self.params.use_gazebo:
            rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_callback)
        self.trans_listener = tf.TransformListener()

        # If using rviz, we can subscribe to nav goal click
        if self.params.rviz:
            rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.rviz_goal_callback)

        # Start following the waypoints
        self.set_new_waypoint()
    
    ########## SUBSCRIBER CALLBACKS ##########

    def gazebo_callback(self, msg):
        if "turtlebot3_burger" not in msg.name:
            return

        pose = msg.pose[msg.name.index("turtlebot3_burger")]
        self.x = pose.position.x
        self.y = pose.position.y
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.theta = euler[2]

    def rviz_goal_callback(self, msg):
        """ callback for a pose goal sent through rviz """
        origin_frame = "/map" if self.params.mapping else "/odom"
        print("Rviz command received!")

        try:
            nav_pose_origin = self.trans_listener.transformPose(origin_frame, msg)
            self.x_g = nav_pose_origin.pose.position.x
            self.y_g = nav_pose_origin.pose.position.y
            quaternion = (nav_pose_origin.pose.orientation.x,
                          nav_pose_origin.pose.orientation.y,
                          nav_pose_origin.pose.orientation.z,
                          nav_pose_origin.pose.orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            self.theta_g = euler[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass
        
        rospy.loginfo("Set new goal from rviz.")
        self.mode = Mode.NAV

    def nav_pose_callback(self, msg):
        self.x_g = msg.x
        self.y_g = msg.y
        self.theta_g = msg.theta
        self.mode = Mode.NAV

    def person_callback(self, msg):
        # I wanna know where the robot is when it sees the person
        loc = Pose2D()
        loc.x = self.x
        loc.y = self.y
        loc.theta = self.theta
        self.person_pub.publish(loc)

    def collide_cb(self,msg):
        if msg.data==True:
            rospy.loginfo("Potential collision detected. Stopping.")
            self.mode = Mode.STOP

    def add_object_to_dict(self, msg):
        '''Adds arbitrary object that was detecetd to our objects dictionary'''
        conf = msg.confidence
        name = msg.name
        distance = float(msg.distance)

        if conf > 0.7 and not self.rescuing:
            if name in self.detected_objects:
                old_confidence = self.detected_objects[name]["confidence"]

                if old_confidence < conf:
                    self.detected_objects[name] = {"pose": [self.x,self.y,self.theta], "confidence": conf, "distance": distance}
                    rospy.loginfo("Updated the pose of {} due to higher confidence ({})".format(name,round(conf,6)))
                    self.publish_marker(name)
            else:      
                self.detected_objects[name] = {"pose": [self.x,self.y,self.theta], "confidence": conf, "distance": distance}
                rospy.loginfo('AYO WE ADDED A: {} ({})'.format(name,round(conf,6)))
                self.publish_marker(name)
            
    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x, msg.origin.position.y)
    
    def map_callback(self, msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if (
            self.map_width > 0
            and self.map_height > 0
            and len(self.map_probs) > 0
        ):
            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                7,
                self.map_probs,
            )

    ########## STATE MACHINE ACTIONS ##########

    def publish_marker(self, name):
        '''Publishes marker at the object location based on the name given'''

        if name not in self.detected_objects or name not in self.valid_objects:
            return
        
        # if name not in self.animal_marker_pubs:
        #     self.animal_marker_pubs[name] = rospy.Publisher('marker/'+str(name), Marker, queue_size=10)
        
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()
        marker.id = self.valid_objects.index(name)

        marker.type = 2 # sphere

        marker.pose.position.x = self.detected_objects[name]['pose'][0] + self.detected_objects[name]['distance']*np.cos(self.theta)
        marker.pose.position.y = self.detected_objects[name]['pose'][1] + self.detected_objects[name]['distance']*np.sin(self.theta)
        marker.pose.position.z = 1

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.animal_marker_pub.publish(marker)

    def nav_to_pose(self):
        """ sends the current desired pose to the navigator """

        nav_g_msg = Pose2D()
        nav_g_msg.x = self.x_g
        nav_g_msg.y = self.y_g
        nav_g_msg.theta = self.theta_g

        self.nav_goal_publisher.publish(nav_g_msg)

    def stay_idle(self):
        """ sends zero velocity to stay put """

        vel_g_msg = Twist()
        self.cmd_vel_publisher.publish(vel_g_msg)
    
    def close_to(self, x, y, theta):
        """ checks if the robot is at a pose within some threshold """

        # return abs(x - self.x) < self.params.pos_eps and \
        #        abs(y - self.y) < self.params.pos_eps and \
        #        abs(wrapToPi(theta - self.theta)) < self.params.theta_eps

        return np.linalg.norm(np.array([self.x - x, self.y - y])) < self.params.pos_eps and \
               abs(wrapToPi(theta - self.theta)) < self.params.theta_eps

    def request_publisher(self):
        '''Asks for TA input'''
        print("\nHere are all the objects detected during exploration", list(self.detected_objects))
        print("Please select the objects you want to rescue one by one.")
        print("Type none when you're done.\n")


        # Call for an infinite loop that keeps executing until an exception occurs
        while True:
            # TODO: Add break when the list is empty or when we reach 3 objects selected
            obj = input("Object to rescue: ")

            if obj == "none":
                break
            elif obj in self.detected_objects:
                detected_obj = self.detected_objects.pop(obj)
                self.selected_objects[obj] = detected_obj
                print("Adding {} to the rescue operation. These animals are left: {}".format(obj, list(self.detected_objects)))
            else:
                print("Sorry the object {} was not found".format(obj))

        # TODO: would be nice to stop until the TA tells us we can go here

        # Run TSP
        if len(self.selected_objects) > 1:
            rospy.loginfo("Solving the traveling zookeeper problem to determine the best rescue operation.")
            # Get just the position
            objects_pos = []
            for val in self.selected_objects.values():
                objects_pos.append(val['pose'][0:2])
            
            # Feed into TSP
            start = [self.x_home, self.y_home]
            length, order = solve_tsp_from_map(start, objects_pos, self.occupancy, self.theta)

            # Create an ordered list we can send to the waypoints
            names = ""
            optimal_waypoints = []
            print(order)
            for i in order:
                name = list(self.selected_objects)[i-1]
                names += str(name) + ", "
                optimal_waypoints.append(self.selected_objects[name]['pose'])
            
            rospy.loginfo("We take {} units of time by going in this order: {}".format(round(length,2),names))
            
        else:
            # No need to run TSP for a single object
            optimal_waypoints = [self.selected_objects[list(self.selected_objects)[0]]['pose']]

        # Set the waypoints object
        self.rescuing = True
        self.waypoints = optimal_waypoints

    def set_new_waypoint(self):
        '''Set waypoint to next in list. 
           If no more in list, set to go home. 
           If already home ask for rescue options.'''

        # No more waypoints and far from home. Go home
        if not self.waypoints and not self.close_to(self.x_home, self.y_home, self.theta_home):
            rospy.loginfo("Robot is going home.")
            self.x_g = self.x_home
            self.y_g = self.y_home
            self.theta_g = self.theta_home
            self.mode = Mode.NAV
        
        # Out of waypoints and close to home. In the exploration phase
        elif not self.waypoints and self.close_to(self.x_home, self.y_home, self.theta_home) and not self.rescuing:
            rospy.loginfo("Robot is going to request more waypoints.")
            self.mode = Mode.IDLE
            self.stay_idle()
            self.request_publisher()
        
        # We've rescued all the animals and we're home. Just sit tight!
        elif not self.waypoints and self.rescuing and self.close_to(self.x_home, self.y_home, self.theta_home):
            self.mode = Mode.IDLE
            self.stay_idle()

        # Set new waypoint
        else:
            self.waypoint = self.waypoints.pop(0)
            self.x_g = self.waypoint[0]
            self.y_g = self.waypoint[1]
            self.theta_g = self.waypoint[2]            

            self.mode = Mode.NAV
            rospy.loginfo("Setting a new waypoint: ({}, {}, {})".format(round(self.waypoint[0],6), round(self.waypoint[1],6), round(self.waypoint[2],6)))

    ########## STATE MACHINE LOOP ##########

    def loop(self):
        """ the main loop of the robot. At each iteration, depending on its
        mode (i.e. the finite state machine's state), if takes appropriate
        actions. This function shouldn't return anything """

        if not self.occupancy:
            rospy.loginfo("Supervisor waiting on a map.")
            return

        if not self.params.use_gazebo:
            try:
                origin_frame = "/map" if self.params.mapping else "/odom"
                translation, rotation = self.trans_listener.lookupTransform(origin_frame, '/base_footprint', rospy.Time(0))
                self.x, self.y = translation[0], translation[1]
                self.theta = tf.transformations.euler_from_quaternion(rotation)[2]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass

        # logs the current mode
        if self.prev_mode != self.mode:
            rospy.loginfo("Current mode: %s", self.mode)
            rospy.loginfo("x goal: %f", self.x_g)
            rospy.loginfo("y goal: %f", self.y_g)
            self.prev_mode = self.mode

        ########## State Machine ##########

        if self.mode == Mode.IDLE:
            if self.rescuing and not self.close_to(self.x_home, self.y_home, self.theta_home):
                # Stop to rescue animals
                # The rescue flag gets set before the waypoints, so if we don't do the second check
                # the robot will try to rescue an animal when it's still home
                print()
                rospy.loginfo("Rescuing the cute animal...")
                rospy.sleep(5.)
                rospy.loginfo("Animal rescued!")
                print()
                # TODO: Fix small bug where this runs when we reach the home after we've collected all the animals
            self.set_new_waypoint()

        elif self.mode == Mode.NAV:
            if self.close_to(self.x_g, self.y_g, self.theta_g):
                self.mode = Mode.IDLE
                self.stay_idle()
                time.sleep(3) # just wait for a second so robot is stopped and we have a new map
            else:
                self.nav_to_pose()

        elif self.mode == Mode.STOP:
            # print("Avoiding collision...")
            self.stay_idle()
            nav_g_msg = Pose2D()
            nav_g_msg.x = self.x
            nav_g_msg.y = self.y
            nav_g_msg.theta = self.theta
            self.nav_goal_publisher.publish(nav_g_msg)
            time.sleep(3)
            # Follow the same waypoint as before
            rospy.loginfo("We've probably waited enough time to avoid a collision... Time to move again.")
            self.mode = Mode.NAV

        else:
            raise Exception("This mode is not supported: {}".format(str(self.mode)))

    def run(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()

if __name__ == '__main__':
    sup = Supervisor()
    sup.run()