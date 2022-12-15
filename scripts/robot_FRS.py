#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import Point32, Polygon, PolygonStamped
from nav_msgs.msg import Path
from aa274_final_project.msg import FRS
from std_msgs.msg import Bool

class robot_FRS:
    """
    This node handles calculating the robot's forward reachable set (FRS).
    It takes in planned trajectory and current location.
    It publishes the FRS, represented by a 2D zonotope (aka a square) at each point in the planned trajectory.
    """

    def __init__(self):
        rospy.init_node("robot_FRS",anonymous=True)
        self.start = rospy.get_rostime().to_sec()
        self.v_max = .3
        self.error = .1
        self.traj = np.zeros(2)
        self.ers = np.zeros((4,2))
        self.robotfrs = FRS()
        self.robotfrs.header.frame_id = "map"
        self.FRS_pub = rospy.Publisher("/frs",FRS,queue_size=10)
        self.FRS_view_pub1 = rospy.Publisher("/frs_view1",PolygonStamped,queue_size=10)
        self.FRS_view_pub2 = rospy.Publisher("/frs_view2",PolygonStamped,queue_size=10)
        self.FRS_view_pub3 = rospy.Publisher("/frs_view3",PolygonStamped,queue_size=10)
        # self.ERS_pub = rospy.Publisher("/ers",PolygonStamped,queue_size=10)
        self.collision_pub = rospy.Publisher("/collides",Bool,queue_size=10)
        rospy.Subscriber("/cmd_smoothed_path",Path,self.traj_callback)
        rospy.Subscriber("/ors",FRS,self.ors_callback)
    
    def traj_callback(self, smooth_traj):
        self.traj = np.zeros((len(smooth_traj.poses),2))
        for i in range(len(smooth_traj.poses)):
            self.traj[i,0] = smooth_traj.poses[i].pose.position.x
            self.traj[i,1] = smooth_traj.poses[i].pose.position.y
        self.calc_FRS()
        self.start = rospy.get_rostime().to_sec()

    def ors_callback(self, ORS):
        # align times between reachable sets
        idx = int((rospy.get_rostime().to_sec()-self.start)*10+1)
        poly = ORS.polygons
        collision = False
        for t in range(len(poly)):
            if idx+t > len(self.robotfrs.polygons)-1:
                break
            frs_i = self.robotfrs.polygons[idx+t].polygon.points
            x_low = poly[t].polygon.points[2].x
            x_high = poly[t].polygon.points[0].x
            y_low = poly[t].polygon.points[2].y
            y_high = poly[t].polygon.points[0].y
            for pt in frs_i:
                if pt.x > x_low and pt.x < x_high and pt.y > y_low and pt.y < y_high:
                    collision = True
        self.collision_pub.publish(collision)

    def calc_ERS(self):
        # center is 0,0
        # betas are just error
        # use 2 generators to get a square
        gen = np.array([[1,0],[0,1]])
        beta_gen = np.vstack((gen*self.error,gen*-self.error))
        for i in range(4):
            self.ers[i,:] = beta_gen[i//2*2]+beta_gen[(i+1)//2%2*2+1]
        # data = PolygonStamped()
        # data.header.frame_id = "map"
        # for pt in self.ers:
        #     data.polygon.points.append(Point32(pt[0],pt[1],0))
        # self.ERS_pub.publish(data)
    
    def calc_FRS(self):
        self.calc_ERS()
        centers = np.array(self.traj)
        self.robotfrs.polygons = []
        for c in centers:
            frs_i = c+self.ers
            data = PolygonStamped()
            data.header.frame_id = "map"
            for pt in frs_i:
                data.polygon.points.append(Point32(pt[0],pt[1],0))
            self.robotfrs.polygons.append(data)
        self.FRS_pub.publish(self.robotfrs)

    def view_FRS(self):
        self.FRS_view_pub1.publish(self.robotfrs.polygons[0])
        self.FRS_view_pub2.publish(self.robotfrs.polygons[len(self.robotfrs.polygons)//2])
        self.FRS_view_pub3.publish(self.robotfrs.polygons[-1])

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            #self.calc_FRS()
            if len(self.robotfrs.polygons) > 0:
                self.view_FRS()
            rate.sleep()

if __name__ == '__main__':
    my_frs = robot_FRS()
    my_frs.run()

