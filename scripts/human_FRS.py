#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import Point32, Polygon, PolygonStamped
from nav_msgs.msg import Path
from aa274_final_project.msg import FRS

class human_FRS:
    """
    This node handles calculating the human's forward reachable set (FRS).
    It takes in planned trajectory and current location.
    It publishes the FRS, represented by a 2D zonotope (aka a square) at each point in the planned trajectory.
    """

    def __init__(self):
        rospy.init_node("human_FRS",anonymous=True)
        self.v_max = 0.2
        self.human_x = []
        self.human_y = []
        self.predict_v = np.array([0,0])
        self.error = .2
        self.ers = np.zeros((4,2))
        self.humanfrs = FRS()
        self.humanfrs.header.frame_id = "map"
        self.FRS_pub = rospy.Publisher("/ors",FRS,queue_size=10)
        self.FRS_view_pub1 = rospy.Publisher("/ors_view1",PolygonStamped,queue_size=10)
        self.FRS_view_pub2 = rospy.Publisher("/ors_view2",PolygonStamped,queue_size=10)
        self.FRS_view_pub3 = rospy.Publisher("/ors_view3",PolygonStamped,queue_size=10)
        #self.ERS_pub = rospy.Publisher("/ers",PolygonStamped,queue_size=10)
        #rospy.Subscriber("/????",Pose,self.detection_callback) add this later
    
    def detection_callback(self, loc):
        self.human_x.append(loc.x)
        self.human_y.append(loc.y)
        current_v = np.array([self.human_x[-1]-self.human_x[-2],self.human_y[-1]-self.human_y[-2]])
        self.predict_v = (self.predict_v + current_v) /2

    def calc_ERS(self):
        # center is 0,0
        # betas are just error
        # use 2 generators to get a square
        gen = np.array([[1,0],[0,1]])
        beta_gen = np.vstack((gen*self.error,gen*-self.error))
        for i in range(4):
            self.ers[i,:] = beta_gen[i//2*2]+beta_gen[(i+1)//2%2*2+1]
        #data = PolygonStamped()
        #data.header.frame_id = "map"
        #for pt in self.ers:
        #    data.polygon.points.append(Point32(pt[0],pt[1],0))
        #self.ERS_pub.publish(data)

    def calc_PRS(self,t):
        PRS = np.zeros((4,2))
        gen = np.array([[1,0],[0,1]])
        # [1,1]
        PRS[0,:] = gen[0]+gen[1]
        # [1,-1]
        # [-1,-1]
        # [-1,1]
        return PRS
    
    def calc_FRS(self):
        self.calc_ERS()
        # smoothed traj dt = .1 sec
        centers = np.hstack((np.linspace(0,self.predict_v[0]*3,num=10,endpoint=False),
                             np.linspace(0,self.predict_v[1]*3,num=10,endpoint=False)))
        self.humanfrs.polygons = []
        for t in range(len(centers)):
            PRS_i = centers[t]+self.calc_PRS(t)
            frs_i = PRS_i+self.ers
            data = PolygonStamped()
            data.header.frame_id = "map"
            for pt in frs_i:
                data.polygon.points.append(Point32(pt[0],pt[1],0))
            self.humanfrs.polygons.append(data)
        self.FRS_pub.publish(self.humanfrs)

    def view_FRS(self):
        self.FRS_view_pub1.publish(self.humanfrs.polygons[0])
        self.FRS_view_pub2.publish(self.humanfrs.polygons[len(self.humanfrs.polygons)//2])
        self.FRS_view_pub3.publish(self.humanfrs.polygons[-1])

    def run(self,vis=True):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.calc_FRS()
            if vis:
                self.view_FRS()
            rate.sleep()

if __name__ == '__main__':
    my_frs = robot_FRS()
    my_frs.run()

