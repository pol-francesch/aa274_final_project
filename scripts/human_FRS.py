#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import Point32, Polygon, PolygonStamped, Pose2D
from nav_msgs.msg import Path
from aa274_final_project.msg import FRS, DetectedObject
from utils import wrapToPi

class human_FRS:
    """
    This node handles calculating the human's forward reachable set (FRS).
    It takes in planned trajectory and current location.
    It publishes the FRS, represented by a 2D zonotope (aka a square) at each point in the planned trajectory.
    """

    def __init__(self):
        rospy.init_node("human_FRS",anonymous=True)
        self.acc = .1
        # Initialize human belief at origin and stationary
        self.human_x = []
        self.human_y = []
        self.times = [rospy.get_rostime().to_sec()]
        self.predict_v = np.array([0,0])
        self.error = .25
        self.ers = np.zeros((4,2))
        self.humanfrs = FRS()
        self.humanfrs.header.frame_id = "map"
        self.FRS_pub = rospy.Publisher("/ors",FRS,queue_size=10)
        self.FRS_view_pub1 = rospy.Publisher("/ors_view1",PolygonStamped,queue_size=10)
        self.FRS_view_pub2 = rospy.Publisher("/ors_view2",PolygonStamped,queue_size=10)
        self.FRS_view_pub3 = rospy.Publisher("/ors_view3",PolygonStamped,queue_size=10)
        #self.ERS_pub = rospy.Publisher("/ers",PolygonStamped,queue_size=10)
        rospy.Subscriber("/detector/person",DetectedObject,self.detection_callback)
        self.latest = DetectedObject()
        rospy.Subscriber("/person_pub",Pose2D,self.person_pub_callback)
    
    def detection_callback(self, data):
        self.latest = data

    def person_pub_callback(self,loc):
        if self.latest.confidence > .7:
            self.times.append(rospy.get_rostime().to_sec())
            angle = (wrapToPi(self.latest.thetaleft)+wrapToPi(self.latest.thetaright))/2 + loc.theta
            x = self.latest.distance*np.cos(angle) + loc.x
            y = self.latest.distance*np.sin(angle) + loc.y
            self.human_x.append(x)
            self.human_y.append(y)
            if len(self.human_x)==1: # If not initialized, do that with a better first guess than 0,0
                self.human_x.append(x)
                self.human_y.append(y)
            current_v = np.array([self.human_x[-1]-self.human_x[-2],self.human_y[-1]-self.human_y[-2]])/(self.times[-1]-self.times[-2])
            # I average velocities to make this a little bit more reliable
            self.predict_v = (self.predict_v + current_v) /2
            if np.sum(self.predict_v**2)>.1:
                self.calc_FRS()

    def calc_ERS(self):
        # center is 0,0
        # betas are just error
        # use 2 generators to get a square
        gen = np.array([[1,0],[0,1]])
        beta_gen = np.vstack((gen*self.error,gen*-self.error))
        for i in range(4):
            self.ers[i,:] = beta_gen[i//2*2]+beta_gen[(i+1)//2%2*2+1]

    def calc_PRS(self):
        PRS = np.zeros((11,4,2))
        gen = np.array([[1,0],[0,1]])
        beta_gen = np.vstack((gen,gen*-1))
        avg_x = (self.human_x[-1]+self.human_x[-2])/2
        avg_y = (self.human_y[-1]+self.human_y[-2])/2
        curr_pose = np.array([avg_x,avg_y])
        #np.array([self.human_x[-1],self.human_y[-1]])
        for j in range(len(PRS)):
            t = j*.1
            curr_acc = self.acc*t*beta_gen
            for i in range(4):
                PRS[j,i,:] = (self.predict_v + curr_acc[i//2*2] + curr_acc[(i+1)//2%2*2+1])*t + curr_pose
        return PRS
    
    def calc_FRS(self):
        self.calc_ERS()
        prs = self.calc_PRS()
        # smoothed traj dt = .1 sec
        # find the predicted straight line trajectory of where the human is going
        centers = np.hstack((np.linspace(self.human_x[-1],self.human_x[-1]+self.predict_v[0],num=11),
                             np.linspace(self.human_y[-1],self.human_y[-1]+self.predict_v[1],num=11)))
        self.humanfrs.polygons = []
        for t in range(len(prs)):
            frs_i = prs[t]+self.ers # This should work out to the Minkowski sum
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

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if len(self.humanfrs.polygons) > 0:
                self.view_FRS()
            if rospy.get_rostime().to_sec()-self.times[-1] > 2:
                self.human_x = [-10,-10]
                self.human_y = [0,0]
                self.times = [rospy.get_rostime().to_sec()]
                self.predict_v = np.array([0,0])
                self.error = 0
                self.acc = 0
                self.calc_FRS()
                self.human_x = []
                self.human_y = []
                self.error = 1
                self.acc = .1
            rate.sleep()

if __name__ == '__main__':
    my_frs = human_FRS()
    my_frs.run()

