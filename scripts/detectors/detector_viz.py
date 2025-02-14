#!/usr/bin/env python3

import rospy
import os
# watch out on the order for the next two imports lol
from tf import TransformListener
import numpy as np
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, LaserScan
from aa274_final_project.msg import DetectedObject, DetectedObjectList
from cv_bridge import CvBridge, CvBridgeError
import cv2
import math

CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_TIMEOUT = 1.0

class DetectorViz:

    def __init__(self):
        rospy.init_node('turtlebot_detector', anonymous=True)
        self.bridge = CvBridge()

        self.tf_listener = TransformListener()
        self.last_box_time = rospy.get_rostime()
        rospy.Subscriber('/detector/objects', DetectedObjectList, self.detected_objects_name_callback, queue_size=10)
        self.detected_objects = None
        rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self.compressed_camera_callback, queue_size=1, buff_size=2**24)
        self.viz_pub = rospy.Publisher("/detector/image/compressed", CompressedImage, queue_size=1)

    def load_image_into_numpy_array(self, img):
        """ converts opencv image into a numpy array """

        (im_height, im_width, im_chan) = img.shape
        return np.array(img.data).reshape((im_height, im_width, 3)).astype(np.uint8)

    def detected_objects_name_callback(self, msg):
        rospy.loginfo("There are %i detected objects" % len(msg.objects))
        self.detected_objects = msg
        self.last_box_time = rospy.get_rostime()

    def camera_callback(self, msg):
        """ callback for camera images """

        try:
            img = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            img_bgr8 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        if (msg.header.stamp.to_sec() - self.last_box_time.to_sec()) > BOX_TIMEOUT:
            self.detected_objects = None

        if np.abs(rospy.get_rostime().to_sec() - self.last_box_time.to_sec()) > BOX_TIMEOUT:
            self.detected_objects = None

        self.camera_common(img, img_bgr8)

    def compressed_camera_callback(self, msg):
        """ callback for camera images """

        try:
            img = self.bridge.compressed_imgmsg_to_cv2(msg, "passthrough")
            img_bgr8 = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        if np.abs(rospy.get_rostime().to_sec() - self.last_box_time.to_sec()) > BOX_TIMEOUT:
            self.detected_objects = None
        self.camera_common(img, img_bgr8)
        

    def camera_common(self, img, img_bgr8):
        (img_h,img_w,img_c) = img.shape
        draw_color = (0,255,0)
        if self.detected_objects is not None:
            for ob_msg in self.detected_objects.ob_msgs:
                ymin, xmin, ymax, xmax = [int(x) for x in ob_msg.corners]
                cv2.rectangle(img_bgr8, (xmin,ymin), (xmax,ymax), draw_color, 2)
                # cool add-on by student in 2018 class
                cv2.putText(img_bgr8, ob_msg.name + ":" + str(round(ob_msg.confidence, 2)), (xmin, ymin+13), CV2_FONT, .5, draw_color)
        self.viz_pub.publish(self.bridge.cv2_to_compressed_imgmsg(img_bgr8))


    def run(self):
        rospy.spin()

if __name__=='__main__':
    d = DetectorViz()
    d.run()
