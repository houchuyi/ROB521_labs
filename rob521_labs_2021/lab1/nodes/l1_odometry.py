#!/usr/bin/env python
import rospy
import math
from nav_msgs.msg import Odometry



def get_yaw_from_quarternion(q):
	print('TODO: complete the function that gets yaw from quaternion')


def callback(odom_data):
	print('TODO: complete the call back function for subscriber')


def main():
	try:
		rospy.init_node('odometery')
		print('TODO: initialize the subscriber of odometery here')
		
		rospy.spin()
	except rospy.ROSInterruptException:
		pass
	

if __name__ == '__main__':
	main()
