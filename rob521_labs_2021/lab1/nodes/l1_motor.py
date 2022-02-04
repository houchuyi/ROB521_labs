#!/usr/bin/env python
import rospy
import math
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import String



def publisher_node():
    print('TODO: initialize the publisher node here, \
            and publish wheel command to the cmd_vel topic')

def main():
    try:
        rospy.init_node('motor')
        publisher_node()
    except rospy.ROSInterruptException:
        pass
    

if __name__ == '__main__':
    main()
