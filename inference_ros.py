import tensorflow as tf
import numpy as np
import os
from scipy import misc
import argparse
import sys
import cv2
from utils.ros_utils import image_ros_to_cv2, compressed_ros_to_cv2, cv2_to_compressed_ros, cv2_to_ros
import rospy
from sensor_msgs.msg import Image, CompressedImage


g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])
output_folder = "./test_output"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options))
saver = tf.train.import_meta_graph('./meta_graph/my-model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./salience_model'))
image_batch = tf.get_collection('image_batch')[0]
pred_mattes = tf.get_collection('mask')[0]

rospy.init_node("deep_saliency")
image_pub_lab = rospy.Publisher("deep_salient", Image, queue_size=1)



def rgba2rgb(img):
    return img[:,:,:3]*np.expand_dims(img[:,:,3],2)

def callback (img):
    rgb = image_ros_to_cv2(img)
    rgb = cv2.resize(rgb, (0,0), fx=0.3, fy=0.3) 

    if rgb.shape[2]==4:
        rgb = rgba2rgb(rgb)
    origin_shape = rgb.shape[:2]
    rgb = np.expand_dims(misc.imresize(rgb.astype(np.uint8),[320,320,3],interp="nearest").astype(np.float32)-g_mean,0)

    feed_dict = {image_batch:rgb}
    pred_alpha = sess.run(pred_mattes,feed_dict = feed_dict)
    final_alpha = misc.imresize(np.squeeze(pred_alpha),origin_shape)

    # TODO: repub here if this function works
    image_pub_lab.publish(cv2_to_ros(final_alpha))



def main(args):
    rospy.Subscriber('output', Image, callback, queue_size=1)

    rospy.spin()



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--rgb', type=str,
        help='input rgb',default = None)
    parser.add_argument('--rgb_folder', type=str,
        help='input rgb',default = None)
    parser.add_argument('--gpu_fraction', type=float,
        help='how much gpu is needed, usually 4G is enough',default = 0.8)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
