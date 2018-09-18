import tensorflow as tf
import numpy as np
import os
from scipy import misc
import argparse
import sys
import cv2

g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])
output_folder = "./test_output"

def rgba2rgb(img):
    return img[:,:,:3]*np.expand_dims(img[:,:,3],2)

def main(args):    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = args.gpu_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
        saver = tf.train.import_meta_graph('./meta_graph/my-model.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./salience_model'))
        image_batch = tf.get_collection('image_batch')[0]
        pred_mattes = tf.get_collection('mask')[0]

        while(True):
            cap = cv2.VideoCapture(0)
            ret, rgb = cap.read()
            rgb = cv2.resize(rgb, (0,0), fx=0.3, fy=0.3) 

            if rgb.shape[2]==4:
                rgb = rgba2rgb(rgb)
            origin_shape = rgb.shape[:2]
            rgb = np.expand_dims(misc.imresize(rgb.astype(np.uint8),[320,320,3],interp="nearest").astype(np.float32)-g_mean,0)

            feed_dict = {image_batch:rgb}
            pred_alpha = sess.run(pred_mattes,feed_dict = feed_dict)
            final_alpha = misc.imresize(np.squeeze(pred_alpha),origin_shape)

            # TODO: Find out why its so slow

            cv2.imshow('Salient view',final_alpha)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--rgb', type=str,
        help='input rgb',default = None)
    parser.add_argument('--rgb_folder', type=str,
        help='input rgb',default = None)
    parser.add_argument('--gpu_fraction', type=float,
        help='how much gpu is needed, usually 4G is enough',default = 0.5)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
