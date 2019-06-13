import numpy as np
import tensorflow as tf
from PIL import Image
import time
import cv2 as cv
import YOLOv3_detection
import Helper_Methods


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/model.ckpt', 'Chceckpoint file')

def Run():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )


    classes =Helper_Methods.load_coco_names( 'coco.names.txt' )

    model = YOLOv3_detection.yolo_v3
    boxes, inputs =Helper_Methods.get_boxes_and_inputs ( model, len ( classes ),416,'NCHW' )

    saver = tf.train.Saver ( var_list=tf.global_variables ( scope='detector' ) )

    with tf.Session ( config=config ) as sess:
        t0 = time.time ()
        saver.restore ( sess, FLAGS.ckpt_file )
        print ( 'Model restored in {:.2f}s'.format ( time.time () - t0 ) )
        cap = cv.VideoCapture ("Input image.jpg")

        while cv.waitKey ( 10 ) < 0:


            hasFrame, frame = cap.read ()

            # Stop the program if reached end of video
            if not hasFrame:
                print ( "Done processing !!!" )
                cv.waitKey ( 3000 )
                break

            frame = cv.cvtColor (frame, cv.COLOR_BGR2RGB )

            img = Image.fromarray(frame)
            img_resized = Helper_Methods.letter_box_image ( img, 416, 416, 128 )
            img_resized = img_resized.astype ( np.float32 )

            t0 = time.time ()
            detected_boxes = sess.run (
                boxes, feed_dict={inputs: [img_resized]} )

            print(tf.shape( detected_boxes))
            #print(detected_boxes)
            print(tf.shape(detected_boxes[0]))
            filtered_boxes =Helper_Methods.non_max_suppression(detected_boxes,
                                                    confidence_threshold=0.7,
                                                    iou_threshold=0.5)

            print("Predictions found in {:.2f}s".format(time.time() - t0))
            Helper_Methods.draw_boxes ( filtered_boxes, img, classes, (416,416), True )
            open_cv_image = np.array ( img )
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy ()
            cv.imshow ( "YOLOv3", open_cv_image )



Run()

