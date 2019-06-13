import tensorflow as tf

import time

import YOLOv3_detection
import Helper_Methods

import os
import shutil

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/model.ckpt', 'Chceckpoint file')

tf.app.flags.DEFINE_string('output_dir', './model_exported',
                           """Directory where to export the model.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")

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

        # (re-)create export directory
        export_path = os.path.join (
            tf.compat.as_bytes ( FLAGS.output_dir ),
            tf.compat.as_bytes ( str ( FLAGS.model_version ) ) )
        if os.path.exists ( export_path ):
            shutil.rmtree ( export_path )

        # create model builder
        builder = tf.saved_model.builder.SavedModelBuilder ( export_path )

        # create tensors info
        predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info ( inputs )
        predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info (boxes)

        # build prediction signature
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def (
                inputs={'images': predict_tensor_inputs_info},
                outputs={'scores': predict_tensor_scores_info},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        # save the model
        legacy_init_op = tf.group ( tf.tables_initializer (), name='legacy_init_op' )
        builder.add_meta_graph_and_variables (
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images': prediction_signature
            },
            legacy_init_op=legacy_init_op )

        builder.save ()

    print("Successfully exported GAN model version '{}' into '{}'".format(
        FLAGS.model_version, FLAGS.output_dir))

Run()

