import argparse
import cv2
from inference import Network

# INPUT_STREAM = "test_video.mp4"
# CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes
    #c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    #ct_desc = "The confidence threshold to use with the bounding boxes"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    #optional.add_argument("-c", help=c_desc, default='BLUE')
    #optional.add_argument("-ct", help=ct_desc, default=0.5)
    args = parser.parse_args()

    return args

# def draw_boxes(frame, result, args, width, height):
#     '''
#     Draw bounding boxes onto the frame.
#     '''
#     for box in result[0][0]: # Output shape is 1x1x100x7
#         conf = box[2]
#         if conf >= args.ct:
#             xmin = int(box[3] * width)
#             ymin = int(box[4] * height)
#             xmax = int(box[5] * width)
#             ymax = int(box[6] * height)
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, 1)
#     return frame


def infer_on_video(args):
    # Convert the args for color and confidence
    #args.c = convert_color(args.c)
    #args.ct = float(args.ct)

    ### TODO: Initialize the Inference Engine
    plugin = Network()

    ### TODO: Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Perform inference on the frame
        plugin.async_inference(p_frame)

        ### TODO: Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            ### TODO: Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result, args, width, height)
            # Write out the frame
            out.write(frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()
------------------------------------------------------------------------------

# Video Capture RTSP or mp4 file
INPUT_STREAM = "/content/IMG_7483.mp4"
video_capture = cv2.VideoCapture(INPUT_STREAM) 
FILE_OUTPUT = "/content/output.mp4"
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
out = cv2.VideoWriter(FILE_OUTPUT,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30,(frame_width, frame_height))

cap = cv2.VideoCapture(INPUT_STREAM)

def infer_on_video(args):

    ### TODO: Initialize the Inference Engine
    plugin = Network()

    ### TODO: Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            counter = 0
            while (True):
                ret, image_np = cap.read()
                counter += 1
                if ret:
                  h = image_np.shape[0]
                  w = image_np.shape[1]

                  if counter % 1 == 0:
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    
                    # Draw bounding box while the score larger than 0.7 and perform count
                    final_score = np.squeeze(scores)    
                    count_box = 0
                    for i in range(100):
                        if scores is None or final_score[i] > 0.7:
                            count_box = count_box + 1
                            
                    vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                        np.squeeze(boxes),
                                                                        np.squeeze(classes).astype(np.int32),
                                                                        np.squeeze(scores),
                                                                        category_index,
                                                                        use_normalized_coordinates=True,
                                                                        line_thickness=0,
                                                                        min_score_thresh=0.7)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image_np,'Detected Count  = %d'%(count_box),(50,50), font, 1,(200,255,155),2,cv2.LINE_AA)
                    
                    out.write(image_np)
                    print(count_box)
                else:
                  break
    out.release()
    cap.release()
    cv2.destroyAllWindows()

def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
