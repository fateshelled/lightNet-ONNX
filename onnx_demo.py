import onnxruntime as ort
import numpy as np
import cv2
import time
from tool.utils import post_processing, plot_boxes_cv2
from tool.segment_utils import colorize, overlay

CLASS_NAMES = [
    "car", "bus", "person",
    "bike", "truck", "motor",
    "train", "rider",
    "traffic_sign", "traffic_light",
]


def main(model_path: str, video_path: str, conf_thresh: float, nms_thresh: float):
    cap = cv2.VideoCapture(video_path)
    res, img = cap.read()

    available_providers = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    # providers.append("CPUExecutionProvider")

    session_option = ort.SessionOptions()
    session_option.log_severity_level = 4
    session_option.intra_op_num_threads = 0

    sess = ort.InferenceSession(
        model_path,
        sess_options=session_option,
        providers=providers)

    input = sess.get_inputs()[0]
    input_name = input.name
    input_height = input.shape[2]
    input_width = input.shape[3]
    output_names = [output.name for output in sess.get_outputs()]

    while cap.isOpened():
        res, img = cap.read()
        if res is False:
            break
        t_pre = time.perf_counter()
        # resize and normalize
        blob = cv2.resize(img, (input_width, input_height)
                        ).astype(np.float32) / 255.0
        # HWC -> NCHW
        blob = blob.transpose(2, 0, 1)[None, :, :, :]
        dt_pre = time.perf_counter() - t_pre

        # inference
        t_inf = time.perf_counter()
        outputs = sess.run(output_names, {input_name: blob})
        dt_inf = time.perf_counter() - t_inf

        # Post-process
        t_post = time.perf_counter()
        # boxes = [[x,y,x,y,score,id]]
        boxes = post_processing(img, conf_thresh, nms_thresh, outputs[:2])
        dt_post = time.perf_counter() - t_post

        # Visualization
        t_vis = time.perf_counter()
        # segm = outputs[2][0]  # 1x20x480x640 -> 20x480x640
        # seg_img = np.argmax(segm, axis=0)  # 20x480x640 -> 480x640

        seg_img = outputs[2][0]  # 480x640
        colored_seg = colorize(seg_img)
        overlayed = overlay(img, colored_seg)

        drawn = plot_boxes_cv2(img, boxes[0], class_names=CLASS_NAMES)
        dt_vis = time.perf_counter() - t_vis

        print(f"preprocessing:  {dt_pre:.3f}s")
        print(f"inference:      {dt_inf:.3f}s")
        print(f"postprocessing: {dt_post:.3f}s")
        print(f"visualization: {dt_vis:.3f}s")
        print("")

        cv2.imshow("seg", overlayed)
        cv2.imshow("bbox", drawn)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break


if __name__ == "__main__":
    model_path = "../lightNet-BDD100K-det-semaseg-1280x960.onnx"
    # model_path = "../lightNet-BDD100K-chPruning-det-semaseg-1280x960.onnx"
    video_path = "/dev/video0"
    # video_path = "MOT16-14-raw.webm"
    conf_thresh = 0.45
    nms_thresh = 0.3

    main(model_path, video_path, conf_thresh, nms_thresh)
