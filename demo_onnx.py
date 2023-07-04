import os
import time
import onnxruntime as ort
import numpy as np
import cv2
from tool.utils import post_processing, plot_boxes_cv2
from tool.segment_utils import colorize, overlay


CLASS_NAMES = [
    "car", "bus", "person",
    "bike", "truck", "motor",
    "train", "rider",
    "traffic_sign", "traffic_light",
]


def main(onnx_model_path: str, video_path: str, conf_thresh: float, nms_thresh: float):
    if not os.path.exists(onnx_model_path):
        raise ValueError(f"{onnx_model_path} not exist.")

    if not os.path.exists(video_path):
        raise ValueError(f"{video_path} not exist.")

    cap = cv2.VideoCapture(video_path)

    available_providers = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session_option = ort.SessionOptions()
    session_option.log_severity_level = 4
    session_option.intra_op_num_threads = 0

    sess = ort.InferenceSession(
        onnx_model_path,
        sess_options=session_option,
        providers=providers)

    input = sess.get_inputs()[0]
    input_name = input.name
    input_height = input.shape[2]
    input_width = input.shape[3]
    output_names = [output.name for output in sess.get_outputs()]
    segmentation = len(output_names) > 2

    output_shapes = [output.shape for output in sess.get_outputs()]
    output_types = [output.type for output in sess.get_outputs()]
    print("====Model Info====")
    print(f"Model Path: {onnx_model_path}")
    print(f"Inputs:")
    print(f" - {input.name}, {input.shape}, {input.type}")
    print(f"Outputs:")
    for name, shape, t in zip(output_names, output_shapes, output_types):
        print(f" - {name}, {shape}, {t}")
    print(f"Segmentation available: {segmentation}")
    print()

    while cap.isOpened():
        res, img = cap.read()
        if res is False:
            break

        # Pre-processing
        t_pre = time.perf_counter()
        # resize and normalize
        blob = cv2.resize(img, (input_width, input_height)).astype(np.float32) / 255.0
        # HWC -> NCHW
        blob = blob.transpose(2, 0, 1)[None, :, :, :]
        dt_pre = time.perf_counter() - t_pre

        # inference
        t_inf = time.perf_counter()
        outputs = sess.run(output_names, {input_name: blob})
        dt_inf = time.perf_counter() - t_inf

        # Post-processing
        t_post = time.perf_counter()
        boxes = post_processing(img, conf_thresh, nms_thresh, outputs[0], outputs[1])
        dt_post = time.perf_counter() - t_post

        # Visualization
        t_vis = time.perf_counter()
        if segmentation:
            seg_img = outputs[2][0]
            colored_seg = colorize(seg_img)
            overlayed = overlay(img, colored_seg)
            drawn = plot_boxes_cv2(overlayed, boxes[0], class_names=CLASS_NAMES)
        else:
            drawn = plot_boxes_cv2(img, boxes[0], class_names=CLASS_NAMES)
        dt_vis = time.perf_counter() - t_vis

        print(f"preprocessing:  {dt_pre:.3f}s")
        print(f"inference:      {dt_inf:.3f}s")
        print(f"postprocessing: {dt_post:.3f}s")
        print(f"visualization: {dt_vis:.3f}s")
        print("")

        cv2.imshow("lightNet-ONNX", drawn)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--onnx_model_path',
                        type=str,
                        # default="lightNet-BDD100K-1280x960.onnx",
                        # default="lightNet-BDD100K-1280x960-chPruning.onnx",
                        default="lightNet-BDD100K-det-semaseg-1280x960.onnx",
                        # default="lightNet-BDD100K-chPruning-det-semaseg-1280x960.onnx",
                        help="Onnx Model file path.")
    parser.add_argument('-v', '--video_path',
                        type=str,
                        default="/dev/video0",
                        # default="MOT16-14-raw.webm",
                        help="input video path. Default /dev/video0")
    parser.add_argument('-c', '--conf_thresh',
                        type=float,
                        default=0.45,
                        help="confidence threshold. default 0.45")
    parser.add_argument('-n', '--nms_thresh',
                        type=float,
                        default=0.30,
                        help="nms threshold. default 0.30")
    args = parser.parse_args()

    main(
        args.onnx_model_path,
        args.video_path,
        args.conf_thresh,
        args.nms_thresh
    )
