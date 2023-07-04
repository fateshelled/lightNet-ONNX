import time
import os
import cv2
import numpy as np
import tensorrt as trt
import trt_common
from tool.utils import post_processing, plot_boxes_cv2
from tool.segment_utils import colorize, overlay


CLASS_NAMES = [
    "car", "bus", "person",
    "bike", "truck", "motor",
    "train", "rider",
    "traffic_sign", "traffic_light",
]


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def get_engine(engine_file_path):
    print(f"\033[32mReading engine from file {engine_file_path}\033[0m")
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def main(engine_path: str,
         input_height: int, input_width: int,
         video_path: str,
         conf_thresh: float, nms_thresh: float
         ):

    if not os.path.exists(video_path):
        raise ValueError(f"{video_path} not exist.")

    cap = cv2.VideoCapture(video_path)

    engine = get_engine(engine_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream, output_names = trt_common.allocate_buffers(engine)
    boxes_index = output_names.index("boxes")
    confs_index = output_names.index("confs")
    segmentation = "seg" in output_names
    if segmentation:
        seg_index = output_names.index("seg")

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
        inputs[0].host = np.ascontiguousarray(blob)
        results = trt_common.do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        dt_inf = time.perf_counter() - t_inf

        # Post-processing
        t_post = time.perf_counter()
        boxes = results[boxes_index].reshape(1, -1, 1, 4)
        confs = results[confs_index].reshape(1, -1, len(CLASS_NAMES))
        boxes = post_processing(img, conf_thresh, nms_thresh, boxes, confs)
        dt_post = time.perf_counter() - t_post

        # Visualization
        t_vis = time.perf_counter()
        if segmentation:
            seg_img = results[seg_index].reshape(int(input_height*0.5), int(input_width*0.5)).astype(np.uint8)
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

        cv2.imshow("lightNet-ONNX-trt", drawn)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-e",
        "--engine_path",
        type=str,
        # default="lightNet-BDD100K-1280x960.engine",
        # default="lightNet-BDD100K-1280x960-chPruning.engine",
        default="lightNet-BDD100K-det-semaseg-1280x960.engine",
        # default="lightNet-BDD100K-chPruning-det-semaseg-1280x960.engine",

        help="TensorRT engine file path.")
    parser.add_argument(
        "-ih",
        "--input_height",
        type=int,
        default=960,
        help="Model input height.")
    parser.add_argument(
        "-iw",
        "--input_width",
        type=int,
        default=1280,
        help="Model input width.")
    parser.add_argument(
        "-v",
        "--video_path",
        type=str,
        default="/dev/video0",
        # default="./MOT16-14-raw.webm",
        help="input video path.")
    parser.add_argument('--conf_thresh',
                        type=float,
                        default=0.45,
                        help="confidence threshold. default 0.45")
    parser.add_argument('--nms_thresh',
                        type=float,
                        default=0.30,
                        help="nms threshold. default 0.30")
    args = parser.parse_args()

    main(
        args.engine_path,
        args.input_height, args.input_width,
        args.video_path,
        args.conf_thresh, args.nms_thresh,
    )
