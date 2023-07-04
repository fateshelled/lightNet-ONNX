# lightNet-ONNX
ONNX and TensorRT demo for [lightNet-TRT](https://github.com/daniel89710/lightNet-TRT)

![demo](https://github.com/fateshelled/lightNet-ONNX/assets/53618876/b929891f-4be3-4e66-8933-4791e9e56bc3)

demo video is [MOT16-14](https://motchallenge.net/vis/MOT16-14)

## Requirements
### Convert to ONNX model
- Pytorch
- NumPy
- onnx
- onnxsim

### ONNX demo
- onnxruntime
- OpenCV
- NumPy

### TensorRT demo
- tensorrt
- pycuda
- OpenCV
- NumPy

## Usage
### Convert weights to ONNX model
- Download cfg file and weights file from [lightNet-TRT](https://github.com/daniel89710/lightNet-TRT)
- convert weights to onnx model by following script.
```bash
python convert2onnx.py lightNet-BDD100K-det-semaseg-1280x960.cfg \
                       lightNet-BDD100K-det-semaseg-1280x960.weights \
                       lightNet-BDD100K-det-semaseg-1280x960.onnx 
```

### Convert ONNX model to TensorRT engine
```bash
source convert2trt.bash lightNet-BDD100K-det-semaseg-1280x960.onnx \
                        lightNet-BDD100K-det-semaseg-1280x960.engine
```

### Run ONNX Demo
```bash
python demo_onnx.py --onnx_model_path lightNet-BDD100K-det-semaseg-1280x960.onnx \
                    --video_path {video or webcam path}
```

### Run TensorRT Demo
```bash
python demo_trt.py --engine_path lightNet-BDD100K-det-semaseg-1280x960.engine \
                    --video_path {video or webcam path}
```

## Reference
- https://github.com/daniel89710/lightNet-TRT
- https://github.com/Tianxiaomo/pytorch-YOLOv4
