# lightNet-ONNX
![demo](https://github.com/fateshelled/lightNet-ONNX/assets/53618876/b929891f-4be3-4e66-8933-4791e9e56bc3)

demo video is [MOT16-14](https://motchallenge.net/vis/MOT16-14)

## Usage
### Convert weights to onnx
- Download cfg file and weights file from [lightNet-TRT](https://github.com/daniel89710/lightNet-TRT)
- convert weights to onnx model by following script.
```bash
python convert2onnx.py lightNet-BDD100K-det-semaseg-1280x960.cfg \
                       lightNet-BDD100K-det-semaseg-1280x960.weights \
                       lightNet-BDD100K-det-semaseg-1280x960.onnx 
```

### Run
```bash
python onnx_demo.py --onnx_model_path lightNet-BDD100K-det-semaseg-1280x960.onnx \
                    --video_path {video or webcam path}
```

## Reference
- https://github.com/daniel89710/lightNet-TRT
- https://github.com/Tianxiaomo/pytorch-YOLOv4
