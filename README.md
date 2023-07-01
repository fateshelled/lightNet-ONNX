# lightNet-PyTorch

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
