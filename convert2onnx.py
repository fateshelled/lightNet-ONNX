from tool.darknet2onnx import transform_to_onnx
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('config')
parser.add_argument('weightfile')
parser.add_argument('--batch_size', type=int, help="Static Batchsize of the model. use batch_size<=0 for dynamic batch size")
parser.add_argument('--onnx_file_path', help="Output onnx file path")
parser.add_argument('--opset_version', default=13, type=int, help="opset version of the output onnx")
args = parser.parse_args()
transform_to_onnx(args.config, args.weightfile, args.batch_size, args.onnx_file_path, args.opset_version)

# transform_to_onnx("lightNet-BDD100K-det-semaseg-1280x960.cfg",
#                   "lightNet-BDD100K-det-semaseg-1280x960.weights",
#                   onnx_file_name="lightNet-BDD100K-det-semaseg-1280x960.onnx",
#                   batch_size=1,
#                   opset_version=13)
