from tool.darknet2onnx import transform_to_onnx
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('config', help="Path to the .cfg file of lightNet")
parser.add_argument('weightfile', help="Path to the weights of lightNet")
parser.add_argument('onnx_file_path', help="Output onnx file path")
parser.add_argument('--batch_size', type=int, default=1, help="Static Batchsize of the model. use batch_size<=0 for dynamic batch size. default 1.")
parser.add_argument('--opset_version', default=13, type=int, help="opset version of the output onnx. default 13.")
parser.add_argument('--without_argmax', action="store_true", help="Segmentation output with argmax or not")
args = parser.parse_args()

transform_to_onnx(args.config, args.weightfile,
                  args.onnx_file_path, args.batch_size,
                  args.opset_version, argmax=not args.without_argmax)
