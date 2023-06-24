import sys
import torch
from tool.darknet2pytorch import Darknet


def transform_to_onnx(cfgfile, weightfile, onnx_file_name, batch_size=1, opset_version=13):
    model = Darknet(cfgfile)

    model.print_network()
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    input_names = ["input"]
    output_names = ['boxes', 'confs', "seg", "seg_labeled"]

    if dynamic:
        x = torch.randn((1, 3, model.height, model.width), requires_grad=True)
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, model.height, model.width), requires_grad=True)
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        return onnx_file_name


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('weightfile')
    parser.add_argument('--batch_size', type=int, help="Static Batchsize of the model. use batch_size<=0 for dynamic batch size")
    parser.add_argument('--onnx_file_path', help="Output onnx file path")
    args = parser.parse_args()
    transform_to_onnx(args.config, args.weightfile, args.batch_size, args.onnx_file_path)

