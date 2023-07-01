import sys
import torch
from tool.darknet2pytorch import Darknet


def transform_to_onnx(cfgfile, weightfile, onnx_file_name,
                      batch_size=1, opset_version=13,
                      argmax=True):
    model = Darknet(cfgfile, argmax=argmax)

    model.print_network()
    model.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    dummy_input = torch.randn((1, 3, model.height, model.width), requires_grad=True)
    dummy_output = model(dummy_input)

    segmentation = isinstance(dummy_output, tuple)

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    input_names = ["input"]
    if segmentation:
        output_names = ['boxes', 'confs', "seg"]
    else:
        output_names = ['boxes', 'confs']

    if dynamic:
        dynamic_axes = {"input": {0: "batch_size"},
                        "boxes": {0: "batch_size"},
                        "confs": {0: "batch_size"}}
        if segmentation:
            dynamic_axes["seg"] = {0: "batch_size"}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          dummy_input,
                          onnx_file_name,
                          export_params=True,
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        torch.onnx.export(model,
                          dummy_input,
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

