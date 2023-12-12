import torch

from lab1_backprop.graph import Graph

if __name__ == '__main__':
    model = Graph()

    x1, x2, x3 = torch.rand(3)
    # put onnx model to https://netron.app/ and get comp graph
    torch.onnx.export(
        model,
        (x1, x2, x3),
        'data/graph.onnx',
        opset_version=13,
        export_params=True,
        do_constant_folding=False,
        input_names=['x1', 'x2', 'x3'],
        output_names=['y'],
    )
