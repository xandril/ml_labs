import torch

from lab1_backprop.graph import Graph


def test_model_class(model_class):
    model = model_class()

    for i in range(10):
        x1, x2, x3 = torch.rand(3)

        model.zero_grad()
        y_torch = model(x1, x2, x3)
        y_torch.backward()
        grad_torch = model.w.grad.clone()

        model.zero_grad()
        with torch.no_grad():
            y_manual = model.your_forward_backward(x1, x2, x3)
        grad_manual = model.w.grad.clone()

        print(f'step {i}')
        print('forward')
        print(f'torch: {y_torch.clone().detach().numpy()}, manual: {y_manual.numpy()}')

        print('grad')
        print(f'torch: {grad_torch.clone().numpy()}, manual: {grad_manual.numpy()}')

        assert torch.allclose(y_manual, y_torch, rtol=5e-05, atol=1e-7)
        assert torch.allclose(grad_manual, grad_torch, rtol=5e-05, atol=1e-7)

    print('Tests completed successfully!')


if __name__ == '__main__':
    test_model_class(Graph)
