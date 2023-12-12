from torch import nn, tensor, float32, stack


class Graph(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(tensor([0.950, 0.288], dtype=float32))

    def forward(self, x1: float32, x2: float32, x3: float32) -> float32:
        w0, w1 = self.w

        r1 = w0 * x3 * w1 + w1 * x2 + x3 ** 2
        r2 = (x3 + w1) * (x1 ** 3 - x2 ** 2)
        r3 = x2 ** 3 * w1 / w0
        return (r1 * r2 * r3 + x3) / (x2 + r2 + r3)

    def your_forward_backward(self, x1, x2, x3):
        w0, w1 = self.w

        # forward
        a0 = w0 * x3
        a1 = a0 * w1  # w0 * x3 * w1
        a2 = w1 * x2
        a3 = x3 ** 2
        a4 = a1 + a2
        a5 = a4 + a3  # r1

        a6 = x3 + w1
        a7 = x1 ** 3
        a8 = x2 ** 2
        a9 = a7 - a8
        a10 = a6 * a9  # r2

        a11 = a8 * x2
        a12 = a11 * w1
        a13 = a12 / w0  # r3

        a14 = a5 * a10
        a15 = a14 * a13
        a16 = a15 + x3
        a17 = x2 + a10
        a18 = a17 + a13
        a19 = a16 / a18

        # backward
        da19 = 1.0
        da18 = da19 * (-a16 / a18 ** 2)
        da17 = da18
        da16 = da19 / a18
        da15 = da16
        da14 = da15 * a13
        da13 = da18 + da15 * a14
        da12 = da13 / w0
        # da11 = da12 * w1 not used
        da10 = da17 + da14 * a5
        # da9 = da10 * a6 not used
        # da8 = da11 * x2 - da9 not used
        # da7 = da9 not used
        da6 = da10 * a9
        da5 = da14 * a10
        da4 = da5
        # da3 = da5
        da2 = da4
        da1 = da4
        da0 = da1 * w1

        dw0 = da13 * (-a12 / w0 ** 2) + da0 * x3
        dw1 = da12 * a11 + da6 + da2 * x2 + da1 * a0

        self.w.grad = stack([dw0, dw1])

        return a19
