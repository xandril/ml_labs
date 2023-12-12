from torch import nn, tensor, float32, stack


class Graph(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(tensor([0.950, 0.288], dtype=float32))

    def forward(self, x1, x2, x3):
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

        a1 = a2 = None  # not used

        a5 = a4 + a3  # r1

        a3 = a4 = None  # not used

        a6 = x3 + w1
        a7 = x1 ** 3

        x1 = None  # not used

        a8 = x2 ** 2
        a9 = a7 - a8

        a7 = None  # not used

        a10 = a6 * a9  # r2

        a6 = None  # not used

        a11 = a8 * x2

        a8 = None  # not used

        a12 = a11 * w1
        a13 = a12 / w0  # r3

        a14 = a5 * a10
        a15 = a14 * a13
        a16 = a15 + x3

        a15 = None  # not used

        a17 = x2 + a10
        a18 = a17 + a13

        a17 = None  # not used

        a19 = a16 / a18

        # backward
        da19 = 1.0
        da18 = da19 * (-a16 / a18 ** 2)

        a16 = None  # not used

        da17 = da18
        da16 = da19 / a18

        a18 = da19 = None  # not used

        da15 = da16

        da16 = None

        da14 = da15 * a13

        a13 = None  # not used

        da13 = da18 + da15 * a14

        a14 = da15 = da18 = None  # not used

        da12 = da13 / w0

        # da11 = da12 * w1 not used

        da10 = da17 + da14 * a5

        a5 = da17 = None  # not used

        # da9 = da10 * a6 not used
        # da8 = da11 * x2 - da9 not used
        # da7 = da9 not used

        da6 = da10 * a9

        a9 = da10 = None  # not used

        da5 = da14 * a10

        da14 = None

        da4 = da5

        da5 = None

        # da3 = da5 # not used

        da2 = da4
        da1 = da4

        da4 = None  # not used

        da0 = da1 * w1

        w1 = None  # not used

        dw0 = da13 * (-a12 / w0 ** 2) + da0 * x3

        x3 = w0 = a12 = da0 = da13 = None  # not used

        dw1 = da12 * a11 + da6 + da2 * x2 + da1 * a0

        x2 = a0 = a11 = da1 = da2 = da6 = da12 = None  # not used

        self.w.grad = stack([dw0, dw1])

        return a19
