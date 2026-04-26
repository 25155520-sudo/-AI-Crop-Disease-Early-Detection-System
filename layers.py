import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / 9

    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - self.filter_size + 1,
                           w - self.filter_size + 1,
                           self.num_filters))

        for f in range(self.num_filters):
            for i in range(h - self.filter_size + 1):
                for j in range(w - self.filter_size + 1):
                    region = input[i:i+self.filter_size, j:j+self.filter_size]
                    output[i, j, f] = np.sum(region * self.filters[f])

        return output

class MaxPoolLayer:
    def __init__(self, size):
        self.size = size

    def forward(self, input):
        h, w, num_filters = input.shape
        output = np.zeros((h // self.size,
                           w // self.size,
                           num_filters))

        for f in range(num_filters):
            for i in range(0, h, self.size):
                for j in range(0, w, self.size):
                    region = input[i:i+self.size, j:j+self.size, f]
                    output[i//self.size, j//self.size, f] = np.max(region)

        return output

class DenseLayer:
    def __init__(self, input_len, output_len):
        self.weights = np.random.randn(input_len, output_len) / input_len
        self.bias = np.zeros(output_len)

    def forward(self, input):
        self.last_input = input
        return np.dot(input, self.weights) + self.bias

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)