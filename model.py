import numpy as np
from layers import ConvLayer, MaxPoolLayer, DenseLayer, relu, softmax

class CropDiseaseCNN:
    def __init__(self):
        self.conv = ConvLayer(num_filters=8, filter_size=3)
        self.pool = MaxPoolLayer(size=2)
        self.dense = DenseLayer(15*15*8, 2)  

    def forward(self, image):
        output = self.conv.forward(image)
        output = relu(output)
        output = self.pool.forward(output)

        output = output.flatten()
        output = self.dense.forward(output)
        return softmax(output)