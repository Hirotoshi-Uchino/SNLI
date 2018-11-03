from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class ElementwiseProduct(Layer):

    def __init__(self, row_dim, **kwargs):
        self.row_dim = row_dim # L x D dimeision
        super(ElementwiseProduct, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.row_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(ElementwiseProduct, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input):
        return self.kernel * input

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.row_dim, input_shape[1])

    def get_config(self):
        config = {
            'kernel': K.eval(self.kernel),
        }
        base_config = super(ElementwiseProduct, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))