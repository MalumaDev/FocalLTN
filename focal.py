import tensorflow as tf


class FocalAggreg:

    def __init__(self, gamma=2, stable=True, is_log=True, reduce_type='mean'):
        # self.p = p
        self.stable = stable
        self.gamma = gamma
        self.is_log = is_log
        self.revert_log = tf.math.exp if is_log else lambda x: x
        if reduce_type == 'sum':
            self.reduce_type = tf.math.reduce_sum
        elif reduce_type == 'mean':
            self.reduce_type = tf.math.reduce_mean
        else:
            raise ValueError("reduce_type must be 'sum' or 'mean'")

    def __repr__(self):
        return "FocalAggreg(gamma=" + str(self.gamma) + ", stable=" + str(self.stable) + ")"

    def __call__(self, xs, axis=None, keepdims=False, **kwargs):
        """
        It applies the `Focal Loss` to the input tensor `xs` along the axis `axis`.

        """
        if self.is_log:
            pt = tf.math.exp(xs)
        else:
            pt = xs
            xs = tf.math.log(xs)

        return self.reduce_type(self.revert_log(tf.math.multiply(tf.math.pow((1 - pt), self.gamma), xs)), axis=axis,
                                keepdims=keepdims)