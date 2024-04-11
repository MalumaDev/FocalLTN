from warnings import warn

import tensorflow as tf

"""
Element-wise fuzzy logic operators for tensorflow.
Supports traditional NumPy/Tensorflow broadcasting.

To use in LTN formulas (broadcasting w.r.t. ltn variables appearing in a formula), 
wrap the operator with `ltn.WrapperConnective` or `ltn.WrapperQuantifier`. 
"""

eps = 1e-4
not_zeros = lambda x: (1 - eps) * x + eps
not_ones = lambda x: (1 - eps) * x


class Not_Std:
    def __call__(self, x):
        return 1. - x


class Not_Godel:
    def __call__(self, x):
        return tf.cast(tf.equal(x, 0), x.dtype)


class And_Min:
    def __call__(self, x, y):
        return tf.minimum(x, y)


class And_Prod:
    def __init__(self, stable=True):
        self.stable = stable

    def __call__(self, x, y, stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = not_zeros(x), not_zeros(y)
        return tf.multiply(x, y)


class And_Luk:
    def __call__(self, x, y):
        return tf.maximum(x + y - 1., 0.)


class Or_Max:
    def __call__(self, x, y):
        return tf.maximum(x, y)


class Or_ProbSum:
    def __init__(self, stable=True):
        self.stable = stable

    def __call__(self, x, y, stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = not_ones(x), not_ones(y)
        return x + y - tf.multiply(x, y)


class Or_Luk:
    def __call__(self, x, y):
        return tf.minimum(x + y, 1.)


class Implies_KleeneDienes:
    def __call__(self, x, y):
        return tf.maximum(1. - x, y)


class Implies_Godel:
    def __call__(self, x, y):
        return tf.where(tf.less_equal(x, y), tf.ones_like(x), y)


class Implies_Reichenbach:
    def __init__(self, stable=True):
        self.stable = stable

    def __call__(self, x, y, stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x, y = not_zeros(x), not_ones(y)
        return 1. - x + tf.multiply(x, y)


class Implies_Goguen:
    def __init__(self, stable=True):
        self.stable = stable

    def __call__(self, x, y, stable=None):
        stable = self.stable if stable is None else stable
        if stable:
            x = not_zeros(x)
        return tf.where(tf.less_equal(x, y), tf.ones_like(x), tf.divide(y, x))


class Implies_Luk:
    def __call__(self, x, y):
        return tf.minimum(1. - x + y, 1.)


class Equiv:
    """Returns an operator that computes: And(Implies(x,y),Implies(y,x))"""

    def __init__(self, and_op, implies_op):
        self.and_op = and_op
        self.implies_op = implies_op

    def __call__(self, x, y):
        return self.and_op(self.implies_op(x, y), self.implies_op(y, x))


class Aggreg_Min:
    def __call__(self, xs, axis=None, keepdims=False):
        return tf.reduce_min(xs, axis=axis, keepdims=keepdims)


class Aggreg_Max:
    def __call__(self, xs, axis=None, keepdims=False):
        return tf.reduce_max(xs, axis=axis, keepdims=keepdims)


class Aggreg_Mean:
    def __call__(self, xs, axis=None, keepdims=False):
        return tf.reduce_mean(xs, axis=axis, keepdims=keepdims)


class Aggreg_pMean:
    def __init__(self, p=2, stable=True):
        self.p = p
        self.stable = stable

    def __call__(self, xs, axis=None, keepdims=False, p=None, stable=None):
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        if stable:
            xs = not_zeros(xs)
        return tf.pow(tf.reduce_mean(tf.pow(xs, p), axis=axis, keepdims=keepdims), 1 / p)


class Aggreg_pMeanError:
    def __init__(self, p=2, stable=True):
        self.p = p
        self.stable = stable

    def __call__(self, xs, axis=None, keepdims=False, p=None, stable=None):
        p = self.p if p is None else p
        stable = self.stable if stable is None else stable
        if stable:
            xs = not_ones(xs)
        return 1. - tf.pow(tf.reduce_mean(tf.pow(1. - xs, p), axis=axis, keepdims=keepdims), 1 / p)


class Aggreg_Prod:
    def __call__(self, xs, axis=None, keepdims=False):
        return tf.reduce_prod(xs, axis=axis, keepdims=keepdims)


#########################
# Log-Product Operators #
#########################
class Aggreg_LogProd:
    def __init__(self, stable=True):
        warn("`Aggreg_LogProd` outputs values out of the truth value range [0,1]. "
             "Its usage with other connectives could be compromised."
             "Use it carefully.", UserWarning)
        self.stable = stable

    def __call__(self, xs, stable=None, axis=None, keepdims=False):
        stable = self.stable if stable is None else stable
        if stable:
            xs = not_zeros(xs)
        return tf.reduce_sum(tf.math.log(xs), axis=axis, keepdims=keepdims)


class Aggreg_Sum:
    def __init__(self) -> None:
        warn("`Aggreg_Sum` outputs values out of the truth value range [0,1]. "
             "Its usage with other connectives could be compromised."
             "Use it carefully.", UserWarning)

    def __call__(self, xs, axis=None, keepdims=False):
        return tf.reduce_sum(xs, axis=axis, keepdims=keepdims)


Aggreg_SumLog = Aggreg_LogProd



class FocalAggreg():

    def __init__(self, gamma=2, stable=True, is_log=True, alpha=1):
        # self.p = p
        self.stable = stable
        self.alpha = alpha
        self.gamma = gamma
        self.is_log = is_log

    def __repr__(self):
        return "FocalAggreg(gamma=" + str(self.gamma) + ", stable=" + str(self.stable) + ")"

    def __call__(self, xs, axis=None, keepdims=False, **kwargs):
        """
        It applies the `pMeanError` aggregation operator to the given formula's :ref:`grounding <notegrounding>`
        on the selected dimensions.

        Parameters
        ----------
        xs : :class:`torch.Tensor`
            :ref:`Grounding <notegrounding>` of formula on which the aggregation has to be performed.
        dim : :obj:`tuple` of :obj:`int`, default=None
            Tuple containing the indexes of dimensions on which the aggregation has to be performed.
        keepdim : :obj:`bool`, default=False
            Flag indicating whether the output has to keep the same dimensions as the input after
            the aggregation.
        mask : :class:`torch.Tensor`, default=None
            Boolean mask for excluding values of 'xs' from the aggregation. It is internally used for guarded
            quantification. The mask must have the same shape of 'xs'. `False` means exclusion, `True` means inclusion.
        p : :obj:`int`, default=None
            Value of hyper-parameter `p` of the `pMeanError` fuzzy aggregation operator.
        stable: :obj:`bool`, default=None
            Flag indicating whether to use the :ref:`stable version <stable>` of the operator or not.

        Returns
        ----------
        :class:`torch.Tensor`
            `pMeanError` fuzzy aggregation of the formula.

        Raises
        ------
        :class:`ValueError`
            Raises when the :ref:`grounding <notegrounding>` of the formula ('xs') and the mask do not have the same
            shape.
            Raises when the 'mask' is not boolean.
        """
        if self.is_log:
            pt = tf.math.exp(xs/self.alpha)
        else:
            pt = xs
            xs = tf.math.log(xs)*self.alpha

        return self.alpha * tf.math.reduce_sum(tf.math.multiply(tf.math.pow((1 - pt), self.gamma), xs), axis=axis,
                                               keepdims=keepdims)

