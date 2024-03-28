import tensorflow as tf


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
