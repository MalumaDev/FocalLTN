import tensorflow as tf


class FocalAggreg:

    def __init__(self, p=2, gamma=2, stable=True):
        self.p = p
        self.stable = stable
        self.alpha = 1
        self.gamma = gamma

    def __repr__(self):
        return "FocalAggreg(p=" + str(self.p) + ", gamma=" + str(self.gamma) + ", stable=" + str(self.stable) + ")"

    def __call__(self, xs, dim=None, keepdim=False, mask=None, p=None, stable=None):
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
        xs = xs + 1e-80

        if mask is not None:
            if mask.shape != xs.shape:
                raise ValueError("'xs' and 'mask' must have the same shape.")
            if not mask.dtype == tf.bool:  # isinstance(mask, torch.BoolTensor):
                raise ValueError("'mask' must be a torch.BoolTensor.")
        else:
            mask = tf.ones_like(xs, dtype=tf.bool)

        masked = tf.where(~mask, tf.zeros_like(xs), xs)
        xs = tf.log(xs)

        return -self.alpha * tf.sum(tf.mul(tf.pow((1 - masked), self.gamma), xs), dim=dim,
                                    keepdim=keepdim)
