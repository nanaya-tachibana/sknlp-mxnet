from mxnet.gluon.loss import SigmoidBCELoss, _apply_weighting, _reshape_like


class SampledSigmoidBCELoss(SigmoidBCELoss):

    def hybrid_forward(self, F, pred, label, label_mask,
                       sample_weight=None, pos_weight=None):
        label = _reshape_like(F, label, pred)
        if not self._from_sigmoid:
            if pos_weight is None:
                # We use the stable formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
                loss = F.relu(pred) - pred * label + \
                    F.Activation(-F.abs(pred), act_type='softrelu')
            else:
                # We use the stable formula: x - x * z + (1 + z * pos_weight - z) * \
                #    (log(1 + exp(-abs(x))) + max(-x, 0))
                log_weight = 1 + F.broadcast_mul(pos_weight - 1, label)
                loss = (
                    pred - pred * label + log_weight * (
                        F.Activation(-F.abs(pred),
                                     act_type='softrelu') + F.relu(-pred)
                    )
                )
        else:
            eps = 1e-12
            if pos_weight is None:
                loss = -(
                    F.log(pred + eps) * label +
                    F.log(1. - pred + eps) * (1. - label)
                )
            else:
                loss = -(
                    F.broadcast_mul(F.log(pred + eps) * label, pos_weight)
                    + F.log(1. - pred + eps) * (1. - label)
                )
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss * label_mask, axis=self._batch_axis, exclude=True)
