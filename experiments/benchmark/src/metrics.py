import torch
class AveragePrecision:
    """
    Taken from https://github.com/amdegroot/tnt
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.empty(0, dtype=torch.float32, requires_grad=False)
        self.targets = torch.empty(0, dtype=torch.int64, requires_grad=False)
        self.weights = torch.empty(0, dtype=torch.float32, requires_grad=False)

    def update(self, output, target, weight=None):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if weight is not None:
            if not torch.is_tensor(weight):
                weight = torch.from_numpy(weight)
            weight = weight.squeeze()
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if weight is not None:
            assert weight.dim() == 1, 'Weight dimension should be 1'
            assert weight.numel() == target.size(0), \
                'Weight dimension 1 should be the same as that of target'
            assert torch.min(weight) >= 0, 'Weight should be non-negative only'
        assert torch.equal(target**2, target), \
            'targets should be binary (0 or 1)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # store scores and targets
        previous_count = self.scores.size(0) if self.scores.numel() > 0 else 0
        output = output.detach().to(dtype=torch.float32)
        target = target.detach().to(dtype=torch.int64)
        if self.scores.numel() == 0:
            self.scores = output.clone()
            self.targets = target.clone()
        else:
            self.scores = torch.cat((self.scores, output), dim=0)
            self.targets = torch.cat((self.targets, target), dim=0)

        if weight is not None:
            weight = weight.detach().to(dtype=torch.float32)
            if self.weights.numel() == 0:
                previous_weights = torch.ones(previous_count, dtype=torch.float32)
                self.weights = torch.cat((previous_weights, weight), dim=0)
            else:
                self.weights = torch.cat((self.weights, weight), dim=0)
        elif self.weights.numel() > 0:
            unit_weights = torch.ones(output.size(0), dtype=torch.float32)
            self.weights = torch.cat((self.weights, unit_weights), dim=0)

    def get_metric(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0) + 1).float()
        if self.weights.numel() > 0:
            weight = self.weights.new(self.weights.size())
            weighted_truth = self.weights.new(self.weights.size())

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            _, sortind = torch.sort(scores, 0, True)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weighted_truth = truth.float() * weight
                rg = weight.cumsum(0)

            # compute true positive sums
            if self.weights.numel() > 0:
                tp = weighted_truth.cumsum(0)
            else:
                tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)

            # compute average precision
            ap[k] = precision[truth.bool()].sum() / max(truth.sum(), 1)
        return ap


class MeanAveragePrecision:
    def __init__(self):
        self.ap = AveragePrecision()

    def reset(self):
        self.ap.reset()

    def update(self, output, target, weight=None):
        self.ap.update(output, target, weight)

    def get_metric(self):
        return {'map': self.ap.get_metric().mean().item()}

    def get_primary_metric(self):
        return self.get_metric()['map'] 
