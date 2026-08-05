# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from sim_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx


def mask_iou_loss(masks1, masks2, reduction='mean'):
    intersection = torch.sum(masks1 * masks2, dim=1)
    area1 = torch.sum(masks1, dim=1)
    area2 = torch.sum(masks2, dim=1)
    union = area1 + area2 - intersection
    iou = intersection / union
    
    if reduction == 'mean':
        return 1 - iou.mean()
    elif reduction == 'sum':
        return 1 - iou.sum()
    elif reduction == 'none':
        return 1 - iou
    else:
        raise ValueError(f"reduction '{reduction}' not supported")


def cost_mask_loss(pred_masks, tgt_masks):
    '''
    pred_masks: (N1, max_v_l)
    tgt_masks: (N2, max_v_l)
    
    return: (N1, N2)
    '''
    N1 = pred_masks.size(0)
    N2 = tgt_masks.size(0)
    pred_masks = pred_masks.unsqueeze(1).expand(N1, N2, -1).flatten(0, 1)
    tgt_masks = tgt_masks.unsqueeze(0).expand(N1, N2, -1).flatten(0, 1)
    cost = mask_iou_loss(pred_masks, tgt_masks, reduction='none')
    cost = cost.view(N1, N2)
    return cost


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,  cost_class: float = 1, cost_span: float = 1, cost_giou: float = 1,
                 span_loss_type: str = "l1", max_v_l: int = 75,
                 cost_mask_iou: float = 6):
        """Creates the matcher

        Params:
            cost_span: This is the relative weight of the L1 error of the span coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the spans in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.cost_giou = cost_giou
        self.cost_mask_iou = cost_mask_iou
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.foreground_label = 0
        assert cost_class != 0 or cost_span != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_spans": Tensor of dim [batch_size, num_queries, 2] with the predicted span coordinates,
                    in normalized (cx, w) format
                 ""pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "spans": Tensor of dim [num_target_spans, 2] containing the target span coordinates. The spans are
                    in normalized (cx, w) format

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_spans)
        """
        bs, num_queries = outputs["pred_spans"].shape[:2]
        span_targets = targets["span_labels"]
        mask_targets = targets["mask_labels"]

        # Also concat the target labels and spans
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        tgt_spans = torch.cat([v["spans"] for v in span_targets])  # [num_target_spans in batch, 2]
        tgt_ids = torch.full([len(tgt_spans)], self.foreground_label)   # [total #spans in the batch]
        tgt_masks = torch.cat([v for v in mask_targets])  # [total #spans in the batch, max_v_l]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]  # [batch_size * num_queries, total #spans in the batch]

        if self.span_loss_type == "l1":
            # We flatten to compute the cost matrices in a batch
            out_spans = outputs["pred_spans"].flatten(0, 1)  # [batch_size * num_queries, 2]

            # Compute the L1 cost between spans
            cost_span = torch.cdist(out_spans, tgt_spans, p=1)  # [batch_size * num_queries, total #spans in the batch]

            # Compute the giou cost between spans
            # [batch_size * num_queries, total #spans in the batch]
            cost_giou = - generalized_temporal_iou(span_cxw_to_xx(out_spans), span_cxw_to_xx(tgt_spans))

            out_masks = outputs["pred_masks"].flatten(0, 1)  # [batch_size * num_queries, max_v_l]
            cost_mask_iou = cost_mask_loss(out_masks, tgt_masks)
        else:
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, max_v_l * 2)
            pred_spans = pred_spans.view(bs * num_queries, 2, self.max_v_l).softmax(-1)  # (bsz * #queries, 2, max_v_l)
            cost_span = - pred_spans[:, 0][:, tgt_spans[:, 0]] - pred_spans[:, 1][:, tgt_spans[:, 1]]  # (bsz * #queries, #spans)
            # pred_spans = pred_spans.repeat(1, n_spans, 1, 1).flatten(0, 1)  # (bsz * #queries * #spans, max_v_l, 2)
            # tgt_spans = tgt_spans.view(1, n_spans, 2).repeat(bs * num_queries, 1, 1).flatten(0, 1)  # (bsz * #queries * #spans, 2)
            # cost_span = pred_spans[tgt_spans]
            # cost_span = cost_span.view(bs * num_queries, n_spans)

            # giou
            cost_giou = 0

        # Final cost matrix
        # import ipdb; ipdb.set_trace()
        C = self.cost_span * cost_span + self.cost_giou * cost_giou + self.cost_class * cost_class + self.cost_mask_iou * cost_mask_iou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["spans"]) for v in span_targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(
        cost_span=args.set_cost_span, cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class, cost_mask_iou=args.set_cost_mask,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l
    )
