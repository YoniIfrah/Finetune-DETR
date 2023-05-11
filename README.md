# FridgeIT dataset
This is dataset of products which we collected manually by iphone 12 pro max camera.

The dataset contains a list of 5 products: Butter, Cottage, Cream, Milk and Mustard.
# Finetune DETR

The goal of this notebook is to fine-tune Facebook's DETR (DEtection TRansformer).

<img alt="With pre-trained DETR" src="https://github.com/YoniIfrah/DETR_Object_Detection/blob/b01f8c41e4e6b77b868149dca260f57030e0a876/assets/images/current.jpeg" width="375"> -> <img alt="With finetuned DETR" src="https://github.com/YoniIfrah/DETR_Object_Detection/blob/b01f8c41e4e6b77b868149dca260f57030e0a876/assets/images/output.jpeg" width="375">

From left to right: results obtained with pre-trained DETR, and after fine-tuning on the `fridgeIT` dataset.

## Data

DETR will be fine-tuned on a tiny dataset: the [`fridgeIT` dataset](https://github.com/YoniIfrah/DETR_Object_Detection/tree/main/custom).
We refer to it as the `custom` dataset.

There are 2094 images in the training set, and 526 images in the validation set.

We expect the directory structure to be the following:
```
path/to/coco/
├ annotations/  # JSON annotations
│  ├ annotations/custom_train.json
│  └ annotations/custom_val.json
├ train2017/    # training images
└ val2017/      # validation images
```

## Metrics

Typical metrics to monitor, partially shown in [this notebook][metrics-notebook], include:
-   the Average Precision (AP), which is [the primary challenge metric](https://cocodataset.org/#detection-eval) for the COCO dataset,
-   losses (total loss, classification loss, l1 bbox distance loss, GIoU loss),
-   errors (cardinality error, class error).

As mentioned in [the paper](https://arxiv.org/abs/2005.12872), there are 3 components to the matching cost and to the total loss:
-   classification loss,
```python
def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    [...]
    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
    losses = {'loss_ce': loss_ce}
```
-   l1 bounding box distance loss,
```python
def loss_boxes(self, outputs, targets, indices, num_boxes):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h),normalized by the image
       size.
    """
    [...]
    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes
```
-   [Generalized Intersection over Union (GIoU)](https://giou.stanford.edu/) loss, which is scale-invariant.
```python
    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(src_boxes),
        box_ops.box_cxcywh_to_xyxy(target_boxes)))
    losses['loss_giou'] = loss_giou.sum() / num_boxes
```

Moreover, there are two errors:
-   cardinality error,
```python
def loss_cardinality(self, outputs, targets, indices, num_boxes):
    """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty
    boxes. This is not really a loss, it is intended for logging purposes only. It doesn't
    propagate gradients
    """
    [...]
    # Count the number of predictions that are NOT "no-object" (which is the last class)
    card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
    card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
    losses = {'cardinality_error': card_err}
```
-   [class error](https://github.com/facebookresearch/detr/blob/5e66b4cd15b2b182da347103dd16578d28b49d69/models/detr.py#L126),
```python
    # TODO this should probably be a separate loss, not hacked in this one here
    losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
```
where [`accuracy`](https://github.com/facebookresearch/detr/blob/5e66b4cd15b2b182da347103dd16578d28b49d69/util/misc.py#L432) is:
```python
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
```
# finetune-detr
