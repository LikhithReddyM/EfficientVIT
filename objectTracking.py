import cv2
import torch
from configs.retinanet_efficientvit_m4_fpn_1x_coco import model
from mmcv_custom import load_state_dict
# from sort import Sort

# Load the pre-trained object detection model
model_path = './retinanet_efficientvit_m4_fpn_1x_coco.pth'
device = torch.device('cpu')

# _base_ = [
#     './configs/_base_/models/retinanet_efficientvit_fpn.py',
#     './configs/_base_/datasets/coco_detection.py',
#     './configs/_base_/schedules/schedule_1x.py', 
#     './configs/_base_/default_runtime.py'
# ]

# model = dict(
#     pretrained=None,
#     backbone=dict(        
#         type='EfficientViT_M4',
#         pretrained="/root/efficientvit_m4.pth",
#         frozen_stages=-1,
#         ),
#     neck=dict(
#         type='EfficientViTFPN',
#         in_channels=[128, 256, 384],
#         out_channels=256,
#         start_level=0,
#         num_outs=5,
#         num_extra_trans_convs=1,
#         ))

# # optimizer
# optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
#                  paramwise_cfg=dict(custom_keys={'attention_biases': dict(decay_mult=0.),
#                                                  'attention_bias_idxs': dict(decay_mult=0.),
#                                                  }))
# # optimizer_config = dict(grad_clip=None)
# # do not use mmdet version fp16
# # fp16 = None
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# total_epochs = 12

# model = dict(
#     type='RetinaNet',
#     pretrained='torchvision://resnet50',
#     backbone=dict(
#         type='EfficientViT_M4',
#         pretrained="",),
#     neck=dict(
#         type='EfficientViTFPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         start_level=1,
#         add_extra_convs='on_input',
#         num_outs=5),
#     bbox_head=dict(
#         type='RetinaHead',
#         num_classes=80,
#         in_channels=256,
#         stacked_convs=4,
#         feat_channels=256,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             octave_base_scale=4,
#             scales_per_octave=3,
#             ratios=[0.5, 1.0, 2.0],
#             strides=[8, 16, 32, 64, 128]),
#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoder',
#             target_means=[.0, .0, .0, .0],
#             target_stds=[1.0, 1.0, 1.0, 1.0]),
#         loss_cls=dict(
#             type='FocalLoss',
#             use_sigmoid=True,
#             gamma=2.0,
#             alpha=0.25,
#             loss_weight=1.0),
#         loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
#     # training and testing settings
#     train_cfg=dict(
#         assigner=dict(
#             type='MaxIoUAssigner',
#             pos_iou_thr=0.5,
#             neg_iou_thr=0.4,
#             min_pos_iou=0,
#             ignore_iof_thr=-1),
#         allowed_border=-1,
#         pos_weight=-1,
#         debug=False),
#     test_cfg=dict(
#         nms_pre=1000,
#         min_bbox_size=0,
#         score_thr=0.05,
#         nms=dict(type='nms', iou_threshold=0.5),
#         max_per_img=100))

# Load the state dictionary using torch.load
# checkpoint = torch.load(model_path, map_location=device)

# If the checkpoint is a dictionary (not an OrderedDict), use it directly
# if isinstance(checkpoint, dict):
load_state_dict(model, torch.load(model_path))
# else:
    # If it's not a dictionary, assume it's the state dictionary itself
    # load_state_dict(model, torch.load(model_path))

model.eval()
model.to(device)

# # Create a tracker instance (SORT)
# # tracker = Sort()

# # Open a video capture or use webcam (replace 'video_file.mp4' with 0 for webcam)
# cap = cv2.VideoCapture('video_file.mp4')

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Preprocess the frame (you may need to adapt this part based on your model requirements)
#     image_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(device)
#     inputs = [image_tensor]

#     # Perform object detection
#     with torch.no_grad():
#         predictions = model(inputs)

#     # Extract bounding box coordinates from predictions
#     boxes = predictions[0]['boxes'].cpu().numpy() if predictions else []

#     # Update the tracker with new detections
#     # trackers = tracker.update(boxes)

#     # Draw bounding boxes on the frame
#     for i, det in enumerate(trackers):
#         x, y, w, h, track_id = det
#         cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
#         cv2.putText(frame, str(int(track_id)), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Display the result
#     cv2.imshow('Object Tracking', frame)

#     # Break the loop if 'q' key is pressed
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# # Release the video capture object
# cap.release()

# # Close all OpenCV windows
# cv2.destroyAllWindows()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Preprocess the frame (you may need to adapt this part based on your model requirements)
#     image_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(device)
#     inputs = [image_tensor]

#     # Perform object detection
#     with torch.no_grad():
#         predictions = model(inputs)

#     # Extract bounding box coordinates from predictions
#     boxes = predictions[0]['boxes'].cpu().numpy() if predictions else []

#     # Draw bounding boxes on the frame
#     for i, box in enumerate(boxes):
#         x_min, y_min, x_max, y_max = map(int, box)
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#     # Display the result
#     cv2.imshow('Object Detection', frame)

#     # Break the loop if 'q' key is pressed
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# # Release the video capture object
# cap.release()

# # Close all OpenCV windows
# cv2.destroyAllWindows()