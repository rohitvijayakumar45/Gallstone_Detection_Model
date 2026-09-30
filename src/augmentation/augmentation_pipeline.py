import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .acoustic_shadow_aug import AcousticShadowAugment
from .depth_attenuation import DepthAttenuationTransform
from .ultrasound_augments import ReverberationArtifact, SpeckleNoiseTransform


class UltrasoundAugmentationPipeline:
    """Training-only ultrasound augmentation plus val/TTA transforms."""

    def get_train_transforms(self):
        return A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
                A.RandomGamma(gamma_limit=(70, 130), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                DepthAttenuationTransform(max_attenuation=0.4, p=0.4),
                AcousticShadowAugment(shadow_intensity_range=(0.3, 0.8), p=0.3),
                SpeckleNoiseTransform(mean=0, std_range=(0.01, 0.05), p=0.4),
                ReverberationArtifact(p=0.2),
                A.RandomScale(scale_limit=0.2, p=0.4),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.0,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.3,
                ),
                A.ElasticTransform(
                    alpha=30,
                    sigma=5,
                    alpha_affine=5,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.2,
                ),
                A.CoarseDropout(
                    max_holes=3,
                    max_height=40,
                    max_width=40,
                    fill_value=0,
                    p=0.2,
                ),
                A.GridDistortion(num_steps=5, distort_limit=0.2, border_mode=cv2.BORDER_CONSTANT, p=0.15),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
        )

    def get_val_transforms(self):
        return A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    def get_tta_transforms(self):
        norm = lambda: A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0)
        return [
            A.Compose([norm(), ToTensorV2()]),
            A.Compose([A.HorizontalFlip(p=1), norm(), ToTensorV2()]),
            A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1), norm(), ToTensorV2()]),
            A.Compose([A.Rotate(limit=(5, 5), p=1), norm(), ToTensorV2()]),
            A.Compose([A.Rotate(limit=(-5, -5), p=1), norm(), ToTensorV2()]),
            A.Compose([A.RandomGamma(gamma_limit=(85, 115), p=1), norm(), ToTensorV2()]),
        ]
