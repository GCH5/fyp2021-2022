import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ],
)

def get_train_image_only_transforms():
    return A.Compose(
        [
            A.Normalize(p=1.0,max_pixel_value=1.0),
            ToTensorV2(p=1.0),
        ],
        additional_targets={'image': 'image'}
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Normalize(p=1.0,max_pixel_value=1.0),
            ToTensorV2(p=1.0),
        ]
    )