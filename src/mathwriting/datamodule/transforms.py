import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_resize_pad_transform(img_size):
    return [
        A.LongestMaxSize(max_size=max(img_size), interpolation=3),
        A.PadIfNeeded(min_height=img_size[1], min_width=img_size[0],
                      border_mode=0, fill=(255, 255, 255))
    ]

def get_train_transform(img_size, mean, std):
    resize_pad = get_resize_pad_transform(img_size)
    return A.Compose([
        *resize_pad,
        A.Compose([
            A.Affine(translate_percent=0, scale=(0.85, 1.0), rotate=1, border_mode=0,
                     interpolation=3, fill=(255, 255, 255), p=1),
            A.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3,
                             fill=(255, 255, 255), p=0.5),
        ], p=1),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.GaussNoise(std_range=(0.0392, 0.0392), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=(-0.2, 0), p=0.2),
        A.ImageCompression(quality_range=(90, 95), p=0.2),
        A.ToGray(p=1.0),
        A.Normalize(mean=(mean,) * 3, std=(std,) * 3),
        ToTensorV2()
    ])

def get_val_test_transform(img_size, mean, std):
    resize_pad = get_resize_pad_transform(img_size)
    return A.Compose([
        *resize_pad,
        A.ToGray(p=1.0),
        A.Normalize(mean=(mean,) * 3, std=(std,) * 3),
        ToTensorV2()
    ])
