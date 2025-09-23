from typing import List, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
# <--- Toni --->
class CustomImageDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        target: Optional[List[float]] = None,
        preprocesser: Optional[T.Compose] = None,
        video_len: Optional[int] = None
    ):
        super().__init__()
        self.image_paths = image_paths
        self.video_len = video_len
        self.target = target

        # Use provided preprocesser or default CLIP-style
        if preprocesser:
            self.preprocesser = preprocesser
        else:
            self.preprocesser = T.Compose([
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        img_path = self.image_paths[i]
        img = Image.open(img_path).convert('RGB')
        img = self.preprocesser(img)

        if self.video_len is not None:
            img = img.repeat(self.video_len, 1, 1, 1)

        if self.target is not None:
            target = torch.tensor(self.target[i]).float()
        else:
            target = torch.tensor(0.0)  # dummy

        return img, target
    
# </--- Toni --->