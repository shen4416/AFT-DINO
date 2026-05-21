from typing import Dict, List, Optional, Tuple, Union

import random
import PIL
from PIL.Image import Image
from torch import Tensor

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.solarize import RandomSolarization
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE


class DINOTransform(MultiViewTransform):
    """DINO multi-view transform with Local Statistics-Aligned Mixing.

    This class generates two global views and a user-defined number of local
    views. LSA-Mix is applied only to the local views after the standard DINO
    view transformations.
    """

    def __init__(
        self,
        global_crop_size: int = 224,
        global_crop_scale: Tuple[float, float] = (0.4, 1.0),
        local_crop_size: int = 96,
        local_crop_scale: Tuple[float, float] = (0.05, 0.4),
        n_local_views: int = 6,
        hf_prob: float = 0.5,
        vf_prob: float = 0,
        rr_prob: float = 0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.4,
        cj_hue: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: Tuple[float, float, float] = (1.0, 0.1, 0.5),
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        solarization_prob: float = 0.2,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        # First global view transform
        global_transform_0 = DINOViewTransform(
            crop_size=global_crop_size,
            crop_scale=global_crop_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_hue=cj_hue,
            cj_sat=cj_sat,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur[0],
            kernel_size=kernel_size,
            kernel_scale=kernel_scale,
            sigmas=sigmas,
            solarization_prob=0,
            normalize=normalize,
        )

        # Second global view transform
        global_transform_1 = DINOViewTransform(
            crop_size=global_crop_size,
            crop_scale=global_crop_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            cj_prob=cj_prob,
            cj_bright=cj_bright,
            cj_strength=cj_strength,
            cj_contrast=cj_contrast,
            cj_hue=cj_hue,
            cj_sat=cj_sat,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur[1],
            kernel_size=kernel_size,
            kernel_scale=kernel_scale,
            sigmas=sigmas,
            solarization_prob=solarization_prob,
            normalize=normalize,
        )

        # Local view transform
        local_transform = DINOViewTransform(
            crop_size=local_crop_size,
            crop_scale=local_crop_scale,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_hue=cj_hue,
            cj_sat=cj_sat,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur[2],
            kernel_size=kernel_size,
            kernel_scale=kernel_scale,
            sigmas=sigmas,
            solarization_prob=0,
            normalize=normalize,
        )

        local_transforms = [local_transform] * n_local_views
        transforms = [global_transform_0, global_transform_1]
        transforms.extend(local_transforms)
        super().__init__(transforms)

        # Store view transforms for explicit local-view LSA-Mix processing.
        self.global_transform_0 = global_transform_0
        self.global_transform_1 = global_transform_1
        self.local_transform = local_transform
        self.n_local_views = n_local_views

        # Internal LSA-Mix settings for local views.
        # LSA-Mix is enabled by default and can be disabled by setting local_lsa_mix=False.
        self.local_lsa_mix: bool = True
        # Mixing region area ratio range relative to the local view size.
        self.local_mix_min_ratio: float = 0.1
        self.local_mix_max_ratio: float = 0.3

    def _apply_lsa_mix_to_locals(self, local_views: List[Tensor]) -> List[Tensor]:
        """Apply Local Statistics-Aligned Mixing to local views.

        A subset of local views is kept unchanged. The remaining views are
        randomly paired. For each pair, a same-position rectangular region is
        exchanged after aligning the source patch statistics to the target
        patch statistics. The total number of local views is kept unchanged.
        """
        n = len(local_views)

        # Return unchanged views when there are too few local crops for stable pairing.
        if n < 4 or not self.local_lsa_mix:
            return local_views

        indices = list(range(n))
        # Keep two local views unchanged to preserve the standard DINO local-view distribution.
        keep_indices = sorted(random.sample(indices, 2))
        mix_indices = [idx for idx in indices if idx not in keep_indices]
        random.shuffle(mix_indices)

        new_locals: List[Tensor] = []

        # Add unchanged local views first.
        for idx in keep_indices:
            new_locals.append(local_views[idx])

        def _sample_box(h: int, w: int) -> Tuple[int, int, int, int]:
            """Sample a rectangular mixing region for a local view."""
            ratio = random.uniform(self.local_mix_min_ratio, self.local_mix_max_ratio)
            # Use sqrt(ratio) so the region area is approximately ratio * H * W.
            cut_w = max(1, int(w * (ratio ** 0.5)))
            cut_h = max(1, int(h * (ratio ** 0.5)))
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            x1 = max(cx - cut_w // 2, 0)
            x2 = min(cx + cut_w // 2, w)
            y1 = max(cy - cut_h // 2, 0)
            y2 = min(cy + cut_h // 2, h)
            return x1, y1, x2, y2

        def _align_patch_stats(src: Tensor, tgt: Tensor, eps: float = 1e-6) -> Tensor:
            """Align source-patch channel statistics to the target patch."""
            src_mean = src.mean(dim=(1, 2), keepdim=True)
            src_std = src.std(dim=(1, 2), keepdim=True, unbiased=False)
            tgt_mean = tgt.mean(dim=(1, 2), keepdim=True)
            tgt_std = tgt.std(dim=(1, 2), keepdim=True, unbiased=False)
            aligned = (src - src_mean) / (src_std + eps)
            aligned = aligned * tgt_std + tgt_mean
            return aligned

        # Pair the remaining local views and apply LSA-Mix.
        for i in range(0, len(mix_indices), 2):
            idx1 = mix_indices[i]
            if i + 1 >= len(mix_indices):
                # Keep an unmatched view unchanged when the number of mixed views is odd.
                new_locals.append(local_views[idx1])
                break

            idx2 = mix_indices[i + 1]
            v1 = local_views[idx1].clone()
            v2 = local_views[idx2].clone()

            # LSA-Mix expects C x H x W tensors; otherwise keep the views unchanged.
            if v1.dim() != 3 or v2.dim() != 3:
                new_locals.append(local_views[idx1])
                new_locals.append(local_views[idx2])
                continue

            _, h, w = v1.shape
            x1, y1, x2, y2 = _sample_box(h, w)
            if x2 <= x1 or y2 <= y1:
                # Keep the views unchanged if the sampled region is invalid.
                new_locals.append(local_views[idx1])
                new_locals.append(local_views[idx2])
                continue

            # Exchange same-position patches after local statistics alignment.
            patch1 = v1[:, y1:y2, x1:x2].clone()
            patch2 = v2[:, y1:y2, x1:x2].clone()

            patch2_to_1 = _align_patch_stats(patch2, patch1)
            patch1_to_2 = _align_patch_stats(patch1, patch2)

            v1[:, y1:y2, x1:x2] = patch2_to_1
            v2[:, y1:y2, x1:x2] = patch1_to_2

            new_locals.append(v1)
            new_locals.append(v2)

        # Preserve the original number of local views.
        if len(new_locals) > n:
            new_locals = new_locals[:n]
        elif len(new_locals) < n:
            # Fill with unchanged local views if needed.
            for idx in indices:
                if len(new_locals) >= n:
                    break
                new_locals.append(local_views[idx])

        return new_locals

    def __call__(self, image: Union[Tensor, Image]) -> List[Tensor]:
        """Generate two global views and local views with LSA-Mix applied locally."""
        # Generate the two global views.
        global_view_0: Tensor = self.global_transform_0(image)
        global_view_1: Tensor = self.global_transform_1(image)

        # Generate the standard DINO local views before LSA-Mix.
        local_views: List[Tensor] = [
            self.local_transform(image) for _ in range(self.n_local_views)
        ]

        # Apply LSA-Mix to local views.
        local_views = self._apply_lsa_mix_to_locals(local_views)

        # Return two global views and n_local_views local views.
        return [global_view_0, global_view_1] + local_views


class DINOViewTransform:
    def __init__(
        self,
        crop_size: int = 224,
        crop_scale: Tuple[float, float] = (0.4, 1.0),
        hf_prob: float = 0.5,
        vf_prob: float = 0,
        rr_prob: float = 0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.4,
        cj_hue: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 1.0,
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        solarization_prob: float = 0.2,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        transform = [
            T.RandomResizedCrop(
                size=crop_size,
                scale=crop_scale,
                # Type ignore needed because BICUBIC is not recognized as an attribute.
                interpolation=PIL.Image.BICUBIC,  # type: ignore[attr-defined]
            ),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=cj_strength * cj_bright,
                        contrast=cj_strength * cj_contrast,
                        saturation=cj_strength * cj_sat,
                        hue=cj_strength * cj_hue,
                    )
                ],
                p=cj_prob,
            ),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(
                kernel_size=kernel_size,
                scale=kernel_scale,
                sigmas=sigmas,
                prob=gaussian_blur,
            ),
            RandomSolarization(prob=solarization_prob),
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]

        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """Applies the transforms to the input image."""
        transformed: Tensor = self.transform(image)
        return transformed


# Paper-facing alias.
LSAMixDINOTransform = DINOTransform
