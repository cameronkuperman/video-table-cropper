"""SAM 3 adapter with a stable derived-region-vector contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


class Sam3AdapterError(RuntimeError):
    """Raised when SAM 3 cannot be loaded or invoked."""


@dataclass
class Sam3Detection:
    label: str
    mask: np.ndarray
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    vector: np.ndarray
    vector_source: str


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return (vector / norm).astype(np.float32)


def _resize_vector(vector: np.ndarray, target_dim: int = 256) -> np.ndarray:
    flat = vector.astype(np.float32).reshape(-1)
    if flat.size == target_dim:
        return _l2_normalize(flat)
    if flat.size == 0:
        return np.zeros((target_dim,), dtype=np.float32)
    old_positions = np.linspace(0.0, 1.0, num=flat.size, dtype=np.float32)
    new_positions = np.linspace(0.0, 1.0, num=target_dim, dtype=np.float32)
    resized = np.interp(new_positions, old_positions, flat).astype(np.float32)
    return _l2_normalize(resized)


class Sam3Adapter:
    """Thin optional wrapper around official SAM 3 with stable output parsing."""

    def __init__(self, checkpoint_path: str | None = None, config_name: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.config_name = config_name
        self._torch = None
        self.device = "cpu"
        self.model = None
        self.processor = None
        self.backend = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise Sam3AdapterError(
                "PyTorch is required for SAM 3 processing. Install the GPU worker dependencies first."
            ) from exc

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        load_errors: list[str] = []
        loader_candidates = [
            self._load_from_transformers,
            self._load_from_sam3_package,
        ]
        for candidate in loader_candidates:
            try:
                model, processor, backend = candidate()
                self.model = model
                self.processor = processor
                self.backend = backend
                return
            except Exception as exc:  # pragma: no cover - runtime dependent
                load_errors.append(str(exc))

        raise Sam3AdapterError("Unable to load an official SAM 3 image backend: " + " | ".join(load_errors))

    def _load_from_sam3_package(self) -> tuple[Any, Any, str]:
        from sam3.model_builder import build_sam3_image_model  # type: ignore
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore

        if self.checkpoint_path and self.config_name:
            model = build_sam3_image_model(self.config_name, self.checkpoint_path)
        elif self.checkpoint_path:
            model = build_sam3_image_model(checkpoint=self.checkpoint_path)
        elif self.config_name:
            model = build_sam3_image_model(self.config_name)
        else:
            # Official README shows the default no-arg path.
            model = build_sam3_image_model()
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        processor = Sam3Processor(model)
        return model, processor, "sam3_package"

    def _load_from_transformers(self) -> tuple[Any, Any, str]:
        from transformers import Sam3Model, Sam3Processor  # type: ignore

        model_name = self.config_name or "facebook/sam3"
        model = Sam3Model.from_pretrained(model_name)
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()
        processor = Sam3Processor.from_pretrained(model_name)
        return model, processor, "transformers"

    def _maybe_set_image(self, image: Image.Image) -> Any:
        if hasattr(self.processor, "set_image"):
            return self.processor.set_image(image.convert("RGB"))
        return None

    def _invoke_text_prompt(self, image: Image.Image, prompt: str, state: Any) -> Any:
        if hasattr(self.processor, "set_text_prompt"):
            kwargs = {"prompt": prompt}
            if state is not None:
                kwargs["state"] = state
            return self.processor.set_text_prompt(**kwargs)

        if callable(self.processor):
            return self.processor(images=image, text=prompt, return_tensors="pt")

        raise Sam3AdapterError("Loaded SAM 3 processor does not expose a supported text-prompt API.")

    @staticmethod
    def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return None

    def _parse_output(self, output: Any) -> tuple[list[np.ndarray], list[tuple[float, float, float, float]], list[float]]:
        if isinstance(output, dict):
            masks = self._first_present(output, "masks", "pred_masks", "mask")
            boxes = self._first_present(output, "boxes", "bboxes", "bbox")
            scores = self._first_present(output, "scores", "iou_scores", "score")
        else:
            masks = getattr(output, "masks", None)
            boxes = getattr(output, "boxes", None)
            scores = getattr(output, "scores", None)

        if masks is None:
            raise Sam3AdapterError("SAM 3 output did not include masks.")

        masks_np = self._to_numpy(masks)
        boxes_np = self._to_numpy(boxes) if boxes is not None else None
        scores_np = self._to_numpy(scores) if scores is not None else None

        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]
        mask_list = [mask.astype(bool) for mask in masks_np]

        bbox_list: list[tuple[float, float, float, float]] = []
        for idx, mask in enumerate(mask_list):
            if boxes_np is not None and np.array(boxes_np).ndim >= 2 and idx < len(boxes_np):
                box = np.array(boxes_np[idx]).astype(np.float32).reshape(-1)
                if box.size >= 4:
                    bbox_list.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
                    continue
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                bbox_list.append((0.0, 0.0, 0.0, 0.0))
            else:
                bbox_list.append((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))

        score_list: list[float] = []
        for idx in range(len(mask_list)):
            if scores_np is not None and np.array(scores_np).size > idx:
                score_list.append(float(np.array(scores_np).reshape(-1)[idx]))
            else:
                score_list.append(1.0)

        return mask_list, bbox_list, score_list

    def _to_numpy(self, value: Any) -> np.ndarray:
        if value is None:
            return np.array([])
        if isinstance(value, np.ndarray):
            return value
        if self._torch is not None and isinstance(value, self._torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)):
            return np.array(value)
        return np.array(value)

    def _find_feature_tensor(self, value: Any) -> Any:
        if value is None:
            return None
        if self._torch is not None and isinstance(value, self._torch.Tensor):
            if value.ndim in {3, 4}:
                return value
            return None
        if isinstance(value, dict):
            for child in value.values():
                found = self._find_feature_tensor(child)
                if found is not None:
                    return found
        if isinstance(value, (list, tuple)):
            for child in value:
                found = self._find_feature_tensor(child)
                if found is not None:
                    return found
        return None

    def _normalize_target_sizes(self, value: Any, image: Image.Image) -> list[tuple[int, int]]:
        array = self._to_numpy(value)
        if array.size == 0:
            return [(int(image.height), int(image.width))]
        if array.ndim == 1:
            flat = array.reshape(-1)
            if flat.size < 2:
                return [(int(image.height), int(image.width))]
            return [(int(flat[0]), int(flat[1]))]
        return [(int(size[0]), int(size[1])) for size in array.tolist()]

    def _scale_boxes_to_target_sizes(self, boxes: Any, target_sizes: list[tuple[int, int]]) -> Any:
        torch = self._torch
        if torch is None:
            return boxes
        if not target_sizes:
            return boxes

        image_height = torch.tensor([size[0] for size in target_sizes], dtype=boxes.dtype, device=boxes.device)
        image_width = torch.tensor([size[1] for size in target_sizes], dtype=boxes.dtype, device=boxes.device)
        scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
        return boxes * scale_factor.unsqueeze(1)

    def _post_process_transformers_outputs(
        self,
        outputs: Any,
        *,
        target_sizes: list[tuple[int, int]],
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        torch = self._torch
        if torch is None:
            raise Sam3AdapterError("PyTorch is required for SAM 3 post-processing.")

        pred_logits = outputs.pred_logits
        pred_boxes = outputs.pred_boxes
        pred_masks = outputs.pred_masks
        presence_logits = getattr(outputs, "presence_logits", None)

        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            batch_scores = batch_scores * presence_logits.sigmoid()

        batch_boxes = self._scale_boxes_to_target_sizes(pred_boxes, target_sizes)
        batch_masks = pred_masks.sigmoid()

        results: list[dict[str, Any]] = []
        for idx, (scores, boxes, masks) in enumerate(zip(batch_scores, batch_boxes, batch_masks)):
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]
            masks = masks[keep]

            if len(masks) > 0:
                target_height, target_width = target_sizes[idx]
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(0),
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            masks = (masks > mask_threshold).to(torch.long)
            results.append({"scores": scores, "boxes": boxes, "masks": masks})

        return results

    def _vector_from_feature_map(self, feature_tensor: Any, mask: np.ndarray) -> np.ndarray:
        torch = self._torch
        if torch is None:
            return np.zeros((256,), dtype=np.float32)

        tensor = feature_tensor.detach().float().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3:
            return np.zeros((256,), dtype=np.float32)

        if tensor.shape[0] <= 4 and tensor.shape[-1] > tensor.shape[0]:
            tensor = tensor.permute(2, 0, 1)

        channels, height, width = tensor.shape
        mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None, :, :]
        resized_mask = torch.nn.functional.interpolate(mask_tensor, size=(height, width), mode="nearest")[0, 0]
        weight_sum = float(resized_mask.sum())
        if weight_sum <= 1e-8:
            return np.zeros((256,), dtype=np.float32)
        pooled = (tensor * resized_mask).view(channels, -1).sum(dim=1) / weight_sum
        return _resize_vector(pooled.numpy(), 256)

    def _vector_from_masked_rgb(self, image: Image.Image, mask: np.ndarray) -> np.ndarray:
        image_np = np.array(image.convert("RGB"))
        if not mask.any():
            return np.zeros((256,), dtype=np.float32)
        ys, xs = np.nonzero(mask)
        crop = image_np[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        crop_mask = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        crop = crop.copy()
        crop[~crop_mask] = 0
        resized = Image.fromarray(crop).convert("L").resize((16, 16))
        return _resize_vector(np.array(resized, dtype=np.float32).reshape(-1), 256)

    def derive_region_vector(self, image: Image.Image, mask: np.ndarray) -> np.ndarray:
        return self._vector_from_masked_rgb(image, mask)

    def detect_objects(self, image: Image.Image, prompts: tuple[str, ...] = ("table", "person")) -> dict[str, list[Sam3Detection]]:
        if self.backend == "transformers":
            return self._detect_with_transformers(image, prompts)

        image = image.convert("RGB")
        state = self._maybe_set_image(image)
        feature_tensor = self._find_feature_tensor(state)
        results: dict[str, list[Sam3Detection]] = {}

        with self._torch.no_grad() if self._torch is not None else _nullcontext():
            for prompt in prompts:
                output = self._invoke_text_prompt(image, prompt, state)
                masks, boxes, scores = self._parse_output(output)
                detections: list[Sam3Detection] = []
                for mask, bbox, score in zip(masks, boxes, scores):
                    if feature_tensor is not None:
                        vector = self._vector_from_feature_map(feature_tensor, mask)
                        vector_source = "sam_feature_map"
                    else:
                        vector = self._vector_from_masked_rgb(image, mask)
                        vector_source = "masked_rgb_fallback"
                    detections.append(
                        Sam3Detection(
                            label=prompt,
                            mask=mask.astype(bool),
                            bbox_xyxy=bbox,
                            score=float(score),
                            vector=vector,
                            vector_source=vector_source,
                        )
                    )
                results[prompt] = detections
        return results

    def _detect_with_transformers(
        self,
        image: Image.Image,
        prompts: tuple[str, ...],
    ) -> dict[str, list[Sam3Detection]]:
        torch = self._torch
        if torch is None or self.model is None or self.processor is None:
            raise Sam3AdapterError("Transformers SAM 3 backend is not initialized.")

        image = image.convert("RGB")
        results: dict[str, list[Sam3Detection]] = {}
        for prompt in prompts:
            model_inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            target_sizes = self._normalize_target_sizes(model_inputs.get("original_sizes"), image)

            if hasattr(model_inputs, "items"):
                model_inputs = {
                    key: value.to(self.device) if hasattr(value, "to") and callable(value.to) else value
                    for key, value in model_inputs.items()
                    if key != "original_sizes"
                }

            with torch.no_grad():
                outputs = self.model(**model_inputs)

            processed = self._post_process_transformers_outputs(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=target_sizes,
            )[0]

            masks, boxes, scores = self._parse_output(processed)
            detections: list[Sam3Detection] = []
            feature_tensor = self._find_feature_tensor(outputs)
            for mask, bbox, score in zip(masks, boxes, scores):
                if feature_tensor is not None:
                    vector = self._vector_from_feature_map(feature_tensor, mask)
                    vector_source = "sam_model_output"
                else:
                    vector = self._vector_from_masked_rgb(image, mask)
                    vector_source = "masked_rgb_fallback"
                detections.append(
                    Sam3Detection(
                        label=prompt,
                        mask=mask.astype(bool),
                        bbox_xyxy=bbox,
                        score=float(score),
                        vector=vector,
                        vector_source=vector_source,
                    )
                )
            results[prompt] = detections
        return results


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False
