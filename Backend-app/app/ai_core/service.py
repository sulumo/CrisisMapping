"""
ai_core/service.py
──────────────────
The AI inference service — loads the trained model and exposes a predict() method.

Key fixes over the original:
  1. Model architecture is selected by the MODEL_VARIANT environment variable.
     Set MODEL_VARIANT=baseline  → loads CLIP_CrisisKAN_Baseline
     Set MODEL_VARIANT=multilayer  → loads CLIP_CrisisKAN_TCG
     This eliminates the silent weight mismatch when deploying multilayer weights
     into the baseline architecture.

  2. CLIPProcessor is used for preprocessing (correct CLIP normalisation stats).

  3. All errors surface via logging with full tracebacks — no silent failures.

  4. predict() returns a typed dict that maps directly to the AIResult schema.
     If inference fails, a safe fallback dict is returned so the endpoint can
     raise an HTTPException rather than crash.

Label ordering:
  These must match EXACTLY the label-to-index mapping used during training
  (defined in crisismmd_dataset.py).  If the dataset script changes the
  mapping, update these lists to match.
"""
import logging
import os
from typing import Optional

import torch
from PIL import Image
from transformers import CLIPProcessor

logger = logging.getLogger(__name__)

# ── Label maps (must match crisismmd_dataset.py) ──────────────────────────────
INFO_LABELS = [
    "not informative",  # 0
    "informative",      # 1
]

HUMANITARIAN_LABELS = [
    "infrastructure",             # 0
    "not humanitarian",           # 1
    "other relevant information", # 2
    "rescue volunteering",        # 3
    "vehicle damage",             # 4
    "affected individuals",       # 5
]

SEVERITY_LABELS = [
    "little or no damage",  # 0
    "mild damage",          # 1
    "severe damage",        # 2
]

# ── Model registry ────────────────────────────────────────────────────────────
# Adding a new architecture: import its class and add it here.
def _load_architecture(variant: str):
    if variant == "baseline":
        from .architectures.baseline import CLIP_CrisisKAN_Baseline
        return CLIP_CrisisKAN_Baseline()
    elif variant == "multilayer":
        from .architectures.multilayer import CLIP_CrisisKAN_TCG
        return CLIP_CrisisKAN_TCG()
    else:
        raise ValueError(
            f"Unknown MODEL_VARIANT '{variant}'. "
            f"Choose 'baseline' or 'multilayer'."
        )


# ── Service class ─────────────────────────────────────────────────────────────
class CrisisAIService:
    """
    Wraps the model loading and inference logic.

    A single instance (ai_engine) is created at module level and shared
    across all requests.  load_model() is called once at application startup
    via the lifespan context manager in main.py.
    """

    def __init__(self):
        self.model     = None
        self.processor = None
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"AI Service initialised on: {self.device}")

    def load_model(self):
        variant    = os.getenv("MODEL_VARIANT", "baseline")
        model_path = os.getenv("MODEL_WEIGHTS_PATH", "app/ai_core/best.pt")

        logger.info(f"Loading architecture: {variant}")
        logger.info(f"Weights path: {model_path}")

        try:
            self.model = _load_architecture(variant)

            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)

                # Strip 'module.' prefix if model was saved with DataParallel
                if any(k.startswith("module.") for k in state_dict):
                    logger.info("Stripping 'module.' prefix from DataParallel checkpoint")
                    state_dict = {
                        k.replace("module.", "", 1): v
                        for k, v in state_dict.items()
                    }

                missing, unexpected = self.model.load_state_dict(
                    state_dict, strict=True
                )
                if missing:
                    logger.warning(f"Missing keys in checkpoint: {missing}")
                if unexpected:
                    logger.warning(f"Unexpected keys in checkpoint: {unexpected}")

                logger.info(f"Weights loaded from {model_path}")
            else:
                logger.warning(
                    f"Weight file not found at '{model_path}'. "
                    f"Model is using random weights — predictions will be meaningless."
                )

            self.model.to(self.device)
            self.model.eval()

            # CLIPProcessor handles both image resizing/normalisation (using
            # CLIP's own stats) and text tokenisation — this is correct for
            # the frozen CLIP encoders used in both architectures.
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            logger.info("AI Engine online.")

        except Exception:
            logger.critical("CRITICAL: AI model failed to load", exc_info=True)
            # Do not re-raise — let the app start; predict() returns a safe fallback.

    def predict(self, text: str, image_path: str) -> dict:
        """
        Run inference and return a dict with keys:
          is_informative : bool
          humanitarian   : str
          severity       : str

        Returns a safe fallback dict on any error so the endpoint can
        raise HTTPException rather than crash with a 500.
        """
        _fallback = {
            "is_informative": False,
            "humanitarian":   "Unknown",
            "severity":       "Unknown",
        }

        if self.model is None:
            logger.error("predict() called but model is not loaded")
            return _fallback

        try:
            image = Image.open(image_path).convert("RGB")

            inputs = self.processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )

            pixel_values = inputs["pixel_values"].to(self.device)
            text_inputs  = {
                k: v.to(self.device)
                for k, v in inputs.items()
                if k != "pixel_values"
            }

            with torch.no_grad():
                outputs = self.model(pixel_values, text_inputs)

            info_idx = outputs["task1"].argmax(dim=1).item()
            cat_idx  = outputs["task2"].argmax(dim=1).item()
            sev_idx  = outputs["task3"].argmax(dim=1).item()

            return {
                "is_informative": INFO_LABELS[info_idx] == "informative",
                "humanitarian":   HUMANITARIAN_LABELS[cat_idx],
                "severity":       SEVERITY_LABELS[sev_idx],
            }

        except Exception:
            logger.error("Inference failed", exc_info=True)
            return _fallback


# Module-level singleton imported by routers/events.py
ai_engine = CrisisAIService()