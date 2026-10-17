"""
tests/test_ai_service.py
────────────────────────
Unit tests for CrisisAIService — testing the AI wrapper in isolation,
without loading the actual model.

Why test the service separately from the endpoints?
  The endpoint tests (test_events.py) mock out ai_engine.predict() entirely.
  These tests verify the service's own logic — label mapping, fallback
  behaviour, model-not-loaded guard — using lightweight mocks of the
  PyTorch model itself.

monkeypatch vs unittest.mock:
  Both work. monkeypatch (pytest built-in) is used here because it
  automatically restores the original value after each test, with no
  teardown code required.
"""
import pytest
import torch
import torch.nn as nn

from app.ai_core.service import CrisisAIService, HUMANITARIAN_LABELS, SEVERITY_LABELS, INFO_LABELS


class TestLabelMaps:
    """
    The label lists must have the correct length — if they are out of sync
    with the model's output classes the wrong string will be returned silently.
    """
    def test_info_labels_length(self):
        assert len(INFO_LABELS) == 2, "Task 1 has 2 classes"

    def test_humanitarian_labels_length(self):
        assert len(HUMANITARIAN_LABELS) == 6, "Task 2 has 6 classes"

    def test_severity_labels_length(self):
        assert len(SEVERITY_LABELS) == 3, "Task 3 has 3 classes"

    def test_info_labels_contain_informative(self):
        assert "informative"     in INFO_LABELS
        assert "not informative" in INFO_LABELS

    def test_severity_labels_ordered(self):
        """Class 2 should be 'severe damage' — the most important for crisis response."""
        assert SEVERITY_LABELS[2] == "severe damage"


class TestPredictFallback:

    def test_predict_returns_fallback_when_model_not_loaded(self):
        """
        If load_model() was never called (model is None), predict() should
        return the safe fallback dict rather than raising an AttributeError.
        """
        service = CrisisAIService()
        # model is None by default — no load_model() call
        result = service.predict("some text", "nonexistent.jpg")

        assert "is_informative" in result
        assert "humanitarian"   in result
        assert "severity"       in result
        assert result["is_informative"] is False   # fallback is non-informative

    def test_predict_returns_fallback_on_bad_image_path(self, monkeypatch):
        """
        A valid model but an invalid image path should not crash — it should
        return the fallback dict.
        """
        service = CrisisAIService()

        # Give it a dummy model so the "model is None" guard passes
        monkeypatch.setattr(service, "model", nn.Linear(1, 1))

        result = service.predict("test text", "/this/path/does/not/exist.jpg")

        assert result["is_informative"] is False


class TestPredictOutputMapping:

    def _make_mock_model(self, task1_idx=1, task2_idx=0, task3_idx=2):
        """
        Returns a callable that mimics the model's forward pass,
        returning fixed argmax indices without any real computation.
        """
        class MockModel:
            def __call__(self, image, text):
                # Build logits where the desired index has the highest value
                def one_hot(idx, n):
                    t = torch.zeros(1, n)
                    t[0, idx] = 10.0
                    return t
                return {
                    "task1": one_hot(task1_idx, 2),
                    "task2": one_hot(task2_idx, 6),
                    "task3": one_hot(task3_idx, 3),
                }

            def eval(self):
                return self

        return MockModel()

    def test_informative_index_1_maps_to_informative(self, monkeypatch, tmp_path):
        """task1 argmax=1 → is_informative=True"""
        service  = CrisisAIService()
        img_path = tmp_path / "test.jpg"

        # Create a tiny real image so PIL.Image.open succeeds
        from PIL import Image as PILImage
        PILImage.new("RGB", (32, 32)).save(str(img_path))

        from transformers import CLIPProcessor
        monkeypatch.setattr(service, "model",     self._make_mock_model(task1_idx=1))
        monkeypatch.setattr(service, "processor", CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        ))

        result = service.predict("bridge collapsed", str(img_path))
        assert result["is_informative"] is True

    def test_non_informative_index_0_maps_to_false(self, monkeypatch, tmp_path):
        """task1 argmax=0 → is_informative=False"""
        service  = CrisisAIService()
        img_path = tmp_path / "test.jpg"

        from PIL import Image as PILImage
        PILImage.new("RGB", (32, 32)).save(str(img_path))

        from transformers import CLIPProcessor
        monkeypatch.setattr(service, "model",     self._make_mock_model(task1_idx=0))
        monkeypatch.setattr(service, "processor", CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        ))

        result = service.predict("nothing relevant", str(img_path))
        assert result["is_informative"] is False

    def test_severity_index_2_maps_to_severe_damage(self, monkeypatch, tmp_path):
        """task3 argmax=2 → 'severe damage'"""
        service  = CrisisAIService()
        img_path = tmp_path / "test.jpg"

        from PIL import Image as PILImage
        PILImage.new("RGB", (32, 32)).save(str(img_path))

        from transformers import CLIPProcessor
        monkeypatch.setattr(service, "model",     self._make_mock_model(task3_idx=2))
        monkeypatch.setattr(service, "processor", CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        ))

        result = service.predict("destroyed buildings", str(img_path))
        assert result["severity"] == "severe damage"