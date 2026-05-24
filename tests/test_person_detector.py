import person_detector


def test_load_yolo_model_uses_yolo26_default(monkeypatch):
    loaded = []

    monkeypatch.setattr(person_detector, "_YOLO_AVAILABLE", True)
    monkeypatch.setattr(person_detector, "YOLO", lambda model_name: loaded.append(model_name) or object(), raising=False)
    monkeypatch.delenv("YOLO_MODEL_NAME", raising=False)

    assert person_detector.load_yolo_model() is not None
    assert loaded == ["yolo26l-seg.pt"]


def test_load_yolo_model_uses_env_override(monkeypatch):
    loaded = []

    monkeypatch.setattr(person_detector, "_YOLO_AVAILABLE", True)
    monkeypatch.setattr(person_detector, "YOLO", lambda model_name: loaded.append(model_name) or object(), raising=False)
    monkeypatch.setenv("YOLO_MODEL_NAME", "yolo26l-seg.pt")

    assert person_detector.load_yolo_model() is not None
    assert loaded == ["yolo26l-seg.pt"]
