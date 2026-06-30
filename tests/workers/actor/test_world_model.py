import importlib.util
from pathlib import Path

import torch


def _load_world_model_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "verl" / "workers" / "actor" / "world_model.py"
    spec = importlib.util.spec_from_file_location("world_model_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_residual_predictor_starts_as_identity():
    module = _load_world_model_module()
    predictor = module.LatentTransitionPredictor(hidden_size=8, bottleneck_size=4, residual=True)

    hidden_states = torch.randn(3, 8)

    torch.testing.assert_close(predictor(hidden_states), hidden_states)


def test_non_residual_predictor_preserves_shape_and_gradients():
    module = _load_world_model_module()
    predictor = module.LatentTransitionPredictor(hidden_size=8, bottleneck_size=4, residual=False)
    hidden_states = torch.randn(3, 8, requires_grad=True)

    output = predictor(hidden_states)
    output.sum().backward()

    assert output.shape == hidden_states.shape
    assert hidden_states.grad is not None
