import json

import numpy as np

from anomaly.model import LSTMAutoencoder
from core.config import settings


def main() -> None:
    model = LSTMAutoencoder.load(settings.anomaly_model_path, device="cpu")
    with np.load(settings.feature_scaler_path) as scaler:
        mean = np.asarray(scaler["mean"])
        std = np.asarray(scaler["std"])

    threshold_data = json.loads(
        settings.anomaly_threshold_path.read_text(encoding="utf-8")
    )
    threshold = float(threshold_data["threshold"])

    assert mean.shape == (model.input_size,)
    assert std.shape == (model.input_size,)
    print(
        f"Model OK: input={model.input_size}, hidden={model.hidden_size}, "
        f"layers={model.num_layers}"
    )
    print(f"Scaler OK: {mean.shape[0]} features")
    print(f"Threshold OK: {threshold:.6f}")


if __name__ == "__main__":
    main()
