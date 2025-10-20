import os
import torch
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceImage
import numpy as np
import hashlib
import logging
from tqdm import tqdm  # <-- thêm dòng này

logger = logging.getLogger(__name__)

# Thư mục lưu cache
CACHE_DIR = "cache_tda"
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache trong RAM
_tda_cache = {}

def _hash_tensor(tensor):
    """Tạo hash duy nhất cho tensor"""
    tensor_bytes = tensor.detach().cpu().numpy().tobytes()
    return hashlib.md5(tensor_bytes).hexdigest()

def compute_tda_features(attentions, homology_dimensions=[0, 1], image_dim=50):
    """
    Tính đặc trưng TDA cho attention maps.
    Hiển thị tiến độ bằng tqdm, tự động cache vào file và RAM.
    """
    try:
        vr = VietorisRipsPersistence(homology_dimensions=homology_dimensions, metric="precomputed")
        pi = PersistenceImage(sigma=0.1, n_bins=image_dim)

        features = []
        for att in tqdm(attentions, desc="Computing TDA features", ncols=100):
            att = att.detach().cpu()
            key = _hash_tensor(att)
            cache_path = os.path.join(CACHE_DIR, f"{key}.npy")

            # Ưu tiên cache RAM
            if key in _tda_cache:
                features.append(_tda_cache[key])
                continue

            # Load từ file cache
            if os.path.exists(cache_path):
                arr = np.load(cache_path)
                tda_tensor = torch.tensor(arr, dtype=torch.float)
                _tda_cache[key] = tda_tensor
                features.append(tda_tensor)
                continue

            # Tính mới nếu chưa có cache
            att_np = att.numpy()
            dist_matrix = np.clip(1 - att_np, 0, 1)
            diagrams = vr.fit_transform([dist_matrix])[0]
            pi_features = pi.fit_transform([diagrams])[0].flatten()

            np.save(cache_path, pi_features)  # lưu ra file
            tda_tensor = torch.tensor(pi_features, dtype=torch.float)
            _tda_cache[key] = tda_tensor
            features.append(tda_tensor)

        return torch.stack(features)

    except Exception as e:
        logger.error(f"TDA error: {str(e)}")
        raise RuntimeError("Error computing TDA features") from e
# ==============================