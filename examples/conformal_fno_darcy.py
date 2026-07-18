"""
End-to-end: FNO on PDEForge Darcy data + split-conformal calibration.

Reproduces the UQNO-style workflow (Ma et al. 2024, arXiv:2402.01960) WITHOUT
hand-slicing a calibration set out of the training tensors — PDEForge's
four-way split provides it natively.

Requires the optional deps:
    pip install pdeforge[torch] neuraloperator

Run:
    python examples/conformal_fno_darcy.py
"""

import numpy as np

import pdeforge
from pdeforge.uq import conformal_quantile, empirical_coverage


def main(n_samples=800, resolution=64, epochs=8, alpha=0.1):
    import torch
    from neuralop.models import FNO

    # 1. Generate Darcy data at the resolution YOU want — one call.
    data = pdeforge.generate_dataset(
        "darcy_2d",
        n_samples=n_samples,
        resolution={"x": resolution, "y": resolution},
        seed=0,
    )
    splits = data.split(train=0.6, val=0.15, cal=0.15, test=0.1, seed=1)

    def tensors(ds):
        x = torch.as_tensor(np.asarray(ds.inputs), dtype=torch.float32)[:, None]
        y = torch.as_tensor(np.asarray(ds.outputs), dtype=torch.float32)[:, None]
        return x, y

    Xtr, Ytr = tensors(splits["train"])
    Xva, Yva = tensors(splits["val"])

    # 2. Train a small FNO surrogate (CPU-friendly settings).
    model = FNO(n_modes=(12, 12), hidden_channels=32, in_channels=1, out_channels=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i : i + 32]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva), Yva).item()
        print(f"epoch {epoch + 1}/{epochs}  val MSE {vloss:.3e}")

    # 3. Split-conformal calibration on the DEDICATED cal split.
    with torch.no_grad():
        Xca, Yca = tensors(splits["cal"])
        cal_pred = model(Xca).numpy()[:, 0]
        Xte, Yte = tensors(splits["test"])
        test_pred = model(Xte).numpy()[:, 0]

    qhat = conformal_quantile(
        cal_pred, np.asarray(splits["cal"].outputs), alpha=alpha, score="max"
    )
    cov = empirical_coverage(
        test_pred, np.asarray(splits["test"].outputs), qhat, score="max"
    )

    print(f"\nalpha = {alpha}: band half-width qhat = {qhat:.4f}")
    print(f"empirical test coverage = {cov:.3f}  (target >= {1 - alpha})")


if __name__ == "__main__":
    main()
