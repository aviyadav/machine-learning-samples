"""Child process that packages and serves an AOT-compiled model.

Option 6 is the only mitigation that leaves the Python process, and it does so
for a specific reason: ``torch._inductor.aoti_load_package`` is a native
extension entry point, and on some platforms and PyTorch builds a failure there
takes the interpreter down with a segmentation fault rather than raising.  A
crash like that would abort the whole six-mitigation run and report nothing.

Running the packaging, loading and measurement here means the worst case is a
non-zero exit status that the parent can report as "not available in this
environment" -- which is true, and far more useful than a stack trace from the
middle of an unrelated experiment.

The contract is one JSON object on stdout:

``{"ok": true, "timings": [...], "package_bytes": N}``
``{"ok": false, "error": "..."}``
"""

from __future__ import annotations

import json
import sys

import torch

from recompilation_dilemma.config import DatasetConfig, ModelConfig, RunConfig
from recompilation_dilemma.dataset import make_dataset
from recompilation_dilemma.harness import cache_root
from recompilation_dilemma.model import build_model
from recompilation_dilemma.runtime import StepTimer, resolve_device
from recompilation_dilemma.solutions.aot_export import build_dynamic_shapes


def _config_from(payload: dict) -> RunConfig:
    return RunConfig(
        device=payload["device"],
        backend=payload["backend"],
        model=ModelConfig(**payload["model"]),
        dataset=DatasetConfig(**payload["dataset"]),
        cache_dir=payload["cache_dir"],
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(json.dumps({"ok": False, "error": "missing configuration payload"}))
        return 2

    try:
        config = _config_from(json.loads(argv[0]))
        device = resolve_device(config.device)
        data = make_dataset(
            device, config.dataset, feature_dim=config.model.feature_dim
        )
        model = build_model(config.model, seed=0).to(device).eval()

        exported = torch.export.export(
            model, tuple(data[0].values()), dynamic_shapes=build_dynamic_shapes(config)
        )

        package_path = cache_root(config) / "model_aot.pt2"
        package_path.parent.mkdir(parents=True, exist_ok=True)
        if package_path.exists():
            package_path.unlink()

        built = torch._inductor.aoti_compile_and_package(
            exported, package_path=str(package_path)
        )
        serving_model = torch._inductor.aoti_load_package(built)
    except Exception as error:  # noqa: BLE001 - the message is the finding
        first_line = str(error).splitlines()[0] if str(error) else ""
        print(
            json.dumps(
                {"ok": False, "error": f"{type(error).__name__}: {first_line}"[:400]}
            )
        )
        return 1

    timings: list[float] = []
    for sample in data:
        with StepTimer(device) as timer, torch.inference_mode():
            serving_model(*sample.values())
        timings.append(timer.elapsed_ms)

    print(
        json.dumps(
            {
                "ok": True,
                "timings": timings,
                "package_bytes": package_path.stat().st_size
                if package_path.exists()
                else 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
