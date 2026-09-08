from __future__ import annotations

import os

# PyTorch/CUDA must see this before cuBLAS is initialized.
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

from agnam.benchmarking.realx_runner import (
    run_realx_integration_sanity,
)


def main():
    print(
        "CUBLAS_WORKSPACE_CONFIG:",
        os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
    )

    run_realx_integration_sanity(
        verbose=True,
    )


if __name__ == "__main__":
    main()