"""MLX availability probe for scripts/bench-baseline.sh.

A comparison row counts only for the identical artifact (benchmarks/README.md,
four-way identity). This probe asks the installed mlx-lm to load the exact
GGUF file the other engines ran, offline, and records what happened. It never
downloads or converts a model: an MLX conversion is a different artifact.
Prints one JSON object on stdout.
"""

import json
import sys


def main() -> int:
    path = sys.argv[1]
    out = {"available": False, "artifact": path}
    try:
        import mlx.core as mx
        import mlx_lm
    except Exception as exc:  # noqa: BLE001 - recorded verbatim
        out["reason"] = f"mlx-lm import failed: {exc}"
        print(json.dumps(out))
        return 0
    out["mlx"] = mx.__version__
    out["mlx_lm"] = getattr(mlx_lm, "__version__", "unknown")
    try:
        mlx_lm.load(path)
    except Exception as exc:  # noqa: BLE001 - recorded verbatim
        out["reason"] = (
            "mlx-lm cannot load the identical GGUF artifact "
            f"({type(exc).__name__}: {str(exc).splitlines()[0][:200]}); "
            "an MLX-format conversion would be a different artifact, so no MLX row is recorded"
        )
        print(json.dumps(out))
        return 0
    out["available"] = True
    out["reason"] = "mlx-lm loaded the identical artifact; add an MLX bench row"
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
