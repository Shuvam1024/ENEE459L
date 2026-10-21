"""Assignment 2: arithmetic intensity (FLOPs per byte moved) per layer and per model.

Standalone. Everything this needs from graph.py and complexity.py is defined
below, so `python assignment2.py` runs on its own. With no argument it reports
on the toy ResNet from main.py; given a model description JSON it reports on
that instead:

    python assignment2.py
    python assignment2.py some_model.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Constants (graph.py, complexity.py)
# ---------------------------------------------------------------------------

DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}

FLOPS_PER_MAC = 2
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2
BUFFER_DTYPE = "fp32"


# ---------------------------------------------------------------------------
# Graph description (graph.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Layer:
    name: str
    kind: str
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    kernel: tuple[int, int] | None = None
    stride: int = 1
    groups: int = 1
    bias: bool = False
    weight_dtype: str = "fp32"
    act_dtype: str = "fp32"
    reads: tuple[str, ...] = ()

    def elements(self, shape: tuple[int, ...]) -> int:
        n = 1
        for d in shape:
            n *= d
        return n

    @property
    def in_elements(self) -> int:
        return self.elements(self.in_shape)

    @property
    def out_elements(self) -> int:
        return self.elements(self.out_shape)

    @property
    def out_spatial(self) -> int:
        if len(self.out_shape) == 3:
            return self.out_shape[1] * self.out_shape[2]
        return 1


@dataclass
class Graph:
    name: str
    input_shape: tuple[int, ...]
    layers: list[Layer]
    precision: str = "fp32"
    source: str = "unattributed"
    notes: str = ""
    _by_name: dict[str, Layer] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_name = {ly.name: ly for ly in self.layers}

    def __iter__(self) -> Iterable[Layer]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def get(self, name: str) -> Layer | None:
        return self._by_name.get(name)


def load_graph(path: str | Path) -> Graph:
    """Read one model description from JSON."""
    raw = json.loads(Path(path).read_text())
    layers = [
        Layer(
            name=d["name"],
            kind=d["kind"],
            in_shape=tuple(d["in_shape"]),
            out_shape=tuple(d["out_shape"]),
            kernel=tuple(d["kernel"]) if d.get("kernel") else None,
            stride=int(d.get("stride", 1)),
            groups=int(d.get("groups", 1)),
            bias=bool(d.get("bias", False)),
            weight_dtype=d.get("weight_dtype", raw.get("precision", "fp32")),
            act_dtype=d.get("act_dtype", raw.get("precision", "fp32")),
            reads=tuple(d.get("reads", ())),
        )
        for d in raw["layers"]
    ]
    return Graph(
        name=raw["name"],
        input_shape=tuple(raw["input_shape"]),
        layers=layers,
        precision=raw.get("precision", "fp32"),
        source=raw.get("source", "unattributed"),
        notes=raw.get("notes", ""),
    )


def computed(value: Any, source: str, **extra: Any) -> dict[str, Any]:
    return {"value": value, "source": source, "status": "computed", **extra}


def dtype_bytes(name: str) -> float:
    if name not in DTYPE_BYTES:
        raise KeyError(f"unknown dtype {name!r}; known: {sorted(DTYPE_BYTES)}")
    return DTYPE_BYTES[name]


# ---------------------------------------------------------------------------
# Per-layer counts (complexity.py)
# ---------------------------------------------------------------------------

def _layer_parameters(ly: Layer) -> int:
    """Learnable weights and bias for one layer. Batch-norm buffers are excluded."""
    if ly.kind == "conv":
        cout = ly.out_shape[0]
        cin = ly.in_shape[0]
        kh, kw = ly.kernel if ly.kernel is not None else (1, 1)
        nweights = cout * (cin // ly.groups) * kh * kw
        return nweights + (cout if ly.bias else 0)
    if ly.kind == "linear":
        fout = ly.out_shape[0]
        fin = ly.in_shape[0]
        return fout * fin + (fout if ly.bias else 0)
    if ly.kind == "bn":
        return BN_PARAMS_PER_CHANNEL * ly.out_shape[0]
    return 0


def _layer_macs(ly: Layer) -> int:
    """Multiply-accumulates for one layer.

    A convolution does one MAC per weight per output position; a linear layer
    one per weight. Batch norm, ReLU, pooling, add, and flatten are counted as
    zero MACs: they move memory rather than doing multiply-accumulate work.
    """
    if ly.kind == "conv":
        cout = ly.out_shape[0]
        cin = ly.in_shape[0]
        kh, kw = ly.kernel if ly.kernel is not None else (1, 1)
        return cout * (cin // ly.groups) * kh * kw * ly.out_spatial
    if ly.kind == "linear":
        return ly.out_shape[0] * ly.in_shape[0]
    return 0


# ---------------------------------------------------------------------------
# Assignment 2
# ---------------------------------------------------------------------------

def arithmetic_intensity(graph: Graph) -> dict[str, Any]:
    """FLOPs per byte moved, per layer and for the whole graph.

    Every tensor a layer touches is read from or written to DRAM once: its
    input, its weights and buffers, and its output. No cache reuse, no fusion.
    """
    layers = list(graph)
    per_layer: dict[str, dict[str, Any]] = {}
    total_flops = 0
    total_bytes = 0.0

    for i, ly in enumerate(layers):
        if ly.reads:
            producer = graph.get(ly.reads[0])
            in_dtype = producer.act_dtype if producer is not None else graph.precision
        elif i > 0:
            in_dtype = layers[i - 1].act_dtype
        else:
            in_dtype = graph.precision

        in_b = ly.in_elements * dtype_bytes(in_dtype)
        if ly.kind == "add" and len(ly.reads) == 2:
            in_b = 2 * ly.in_elements * dtype_bytes(in_dtype)

        w_b = _layer_parameters(ly) * dtype_bytes(ly.weight_dtype)
        if ly.kind == "bn":
            w_b += BN_BUFFERS_PER_CHANNEL * ly.out_shape[0] * dtype_bytes(BUFFER_DTYPE)

        out_b = ly.out_elements * dtype_bytes(ly.act_dtype)

        nbytes = float(in_b + w_b + out_b)
        flops = int(_layer_macs(ly) * FLOPS_PER_MAC)
        per_layer[ly.name] = {
            "flops": flops,
            "bytes": nbytes,
            "intensity": round(flops / nbytes, 4) if nbytes else 0.0,
        }
        total_flops += flops
        total_bytes += nbytes

    return computed(
        round(total_flops / total_bytes, 4) if total_bytes else 0.0,
        f"{graph.name}: FLOPs over bytes, every byte counted once",
        per_layer=per_layer,
        total_flops=total_flops,
        total_bytes=total_bytes,
        flops_per_mac=FLOPS_PER_MAC,
        cache_reuse_assumed=False,
        note="no fusion and no cache reuse; both would lower the byte column",
    )


# ---------------------------------------------------------------------------
# The toy ResNet from main.py
# ---------------------------------------------------------------------------

def build_synthetic_graph() -> Graph:
    layers = [
        Layer(name="conv1", kind="conv", in_shape=(3, 32, 32), out_shape=(16, 32, 32),
              kernel=(3, 3), bias=False, weight_dtype="fp32", act_dtype="fp32"),
        Layer(name="bn1", kind="bn", in_shape=(16, 32, 32), out_shape=(16, 32, 32),
              weight_dtype="fp32", act_dtype="fp32"),
        Layer(name="relu1", kind="relu", in_shape=(16, 32, 32), out_shape=(16, 32, 32)),
        Layer(name="res_conv", kind="conv", in_shape=(16, 32, 32), out_shape=(16, 32, 32),
              kernel=(3, 3), bias=False, reads=("relu1",)),
        Layer(name="add1", kind="add", in_shape=(16, 32, 32), out_shape=(16, 32, 32),
              reads=("relu1", "res_conv")),
        Layer(name="pool", kind="pool", in_shape=(16, 32, 32), out_shape=(16, 1, 1)),
        Layer(name="flatten", kind="flatten", in_shape=(16, 1, 1), out_shape=(16,)),
        Layer(name="fc", kind="linear", in_shape=(16,), out_shape=(10,), bias=True,
              weight_dtype="fp32", act_dtype="fp32"),
    ]
    return Graph(name="toy_resnet", input_shape=(3, 32, 32), layers=layers,
                 precision="fp32", source="synthetic_test")


def main() -> None:
    graph = load_graph(sys.argv[1]) if len(sys.argv) > 1 else build_synthetic_graph()
    result = arithmetic_intensity(graph)

    print(f"{'layer':<10} {'flops':>12} {'bytes':>12} {'FLOPs/byte':>11}")
    for name, row in result["per_layer"].items():
        print(f"{name:<10} {row['flops']:>12,} {row['bytes']:>12,.0f} {row['intensity']:>11.4f}")
    print(f"{'total':<10} {result['total_flops']:>12,} {result['total_bytes']:>12,.0f} "
          f"{result['value']:>11.4f}")

    out = Path("arithmetic_intensity_results.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
