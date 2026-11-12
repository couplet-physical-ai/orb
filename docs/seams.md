# Simulator seams

A seam is a small adapter from Orb's simulator-neutral world graph to one simulator or interchange format. The built-in seams generate MuJoCo XML or binary glTF. Any executable named `orb-seam-<name>` can implement a new adapter without a dependency on the Orb source tree.

## External seam contract

Orb invokes a configured executable with `--out <directory>` and sends the normalized world graph as one JSON document on standard input. The adapter creates its artifacts beneath the requested output directory and prints the written paths, one per line, to standard output. A non-zero exit is an apply failure; diagnostic text belongs on standard error.

For example, a project may declare:

```hcl
seam "simulator" {
  command = "orb-seam-simulator --format json"
}
```

Then run `uv run orb apply PROJECT --seam simulator`. Alternatively, put `orb-seam-simulator` on `PATH` and omit the command setting. Avoid shell interpolation of untrusted project values; pass arguments as an argument vector and treat the input graph as data.

## World graph

The JSON graph is versioned (`orb` currently emits `0.1`) and uses SI units. It contains normalized `parts`, `assemblies`, `sites`, `items`, `lifecycles`, `processes`, `timelines`, `cameras`, `seams` and extension blocks. Quaternions use `[w, x, y, z]`. Adapters should reject an unsupported graph version and ignore extension fields they do not consume.

A focused adapter can read only the graph sections it supports. `only` and `skip` attributes scope objects to named seams; a `native "NAME" { ... }` block and `*.NAME.orb` addendum can carry simulator-specific data without changing the common physical model. See [`examples/robot-cell`](../examples/robot-cell) for a synthetic project.
