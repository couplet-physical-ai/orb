# orb

**OrbLang** is a vendor-neutral language for describing physical systems as code. It compiles parts, assemblies, sites, motion and lifecycle rules into a simulator-neutral world graph that built-in and external adapters can consume.

![Status: experimental](https://img.shields.io/badge/status-experimental-orange) ![License: Apache--2.0](https://img.shields.io/badge/license-Apache--2.0-blue)

Created at [Exowatt](https://exowatt.com). This project is maintained in the independent, vendor-neutral [Couplet](https://github.com/couplet-physical-ai) organization.

## Status

Experimental. APIs and file formats may change; there is no production support commitment.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/couplet-physical-ai/orb.git
cd orb
uv sync
```

The compiler and MuJoCo XML exporter use only the Python standard library. Install the optional `glb` extra to write binary glTF files:

```sh
uv sync --extra glb
```

## Usage / Quickstart

Validate and compile one of the fully synthetic examples, then write a MuJoCo XML scene:

```sh
uv run orb validate examples/robot-cell
uv run orb ir examples/robot-cell
uv run orb apply examples/robot-cell --seam mujoco
```

With the `glb` extra, use `--seam glb` to create a portable glTF scene. Start from [`examples/warehouse`](examples/warehouse), [`examples/robot-cell`](examples/robot-cell), or [`examples/solar-field`](examples/solar-field), then edit the `.orb` files. See [`docs/language.md`](docs/language.md) for the language and [`docs/seams.md`](docs/seams.md) to implement an external simulator adapter.

## Configuration

Orb reads the project directory passed to the CLI (default: the current directory). It writes generated files under that project's `build/<seam>/` directory. `orb apply --seam NAME` selects an exporter; a `seam "NAME" { command = "..." }` block or an `orb-seam-NAME` executable on `PATH` can provide a custom adapter. No hosted account, network service, or vendor identity is required for the quickstart.

## Scope and prerequisites

This initial public core includes the OrbLang compiler, world graph, synthetic examples, the external seam protocol, and two built-in outputs: MuJoCo XML and binary glTF. It does not include STEP/CAD or Minecraft exporters, Git `source`/`pull`, phone-scan capture, USD export, Isaac or other remote simulator management, a desktop application, hosted collaboration or storage, cloud presentation/live-control features, product-specific adapters, or release installers. Those features require separate, independently reviewable adapters or packaging. No robot or site meshes, scans, photographs, or proprietary SDKs are bundled.

The GLB writer uses NumPy and Pillow when the `glb` extra is installed. MuJoCo and other simulator applications are separate prerequisites; they are needed to open or run exported files, not to generate MuJoCo XML. Importing USD files additionally requires a separately installed `usd-core` package under its upstream license; it is not declared, installed, or redistributed here. Any future third-party SDK or asset must be obtained from its owner and reviewed under its own license before use or redistribution.

## License

Original code is licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

## Contributing

See [CONTRIBUTING](https://github.com/couplet-physical-ai/.github/blob/main/CONTRIBUTING.md). Contributions require DCO sign-off (`git commit -s`); no CLA is required.

## Security

Please report vulnerabilities privately using [GitHub private vulnerability reporting](https://github.com/couplet-physical-ai/orb/security/advisories/new), or contact software@exowatt.com. See the [security policy](https://github.com/couplet-physical-ai/.github/blob/main/SECURITY.md); do not open a public issue for a vulnerability.
