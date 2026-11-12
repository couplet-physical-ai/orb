# OrbLang quick reference

OrbLang describes a project as one or more `.orb` files. A project can be a directory containing `main.orb`, or a path to one `.orb` file. Blocks may refer to other blocks regardless of their declaration order.

## Parts, assemblies, sites and items

```hcl
part "crate" {
  mass = 4 kg
  visual "box" { size = [0.4, 0.3, 0.3] m  color = "blue" }
  collision "box" { size = [0.4, 0.3, 0.3] m }
}

site "warehouse" {
  floor = [12, 8] m
  zone "aisle" { at = [0, 0, 0] m  size = [3, 8] m }
}

item "crate-01" {
  of = part.crate
  location = site.warehouse.zone.aisle
  at = [1, 2, 0.15] m
}
```

`part` describes a reusable kind; `assembly` groups linked parts; `site` contains zones and fixtures; `item` places a physical instance. Geometry supports boxes, spheres, cylinders, capsules and project-relative meshes. A collision shape may be separate from the rendered visual.

## Units and expressions

Physical values carry units: `2 m`, `35 cm`, `5 kg`, `90 deg`, `2 s`, and derived values such as `12 N` or `0.3 kg*m2`. The compiler checks dimensions and normalizes the world graph to SI units. Use `let` for local values, `fn` for functions and `module` for parameterized reusable structures.

```hcl
let pitch = 1.2 m
fn grid_width(count) { return (count - 1) * pitch }
let width = grid_width(6)
```

OrbLang also supports lists, comprehensions, conditionals, references, imports, scene composition, motion timelines, lifecycle transitions and seam-specific extensions. The parser rejects unknown attributes, invalid units, unresolved references and invalid state transitions before any output is written.

## CLI

```sh
uv run orb validate examples/warehouse
uv run orb ir examples/warehouse
uv run orb apply examples/warehouse --seam mujoco
```

`validate` compiles and checks a project, `ir` prints its JSON world graph, and `apply` sends that graph to a selected built-in or external seam. Generated files go to `build/<seam>/` in the project.

## Imports and scene composition

A project can import another OrbLang file or directory under a namespace. Imported source files stay separate, and references use that namespace:

```hcl
import "../parts/catalog.orb" as catalog
let robot = instance(catalog.assembly.gripper, id = "gripper-01")
```

`instance` and `compose` build reusable, immutable scenes; `output` places a scene into the world graph. Instance paths are stable IDs that seams can use for selection:

```hcl
let cell = compose(
  instance(part.crate, id = "left"),
  instance(part.crate, id = "right", at = [1, 0, 0] m)
)
output "work-cell" { value = cell  label = "Synthetic work cell" }
camera "overview" { view = overview(cell, pitch = 45 deg) }
```

## Motion and views

Motion timelines describe item placement over time; joint timelines can drive articulated assemblies. The compiler resolves units and references before a seam receives the graph. `bounds(scene)` and `overview(scene, ...)` provide geometry-aware view data for a camera or shot. Camera framing is calculated from the scene geometry and does not require a rendering service.
