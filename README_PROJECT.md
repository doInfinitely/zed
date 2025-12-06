# Graph View for Zed Editor

A force-directed graph visualization panel for the Zed code editor that displays project structure as an interactive network of files and their import relationships.

## Original Repository

**Zed Editor** - https://github.com/zed-industries/zed

Zed is a high-performance, multiplayer code editor written in Rust using GPUI (GPU-accelerated UI framework). It was chosen for this project because:
- Substantial codebase (~500k+ lines of Rust)
- Well-documented architecture with clear extension points
- Modern Rust patterns and async programming
- Active development community

## What I Built

### Graph View Panel

A new panel that provides a visual, interactive graph representation of your project's file structure and dependencies:

**Core Features:**
- **Force-directed graph layout** - Files are represented as hexagonal nodes that naturally arrange themselves using spring physics simulation
- **Import relationship visualization** - Edges (arrows) show which files import other files
- **Directory lassos** - Files are grouped by directory with smooth convex hull boundaries using bezier curves
- **Hierarchical nesting** - Child directories are visually contained within parent directory lassos
- **Interactive nodes** - Drag nodes to rearrange the graph; they spring back when released
- **Click to open** - Click any node to open that file in the editor
- **Regex filtering** - Filter visible files using regex patterns
- **Dynamic scaling** - Canvas automatically expands for large projects

**Multi-language Support:**
Parses imports from 30+ languages including:
- Python, JavaScript/TypeScript, Rust, Go, C/C++, Java, Ruby, Swift, Elixir
- CSS, HTML, Vue, JSON, YAML, TOML, Markdown
- And many more...

**Visual Design:**
- Sunset-orange (#dd5013) hexagonal nodes
- Lavender (#bdb7fc) directory lassos with bezier curves
- Cocoa (#5d3d3a) edge arrows
- Smooth animations and spring physics

## Architecture Overview

### New Crate: `graph_view`

```
crates/graph_view/
├── Cargo.toml           # Dependencies: gpui, workspace, project, regex, smol
└── src/
    └── graph_view.rs    # Main implementation (~2600 lines)
```

### Key Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        GraphView Panel                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ FileGraph   │  │ GraphLayout │  │ Force Simulation        │  │
│  │ - nodes     │  │ - positions │  │ - repulsion             │  │
│  │ - edges     │  │ - velocities│  │ - attraction            │  │
│  │ - imports   │  │ - directories│ │ - hierarchical nesting  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Canvas Rendering                         ││
│  │  - Convex hull lassos with bezier curves                   ││
│  │  - Hexagonal nodes                                          ││
│  │  - Directed edge arrows                                     ││
│  │  - Interactive drag & click handling                        ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Project Scanning**: Subscribes to `Project` events to detect file changes
2. **Import Parsing**: Language-specific regex parsers extract import statements
3. **Graph Building**: Creates nodes for files and edges for import relationships
4. **Layout Initialization**: Positions nodes hierarchically by directory
5. **Force Simulation**: Continuous async loop applies physics forces
6. **Rendering**: GPUI canvas draws nodes, edges, and lassos

### Integration Points

- **Workspace Panel System**: Implements `Panel` trait for dock integration
- **Project Events**: Subscribes to `WorktreeAdded`, `WorktreeUpdatedEntries`
- **Settings**: `GraphViewSettings` for dock position and size
- **Actions**: `ToggleGraphView` action with keybinding

## Setup + Run Steps

### Prerequisites

- macOS (primary platform) or Linux
- Rust toolchain (1.75+)
- Xcode Command Line Tools (macOS)

### Building

```bash
# Clone the forked repository
git clone https://github.com/YOUR_USERNAME/zed.git
cd zed

# Build in release mode (recommended for performance)
cargo build --release

# Run Zed
./target/release/zed
```

### Using the Graph View

1. Open a project in Zed
2. Toggle the Graph View panel:
   - Use the command palette: `graph view: toggle`
   - Or click the graph icon in the dock
3. The graph will populate with your project's files
4. Interact with the graph:
   - **Drag nodes** to rearrange
   - **Click nodes** to open files
   - **Use regex filter** to focus on specific files
   - **Scroll** to navigate large graphs

## Technical Decisions

### Why GPUI Canvas?

The graph visualization uses GPUI's `canvas` element for custom drawing rather than composing standard UI elements because:
- **Performance**: Direct GPU rendering handles thousands of nodes smoothly
- **Custom shapes**: Bezier curves for lassos, hexagons for nodes
- **Continuous animation**: 60fps force simulation without layout thrashing

### Why Convex Hull + Bezier Curves for Lassos?

Initial implementations used circles, but:
- Circles wasted space and overlapped awkwardly
- Convex hulls tightly wrap the actual node positions
- Bezier curves between hull points create smooth, organic shapes
- Padding expansion ensures nodes stay visually inside

### Why Force-Directed Layout?

- **Self-organizing**: Related files naturally cluster together
- **Interactive**: Users can manually adjust and explore
- **Scalable**: Works for small and large projects
- **Intuitive**: Physical metaphor is easy to understand

### Why Per-Language Import Parsing?

Each language has unique import syntax:
- Python: `import x` and `from x import y`
- JavaScript: `import`, `require()`, dynamic imports
- Rust: `use`, `mod`, `crate::`
- Go: `import "path"`
- etc.

Regex-based parsing is fast and handles most common patterns without needing full AST parsing.

### Dynamic Canvas Scaling

Large projects need more space:
```rust
let scale_factor = 1.0 + (node_count as f32).sqrt() * 0.15;
let canvas_size = 1200.0 + (node_count as f32).sqrt() * 800.0;
```

This gives logarithmic growth - enough space without being excessive.

### Hierarchical Containment

Child directories are attracted toward parent directory centroids with containment forces that push nodes back if they stray outside the parent's bounds. This creates the visual nesting effect.

## Files Changed

### New Files
- `crates/graph_view/Cargo.toml` - Crate manifest
- `crates/graph_view/src/graph_view.rs` - Main implementation

### Modified Files
- `Cargo.toml` - Added `graph_view` to workspace dependencies
- `crates/zed/Cargo.toml` - Added `graph_view` dependency
- `crates/zed/src/main.rs` - Initialize graph_view on startup
