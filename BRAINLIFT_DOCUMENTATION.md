# Brainlift Documentation: Graph View for Zed Editor

## Project Overview

**Goal**: Build an interactive graph visualization panel for the Zed code editor that displays project files as nodes and import relationships as edges.

**Tech Stack**: Rust, GPUI (Zed's GPU-accelerated UI framework)

**Prior Experience**: No prior Rust experience; first time working with GPUI framework.

---

## Day 1-2: Research, Learning, and Codebase Onboarding

### AI Prompts Used

1. "Explain the Zed editor architecture and how panels are integrated"
2. "How does GPUI work? What are Entity, Context, Window?"
3. "Show me examples of existing panels in Zed and how they implement the Panel trait"
4. "What's the difference between `Model<T>` and `Entity<T>` in GPUI?" (learned these were renamed)
5. "How do I subscribe to project events to detect when files change?"

### Learning Breakthroughs

**Rust Ownership Model**
- Initially struggled with borrow checker errors when trying to access `self` in closures
- Breakthrough: Learned to use `cx.listener()` pattern for event handlers that need mutable access
- Breakthrough: Variable shadowing with `.clone()` for async contexts

**GPUI Architecture**
- `Entity<T>` is a handle to state, not the state itself
- `Context<T>` provides mutable access during updates
- `Window` is now passed explicitly (recent API change)
- `cx.spawn()` takes async closures for background work

**Panel Integration**
- Panels implement the `Panel` trait with methods like `position()`, `size()`, `icon()`
- Must register settings and actions in `init()` function
- Workspace manages panel lifecycle and dock positioning

### Codebase Map Created

```
crates/
├── gpui/           # UI framework - Entity, Context, Window, elements
├── workspace/      # Panel trait, dock system, workspace management
├── project/        # File system, worktrees, project events
├── editor/         # Text editing (reference for patterns)
├── terminal_view/  # Example panel implementation
└── zed/            # Main app initialization
```

### Technical Decisions

**Decision**: Create a new crate `graph_view` rather than adding to existing crate
- **Rationale**: Follows Zed's modular architecture pattern
- **Trade-off**: More boilerplate but cleaner separation of concerns

**Decision**: Use canvas element for graph rendering
- **Rationale**: Need custom shapes (hexagons, bezier curves) and smooth animations
- **Alternative considered**: Compose from div elements - rejected due to performance concerns

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| `AppContext` not found | API changed - use `App` instead |
| `ModelContext` not found | API changed - use `Context<T>` instead |
| Closures can't capture `&mut self` | Use `cx.listener()` pattern |
| Panel not appearing | Need to register in workspace and add to `main.rs` |

---

## Day 3-4: Core Development

### AI Prompts Used

1. "How do I draw custom shapes with PathBuilder in GPUI?"
2. "Implement a force-directed graph layout algorithm in Rust"
3. "How do I handle mouse drag events in GPUI?"
4. "Parse Python import statements with regex in Rust"
5. "What's the idiomatic way to run a continuous animation loop in GPUI?"

### What Changed & Why

**Initial Implementation: Basic Panel**
```rust
// Started with simple file listing
fn render(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
    div().child(Label::new(format!("Files: {}", self.graph.nodes.len())))
}
```
- **Why**: Validate panel integration before adding complexity

**Added Canvas Rendering**
- **Why**: Standard divs couldn't handle custom shapes and animations
- **Challenge**: PathBuilder API wasn't documented well
- **Solution**: Read GPUI source code to find `curve_to()`, `close()` methods

**Force-Directed Layout**
```rust
// Core physics loop
fn update_forces(&mut self) {
    // Repulsion between all nodes
    // Attraction along edges
    // Attraction to directory centroids
    // Damping for stability
}
```
- **Why**: Creates natural clustering of related files
- **Breakthrough**: Using `cx.spawn()` with `smol::Timer` for continuous simulation

### Learning Breakthroughs

**Canvas Rendering in GPUI**
- `canvas()` takes two closures: prepaint (collect data) and paint (draw)
- `PathBuilder::fill()` vs `PathBuilder::stroke()` for filled vs outlined shapes
- `window.paint_path()` to actually render

**Async Patterns**
- `cx.spawn(async move |this, cx| ...)` for Entity methods
- `cx.background_spawn()` for CPU-intensive work off main thread
- `Task` must be stored or `.detach()`ed to keep running

**Mouse Event Handling**
```rust
.on_mouse_down(MouseButton::Left, cx.listener(|this, event, window, cx| {
    // Find clicked node, start dragging
}))
.on_mouse_move(cx.listener(|this, event, window, cx| {
    // Update dragged node position
}))
.on_mouse_up(MouseButton::Left, cx.listener(|this, event, window, cx| {
    // Release node, detect click vs drag
}))
```

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Nodes not draggable | Track `drag_offset` from initial click position |
| Click vs drag detection | Track `has_dragged` flag, threshold of 5px movement |
| Graph never settles | Add damping factor (0.85) and velocity threshold |
| Arrows pointing wrong way | Swap from/to in edge rendering |

---

## Day 4-5: Feature Expansion

### AI Prompts Used

1. "Parse JavaScript/TypeScript import statements including dynamic imports"
2. "Implement Graham scan convex hull algorithm in Rust"
3. "How do I draw smooth bezier curves through a set of points?"
4. "Add regex filtering to a Rust application"
5. "How to resolve relative imports in different programming languages"

### What Changed & Why

**Multi-Language Import Parsing**
- Added parsers for 30+ languages
- **Why**: Make the tool useful for any project, not just Python
- **Challenge**: Each language has unique import syntax
- **Solution**: Language enum with dedicated parser functions

**Convex Hull Lassos**
- Replaced circle lassos with convex hull + bezier curves
- **Why**: Circles wasted space and overlapped badly
- **Algorithm**: Graham scan for hull, quadratic beziers between midpoints

```rust
fn convex_hull(points: &[Point<f32>]) -> Vec<Point<f32>> {
    // Sort by polar angle from lowest point
    // Graham scan: maintain stack of left turns
}
```

**Hierarchical Directory Containment**
- Child directories attracted to parent centroids
- Containment force pushes strays back inside
- **Why**: Visual hierarchy matches file system hierarchy

**Regex Filtering**
- Added `TextInput` for filter pattern
- Recompile regex on change, filter nodes in layout
- **Why**: Large projects need ability to focus on subsets

### Technical Decisions

**Decision**: Use regex for import parsing, not AST
- **Rationale**: Fast, covers 90% of cases, no parser dependencies
- **Trade-off**: May miss edge cases (multi-line imports, comments)

**Decision**: Bezier curves for lasso boundaries
- **Rationale**: Smoother, more organic appearance
- **Implementation**: Hull points become control points, draw to midpoints

**Decision**: Dynamic canvas scaling
```rust
let scale = 1.0 + (node_count as f32).sqrt() * 0.15;
```
- **Rationale**: Large projects need more space but not linearly
- **Why sqrt**: Logarithmic growth prevents excessive canvas sizes

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| `quadratic_to` not found | Method is `curve_to(to, control)` |
| Lassos overlap chaotically | Removed inter-lasso repulsion, use containment instead |
| Big projects too dense | Dynamic canvas scaling based on node count |
| 2-point hulls look bad | Fallback to circle for small directories |

---

## Day 6-7: Polish, Documentation, Deployment

### AI Prompts Used

1. "Best practices for Rust error handling in async contexts"
2. "How to implement Settings trait in Zed"
3. "Generate documentation for a Rust project"

### What Changed & Why

**Visual Polish**
- Hexagonal nodes (sunset-orange #dd5013)
- Lavender lassos (#bdb7fc)
- Cocoa edges (#5d3d3a)
- **Why**: Distinctive, non-generic aesthetic

**Performance Optimization**
- Increased initial stabilization iterations (50 → 100)
- Dynamic bounds prevent wasted computation
- Node position clamping prevents runaway physics

**Code Cleanup**
- Removed unused functions (flagged by warnings)
- Consistent error handling with `?` propagation
- Documentation comments on public items

### Final Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     GraphView Panel                          │
├──────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌──────────────────────────┐│
│ │ FileGraph   │ │ GraphLayout │ │ Import Parsers           ││
│ │ HashMap of  │ │ NodeLayout  │ │ - Python, JS/TS, Rust    ││
│ │ nodes/edges │ │ positions   │ │ - Go, C/C++, Java, etc.  ││
│ └─────────────┘ └─────────────┘ └──────────────────────────┘│
├──────────────────────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────────────────────┐│
│ │              Force Simulation (async loop)               ││
│ │  - Node repulsion (inverse square)                       ││
│ │  - Edge attraction (spring)                              ││
│ │  - Directory clustering                                  ││
│ │  - Hierarchical containment                              ││
│ └──────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────┤
│ ┌──────────────────────────────────────────────────────────┐│
│ │              Canvas Rendering                            ││
│ │  - Convex hull lassos (bezier curves)                   ││
│ │  - Hexagonal nodes                                       ││
│ │  - Directed arrows                                       ││
│ │  - Interactive drag/click                                ││
│ └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| TextInput not editable | Use `TextInput::new()` with `on_change` listener |
| Nodes escape bounds | Add boundary clamping + velocity reversal |
| Settings not persisting | Implement `Settings` trait properly |

---

## Key Learnings Summary

### Rust-Specific

1. **Ownership in UI**: Clone data for closures, use `Arc` for shared state
2. **Async patterns**: `cx.spawn()` for Entity work, `background_spawn()` for CPU
3. **Error handling**: Prefer `?` propagation, use `.log_err()` for fire-and-forget
4. **Pattern matching**: Powerful for handling enums and Options

### GPUI-Specific

1. **Recent API changes**: `Entity` replaces `Model/View`, `App` replaces `AppContext`
2. **Canvas rendering**: Two-phase prepaint/paint pattern
3. **Event handling**: `cx.listener()` for self-mutating callbacks
4. **Panel integration**: Trait implementation + registration in init

### Force-Directed Graphs

1. **Physics balance**: Repulsion vs attraction constants are critical
2. **Damping**: Essential for stability (0.85 works well)
3. **Hierarchical forces**: Add containment for nested structures
4. **Performance**: Limit iterations, use spatial data structures for large graphs

### AI-Assisted Development

1. **Code generation**: Great for boilerplate, needs verification
2. **API discovery**: Faster than documentation for unfamiliar codebases
3. **Debugging**: Excellent at explaining error messages
4. **Architecture**: Good for discussing trade-offs, less good at holistic design

---

## Metrics

- **Lines of code written**: ~2,600 (graph_view.rs)
- **Languages supported**: 30+
- **Time to first working prototype**: ~4 hours
- **Time to polished feature**: ~20 hours
- **Compiler errors fixed**: 50+ (ownership, lifetimes, API changes)
- **AI prompts used**: ~100+


