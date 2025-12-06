use anyhow::{Context as _, Result};
use collections::HashMap;
use db::kvp::KEY_VALUE_STORE;
use gpui::{
    actions, Action, App, AsyncApp, AsyncWindowContext, Context, ElementId, Entity, EventEmitter,
    FocusHandle, Focusable, Hsla, InteractiveElement, KeyDownEvent, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, ParentElement, PathBuilder, Pixels, Point, Render, Styled,
    Subscription, Task, WeakEntity, Window, canvas, div, point, px, rgb,
};
use project::{Project, ProjectPath, WorktreeId};
use regex::Regex;
use serde::{Deserialize, Serialize};
use settings::{DockSide, RegisterSetting, Settings};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use ui::{prelude::*, IconName, Label};
use util::{ResultExt, TryFutureExt};
use workspace::{
    dock::{DockPosition, Panel, PanelEvent},
    Workspace,
};

const GRAPH_VIEW_KEY: &str = "GraphView";

actions!(graph_view, [ToggleFocus, UpdateFilter]);

pub fn init(cx: &mut App) {
    cx.observe_new(|workspace: &mut Workspace, _, _cx| {
        workspace.register_action(|workspace, _: &ToggleFocus, window, cx| {
            workspace.toggle_panel_focus::<GraphView>(window, cx);
        });
    })
    .detach();
}

#[derive(Debug, Clone, Copy, PartialEq, RegisterSetting)]
pub struct GraphViewSettings {
    pub button: bool,
    pub default_width: Pixels,
    pub dock: DockSide,
}

impl Settings for GraphViewSettings {
    fn from_settings(_content: &settings::SettingsContent) -> Self {
        Self {
            button: true,
            default_width: px(360.0),
            dock: DockSide::Right,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct SerializedGraphView {
    width: Option<Pixels>,
}

pub struct GraphView {
    project: Entity<Project>,
    workspace: WeakEntity<Workspace>,
    width: Option<Pixels>,
    focus_handle: FocusHandle,
    pending_serialization: Task<Option<()>>,
    graph: ProjectGraph,
    layout: GraphLayout,
    dragging_node: Option<usize>,
    drag_offset: Point<f32>,
    drag_start_pos: Point<f32>,
    has_dragged: bool,
    filter_pattern: String,
    compiled_filter: Option<Regex>,
    simulation_task: Task<()>,
    _subscriptions: Vec<Subscription>,
}

struct GraphLayout {
    nodes: Vec<NodeLayout>,
    directories: HashMap<PathBuf, DirectoryGroup>,
    viewport_offset: Point<Pixels>,
    is_running: bool,
    // Adaptive damping state
    current_damping: f32,
    high_energy_frames: u32,
    base_damping: f32,
}

impl Default for GraphLayout {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            directories: HashMap::default(),
            viewport_offset: Point::default(),
            is_running: false,
            current_damping: 0.85,
            high_energy_frames: 0,
            base_damping: 0.85,
        }
    }
}

#[derive(Clone)]
struct NodeLayout {
    path: PathBuf,
    directory: PathBuf,
    position: Point<f32>,
    velocity: Point<f32>,
    is_pinned: bool,
    worktree_id: Option<WorktreeId>,
}

#[derive(Clone)]
struct DirectoryGroup {
    path: PathBuf,
    node_indices: Vec<usize>,
    centroid: Point<f32>,
}

#[derive(Default)]
struct ProjectGraph {
    nodes: HashMap<PathBuf, FileNode>,
    edges: Vec<ImportEdge>,
    // Map from module name to file path for quick lookup
    module_to_path: HashMap<String, PathBuf>,
}

struct FileNode {
    path: PathBuf,
    worktree_id: WorktreeId,
    imports: Vec<String>,
}

#[derive(Clone)]
struct ImportEdge {
    from: PathBuf,
    to: PathBuf,
    import_name: String,
}

impl GraphView {
    pub async fn load(
        workspace: WeakEntity<Workspace>,
        mut cx: AsyncWindowContext,
    ) -> Result<Entity<Self>> {
        let serialized_panel = match workspace
            .read_with(&cx, |workspace, _| GraphView::serialization_key(workspace))
            .ok()
            .flatten()
        {
            Some(serialization_key) => cx
                .background_spawn(async move { KEY_VALUE_STORE.read_kvp(&serialization_key) })
                .await
                .context("loading graph view")
                .log_err()
                .flatten()
                .map(|panel| serde_json::from_str::<SerializedGraphView>(&panel))
                .transpose()
                .log_err()
                .flatten(),
            None => None,
        };

        workspace.update_in(&mut cx, |workspace, window, cx| {
            let panel = Self::new(workspace, window, cx);
            if let Some(serialized_panel) = serialized_panel {
                panel.update(cx, |panel, cx| {
                    panel.width = serialized_panel.width;
                    cx.notify();
                });
            }
            panel
        })
    }

    fn new(workspace: &Workspace, window: &mut Window, cx: &mut Context<Workspace>) -> Entity<Self> {
        let project = workspace.project().clone();
        let workspace_handle = cx.entity().downgrade();
        
        cx.new(|cx: &mut Context<Self>| {
            let focus_handle = cx.focus_handle();
            
            let mut subscriptions = Vec::new();
            
            subscriptions.push(cx.observe(&project, |this, _, cx| {
                this.update_graph_initial(cx);
                cx.notify();
            }));
            
            subscriptions.push(cx.subscribe_in(&project, window, |_this, _project, event, window, cx| {
                match event {
                    project::Event::WorktreeAdded(_)
                    | project::Event::WorktreeUpdatedEntries(_, _)
                    | project::Event::WorktreeOrderChanged => {
                        cx.defer_in(window, |this, _, cx| {
                            this.update_graph_initial(cx);
                            cx.notify();
                        });
                    }
                    _ => {}
                }
            }));
            
            let mut this = Self {
                project: project.clone(),
                workspace: workspace_handle,
                width: None,
                focus_handle,
                pending_serialization: Task::ready(None),
                graph: ProjectGraph::default(),
                layout: GraphLayout::default(),
                dragging_node: None,
                drag_offset: Point { x: 0.0, y: 0.0 },
                drag_start_pos: Point { x: 0.0, y: 0.0 },
                has_dragged: false,
                filter_pattern: String::new(),
                compiled_filter: None,
                simulation_task: Task::ready(()),
                _subscriptions: subscriptions,
            };
            this.update_graph_initial(cx);
            this.start_simulation(cx);
            this
        })
    }

    fn serialization_key(workspace: &Workspace) -> Option<String> {
        workspace
            .database_id()
            .map(|id| i64::from(id).to_string())
            .or(workspace.session_id())
            .map(|id| format!("{}-{:?}", GRAPH_VIEW_KEY, id))
    }

    fn serialize(&mut self, cx: &mut Context<Self>) {
        let Some(_workspace) = self.workspace.upgrade() else {
            return;
        };
        let Some(serialization_key) = self
            .workspace
            .read_with(cx, |workspace, _| {
                GraphView::serialization_key(workspace)
            })
            .ok()
            .flatten()
        else {
            return;
        };

        let width = self.width;
        self.pending_serialization = cx.background_spawn(
            async move {
                KEY_VALUE_STORE
                    .write_kvp(
                        serialization_key,
                        serde_json::to_string(&SerializedGraphView { width })?,
                    )
                    .await?;
                anyhow::Ok(())
            }
            .log_err(),
        );
    }

    fn update_graph_initial(&mut self, cx: &mut Context<Self>) {
        self.graph = ProjectGraph::default();
        
        let worktrees = self.project.read(cx).worktrees(cx).collect::<Vec<_>>();
        
        if worktrees.is_empty() {
            return;
        }
        
        // First pass: collect all supported files
        for worktree in &worktrees {
            let worktree_id = worktree.read(cx).id();
            let snapshot = worktree.read(cx).snapshot();
            let worktree_root = snapshot.abs_path();
            
            for entry in snapshot.entries(false, 0) {
                if entry.is_file() {
                    if let Some(extension) = entry.path.as_std_path().extension() {
                        let ext_str = extension.to_string_lossy();
                        if Language::from_extension(&ext_str).is_some() {
                            let path = worktree_root.join(entry.path.as_std_path());
                            let module_name = Self::path_to_module_name(&path, &worktree_root);
                            
                            self.graph.nodes.insert(
                                path.clone(),
                                FileNode {
                                    path: path.clone(),
                                    worktree_id,
                                    imports: Vec::new(),
                                },
                            );
                            
                            if let Some(module_name) = module_name {
                                self.graph.module_to_path.insert(module_name, path);
                            }
                        }
                    }
                }
            }
        }
        
        // Second pass: parse imports and build edges
        let nodes: Vec<_> = self.graph.nodes.values()
            .map(|node| node.path.clone())
            .collect();
        
        for path in nodes {
            let language = path.extension()
                .and_then(|ext| Language::from_extension(&ext.to_string_lossy()));
            
            let Some(language) = language else {
                continue;
            };
            
            if let Ok(content) = std::fs::read_to_string(&path) {
                let imports = Self::parse_imports(&content, language);
                
                if let Some(node) = self.graph.nodes.get_mut(&path) {
                    node.imports = imports.clone();
                }
                
                // Build edges - try to match imports to known files
                for import in imports {
                    // Try direct module name match
                    if let Some(target_path) = self.graph.module_to_path.get(&import) {
                        self.graph.edges.push(ImportEdge {
                            from: path.clone(),
                            to: target_path.clone(),
                            import_name: import.clone(),
                        });
                        continue;
                    }
                    
                    // Try to find matching file by various strategies
                    if let Some(target_path) = self.find_import_target(&path, &import, language) {
                        self.graph.edges.push(ImportEdge {
                            from: path.clone(),
                            to: target_path,
                            import_name: import,
                        });
                    }
                }
            }
        }
        
        // Initialize layout
        self.initialize_layout();
    }
    
    fn find_import_target(&self, source_path: &Path, import: &str, language: Language) -> Option<PathBuf> {
        // Get the directory of the source file
        let source_dir = source_path.parent()?;
        
        // Try various resolution strategies based on language
        let candidates: Vec<PathBuf> = match language {
            Language::Python => {
                let parts: Vec<&str> = import.split('.').collect();
                let relative_path = parts.join("/");
                vec![
                    source_dir.join(format!("{}.py", relative_path)),
                    source_dir.join(&relative_path).join("__init__.py"),
                ]
            }
            Language::JavaScript | Language::TypeScript => {
                let base = import.trim_start_matches("@");
                vec![
                    source_dir.join(format!("{}.js", base)),
                    source_dir.join(format!("{}.ts", base)),
                    source_dir.join(format!("{}.jsx", base)),
                    source_dir.join(format!("{}.tsx", base)),
                    source_dir.join(base).join("index.js"),
                    source_dir.join(base).join("index.ts"),
                ]
            }
            Language::Rust => {
                vec![
                    source_dir.join(format!("{}.rs", import)),
                    source_dir.join(import).join("mod.rs"),
                ]
            }
            Language::Go => {
                // Go imports are package paths
                vec![]
            }
            Language::C | Language::Cpp => {
                vec![
                    source_dir.join(import),
                    source_dir.join(format!("{}.h", import)),
                    source_dir.join(format!("{}.hpp", import)),
                ]
            }
            Language::Ruby => {
                vec![
                    source_dir.join(format!("{}.rb", import)),
                ]
            }
            Language::Java | Language::Kotlin | Language::Scala => {
                // Java imports are fully qualified class names
                let parts: Vec<&str> = import.split('.').collect();
                if let Some(last) = parts.last() {
                    let ext = match language {
                        Language::Java => "java",
                        Language::Kotlin => "kt",
                        Language::Scala => "scala",
                        _ => return None,
                    };
                    vec![source_dir.join(format!("{}.{}", last, ext))]
                } else {
                    vec![]
                }
            }
            _ => {
                // For other languages, try direct match
                vec![
                    source_dir.join(import),
                ]
            }
        };
        
        // Check if any candidate exists in our graph
        for candidate in candidates {
            if self.graph.nodes.contains_key(&candidate) {
                return Some(candidate);
            }
        }
        
        // Try fuzzy matching by filename
        let import_filename = Path::new(import)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(import);
        
        for (path, _) in &self.graph.nodes {
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                let name_without_ext = filename.rsplit('.').last().unwrap_or(filename);
                if name_without_ext == import_filename || filename == import_filename {
                    return Some(path.clone());
                }
            }
        }
        
        None
    }
    
    fn initialize_layout(&mut self) {
        self.layout = GraphLayout::default();
        
        // Filter nodes based on regex pattern
        let filtered_nodes: Vec<_> = self.graph.nodes.iter()
            .filter(|(path, _)| self.node_matches_filter(path))
            .collect();
        
        let node_count = filtered_nodes.len();
        if node_count == 0 {
            return;
        }
        
        // Group nodes by directory
        let mut dir_to_nodes: HashMap<PathBuf, Vec<usize>> = HashMap::default();
        
        // Collect all directories and sort by depth (shallowest first)
        let mut all_dirs: Vec<PathBuf> = filtered_nodes.iter()
            .map(|(path, _)| path.parent().unwrap_or(path.as_path()).to_path_buf())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        all_dirs.sort_by_key(|p| p.components().count());
        
        // Assign positions to directories hierarchically
        // Scale layout based on node count for big projects
        let scale_factor = 1.0 + (node_count as f32).sqrt() * 0.15;
        let mut dir_positions: HashMap<PathBuf, Point<f32>> = HashMap::default();
        let base_x = 400.0 * scale_factor;
        let base_y = 400.0 * scale_factor;
        let base_radius = 250.0 * scale_factor;
        
        // Position root-level directories in a circle
        let root_dirs: Vec<_> = all_dirs.iter()
            .filter(|dir| {
                !all_dirs.iter().any(|other| {
                    other != *dir && dir.starts_with(other)
                })
            })
            .collect();
        
        // Give more space between root directories based on how many there are
        let dir_spacing_radius = base_radius + (root_dirs.len() as f32 * 50.0);
        
        for (i, dir) in root_dirs.iter().enumerate() {
            let angle = (i as f32 / root_dirs.len().max(1) as f32) * std::f32::consts::TAU;
            let x = base_x + dir_spacing_radius * angle.cos();
            let y = base_y + dir_spacing_radius * angle.sin();
            dir_positions.insert((*dir).clone(), Point { x, y });
        }
        
        // Position child directories relative to their parents
        for dir in &all_dirs {
            if dir_positions.contains_key(dir) {
                continue;
            }
            
            // Find parent directory
            if let Some(parent_pos) = all_dirs.iter()
                .filter(|other| *other != dir && dir.starts_with(other))
                .max_by_key(|p| p.components().count())
                .and_then(|parent| dir_positions.get(parent))
            {
                // Position child near parent with some offset (scaled)
                let offset_angle = (dir.to_string_lossy().len() as f32 * 0.5) % std::f32::consts::TAU;
                let offset_dist = 100.0 * scale_factor;
                dir_positions.insert(dir.clone(), Point {
                    x: parent_pos.x + offset_dist * offset_angle.cos(),
                    y: parent_pos.y + offset_dist * offset_angle.sin(),
                });
            } else {
                // Fallback position
                dir_positions.insert(dir.clone(), Point { x: base_x, y: base_y });
            }
        }
        
        // Create nodes positioned around their directory centers
        for (i, (path, file_node)) in filtered_nodes.iter().enumerate() {
            let directory = path.parent().unwrap_or(path.as_path()).to_path_buf();
            let dir_center = dir_positions.get(&directory).cloned().unwrap_or(Point { x: base_x, y: base_y });
            
            // Get index within this directory for positioning (scaled for big projects)
            let dir_node_count = dir_to_nodes.entry(directory.clone()).or_insert_with(Vec::new).len();
            let angle = (dir_node_count as f32 * 0.8) + (i as f32 * 0.3);
            let node_radius = (60.0 + (dir_node_count as f32 * 20.0)) * scale_factor;
            
            let x = dir_center.x + node_radius * angle.cos();
            let y = dir_center.y + node_radius * angle.sin();
            
            self.layout.nodes.push(NodeLayout {
                path: (*path).clone(),
                directory: directory.clone(),
                position: Point { x, y },
                velocity: Point { x: 0.0, y: 0.0 },
                is_pinned: false,
                worktree_id: Some(file_node.worktree_id),
            });
            
            dir_to_nodes.get_mut(&directory).unwrap().push(self.layout.nodes.len() - 1);
        }
        
        // Create directory groups
        for (dir_path, node_indices) in dir_to_nodes {
            self.layout.directories.insert(
                dir_path.clone(),
                DirectoryGroup {
                    path: dir_path,
                    node_indices,
                    centroid: Point { x: 0.0, y: 0.0 },
                },
            );
        }
        
        // Run initial iterations to stabilize
        for _ in 0..100 {
            self.update_forces();
        }
        
        self.layout.is_running = true;
    }
    
    fn start_simulation(&mut self, cx: &mut Context<Self>) {
        self.layout.is_running = true;
        self.simulation_task = cx.spawn(async |this, cx| {
            loop {
                cx.background_spawn(async {
                    smol::Timer::after(std::time::Duration::from_millis(16)).await
                })
                .await;
                
                let should_continue: bool = this
                    .update(cx, |this, cx| {
                        if !this.layout.is_running {
                            return false;
                        }
                        
                        // Always update forces (dragged node is pinned so won't move)
                        this.update_forces();
                        
                        // Check if simulation has stabilized (only when not dragging)
                        if this.dragging_node.is_none() {
                            let max_velocity = this.layout.nodes.iter()
                                .filter(|n| !n.is_pinned)
                                .map(|n| (n.velocity.x * n.velocity.x + n.velocity.y * n.velocity.y).sqrt())
                                .fold(0.0f32, f32::max);
                            
                            if max_velocity < 0.1 {
                                this.layout.is_running = false;
                            }
                        }
                        
                        cx.notify();
                        true
                    })
                    .ok()
                    .unwrap_or(false);
                
                if !should_continue {
                    break;
                }
            }
        });
    }
    
    fn update_forces(&mut self) {
        let node_count = self.layout.nodes.len();
        if node_count == 0 {
            return;
        }
        
        // Constants for force-directed layout
        let repulsion_strength = 50000.0;
        let edge_attraction_strength = 0.01;
        let directory_attraction_strength = 0.03;
        let hierarchy_attraction_strength = 0.05;  // Attraction of child dirs to parent dirs
        let containment_strength = 0.1;  // Force to keep children inside parents
        
        // Use adaptive damping
        let damping = self.layout.current_damping;
        
        // Update directory centroids and calculate radii
        let mut dir_radii: HashMap<PathBuf, f32> = HashMap::default();
        for dir_group in self.layout.directories.values_mut() {
            let mut centroid = Point { x: 0.0, y: 0.0 };
            for &idx in &dir_group.node_indices {
                if idx < self.layout.nodes.len() {
                    centroid.x += self.layout.nodes[idx].position.x;
                    centroid.y += self.layout.nodes[idx].position.y;
                }
            }
            if !dir_group.node_indices.is_empty() {
                let count = dir_group.node_indices.len() as f32;
                centroid.x /= count;
                centroid.y /= count;
            }
            dir_group.centroid = centroid;
            
            // Calculate radius (max distance from centroid to any node + padding)
            let mut max_dist = 0.0f32;
            for &idx in &dir_group.node_indices {
                if idx < self.layout.nodes.len() {
                    let dx = self.layout.nodes[idx].position.x - centroid.x;
                    let dy = self.layout.nodes[idx].position.y - centroid.y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    max_dist = max_dist.max(dist);
                }
            }
            dir_radii.insert(dir_group.path.clone(), max_dist + 70.0);
        }
        
        // Build parent-child relationships
        let dir_paths: Vec<PathBuf> = self.layout.directories.keys().cloned().collect();
        let mut child_to_parent: HashMap<PathBuf, PathBuf> = HashMap::default();
        for child in &dir_paths {
            // Find the deepest parent directory
            if let Some(parent) = dir_paths.iter()
                .filter(|p| *p != child && child.starts_with(p))
                .max_by_key(|p| p.components().count())
            {
                child_to_parent.insert(child.clone(), parent.clone());
            }
        }
        
        // Calculate forces for each node
        for i in 0..node_count {
            let mut force = Point { x: 0.0, y: 0.0 };
            
            // Repulsion from other nodes
            for j in 0..node_count {
                if i != j {
                    let dx = self.layout.nodes[i].position.x - self.layout.nodes[j].position.x;
                    let dy = self.layout.nodes[i].position.y - self.layout.nodes[j].position.y;
                    let dist_sq = (dx * dx + dy * dy).max(1.0);
                    let dist = dist_sq.sqrt();
                    
                    force.x += (dx / dist) * (repulsion_strength / dist_sq);
                    force.y += (dy / dist) * (repulsion_strength / dist_sq);
                }
            }
            
            // Attraction to directory centroid
            let node_dir = self.layout.nodes[i].directory.clone();
            if let Some(dir_group) = self.layout.directories.get(&node_dir) {
                if dir_group.node_indices.len() > 1 {
                    let dx = dir_group.centroid.x - self.layout.nodes[i].position.x;
                    let dy = dir_group.centroid.y - self.layout.nodes[i].position.y;
                    
                    force.x += dx * directory_attraction_strength;
                    force.y += dy * directory_attraction_strength;
                }
            }
            
            // Hierarchical containment: attract nodes toward parent directory's centroid
            // and push them back if they're outside parent's bounding area
            if let Some(parent_dir) = child_to_parent.get(&node_dir) {
                if let Some(parent_group) = self.layout.directories.get(parent_dir) {
                    let parent_centroid = parent_group.centroid;
                    let parent_radius = dir_radii.get(parent_dir).copied().unwrap_or(200.0);
                    
                    let node_pos = self.layout.nodes[i].position;
                    let dx = node_pos.x - parent_centroid.x;
                    let dy = node_pos.y - parent_centroid.y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    
                    // Attract toward parent centroid
                    force.x += (parent_centroid.x - node_pos.x) * hierarchy_attraction_strength;
                    force.y += (parent_centroid.y - node_pos.y) * hierarchy_attraction_strength;
                    
                    // If outside parent's radius, push back strongly
                    let allowed_radius = parent_radius - 80.0;  // Leave margin for child's own nodes
                    if dist > allowed_radius && dist > 1.0 {
                        let overshoot = dist - allowed_radius;
                        force.x -= (dx / dist) * overshoot * containment_strength;
                        force.y -= (dy / dist) * overshoot * containment_strength;
                    }
                }
            }
            
            self.layout.nodes[i].velocity.x = (self.layout.nodes[i].velocity.x + force.x) * damping;
            self.layout.nodes[i].velocity.y = (self.layout.nodes[i].velocity.y + force.y) * damping;
        }
        
        // Calculate attraction forces for edges
        for edge in &self.graph.edges {
            if let (Some(from_idx), Some(to_idx)) = (
                self.layout.nodes.iter().position(|n| n.path == edge.from),
                self.layout.nodes.iter().position(|n| n.path == edge.to),
            ) {
                let dx = self.layout.nodes[to_idx].position.x - self.layout.nodes[from_idx].position.x;
                let dy = self.layout.nodes[to_idx].position.y - self.layout.nodes[from_idx].position.y;
                
                let force_x = dx * edge_attraction_strength;
                let force_y = dy * edge_attraction_strength;
                
                self.layout.nodes[from_idx].velocity.x += force_x;
                self.layout.nodes[from_idx].velocity.y += force_y;
                self.layout.nodes[to_idx].velocity.x -= force_x;
                self.layout.nodes[to_idx].velocity.y -= force_y;
            }
        }
        
        // Update positions (but not for pinned nodes) with bounds constraints
        // Scale bounds based on node count: more nodes = more space needed
        let min_bound = 50.0;
        let base_size = 1500.0;
        let size_per_node = 80.0;  // Each node adds ~80px of space
        let max_bound = base_size + (node_count as f32).sqrt() * size_per_node * 10.0;
        let max_bound_x = max_bound;
        let max_bound_y = max_bound;
        
        for node in &mut self.layout.nodes {
            if !node.is_pinned {
                node.position.x += node.velocity.x;
                node.position.y += node.velocity.y;
                
                // Keep nodes within bounds
                node.position.x = node.position.x.max(min_bound).min(max_bound_x);
                node.position.y = node.position.y.max(min_bound).min(max_bound_y);
                
                // Bounce off boundaries by reversing velocity
                if node.position.x <= min_bound || node.position.x >= max_bound_x {
                    node.velocity.x *= -0.5;
                }
                if node.position.y <= min_bound || node.position.y >= max_bound_y {
                    node.velocity.y *= -0.5;
                }
            }
        }
        
        // Adaptive damping: calculate total kinetic energy
        let kinetic_energy: f32 = self.layout.nodes.iter()
            .filter(|n| !n.is_pinned)
            .map(|n| n.velocity.x * n.velocity.x + n.velocity.y * n.velocity.y)
            .sum();
        
        // Threshold for "high energy" scales with node count
        let high_energy_threshold = 50.0 * node_count as f32;
        let low_energy_threshold = 5.0 * node_count as f32;
        let frames_before_increase = 30;  // ~0.5 seconds at 60fps
        
        if kinetic_energy > high_energy_threshold {
            self.layout.high_energy_frames += 1;
            
            // If energy has been high for a while, increase damping
            if self.layout.high_energy_frames > frames_before_increase {
                // Ratchet up damping (lower value = more damping)
                // Go from 0.85 -> 0.75 -> 0.65 -> 0.55 -> 0.45 (min)
                self.layout.current_damping = (self.layout.current_damping - 0.05).max(0.45);
                self.layout.high_energy_frames = 0;  // Reset counter for next ratchet
            }
        } else if kinetic_energy < low_energy_threshold {
            // Energy is low, gradually return to base damping
            self.layout.high_energy_frames = 0;
            if self.layout.current_damping < self.layout.base_damping {
                // Slowly restore damping (increase by small amount each frame)
                self.layout.current_damping = (self.layout.current_damping + 0.002).min(self.layout.base_damping);
            }
        } else {
            // Energy is moderate, don't change damping but reset high energy counter
            self.layout.high_energy_frames = 0;
        }
    }
    
    /// Compute convex hull using Graham scan algorithm
    fn convex_hull(points: &[Point<f32>]) -> Vec<Point<f32>> {
        if points.len() < 3 {
            return points.to_vec();
        }
        
        // Find the point with lowest y (and leftmost if tied)
        let mut start_idx = 0;
        for i in 1..points.len() {
            if points[i].y < points[start_idx].y 
                || (points[i].y == points[start_idx].y && points[i].x < points[start_idx].x) {
                start_idx = i;
            }
        }
        
        let start = points[start_idx];
        
        // Sort points by polar angle with respect to start point
        let mut sorted: Vec<Point<f32>> = points.iter().cloned().collect();
        sorted.sort_by(|a, b| {
            let angle_a = (a.y - start.y).atan2(a.x - start.x);
            let angle_b = (b.y - start.y).atan2(b.x - start.x);
            angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Graham scan
        let mut hull: Vec<Point<f32>> = Vec::new();
        
        for p in sorted {
            while hull.len() >= 2 {
                let a = hull[hull.len() - 2];
                let b = hull[hull.len() - 1];
                // Cross product to check if we make a left turn
                let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
                if cross <= 0.0 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(p);
        }
        
        hull
    }
    
    fn handle_node_drag(&mut self, position: Point<Pixels>, cx: &mut Context<Self>) {
        if let Some(node_idx) = self.dragging_node {
            if let Some(node) = self.layout.nodes.get_mut(node_idx) {
                let pos_x: f32 = position.x.into();
                let pos_y: f32 = position.y.into();
                node.position = Point { x: pos_x, y: pos_y };
                node.velocity = Point { x: 0.0, y: 0.0 };
                node.is_pinned = true;
            }
            self.layout.is_running = true;
            cx.notify();
        }
    }
    
    fn release_node(&mut self, cx: &mut Context<Self>) {
        if let Some(node_idx) = self.dragging_node {
            if let Some(node) = self.layout.nodes.get_mut(node_idx) {
                node.is_pinned = false;
            }
            self.dragging_node = None;
            self.layout.is_running = true;
            cx.notify();
        }
    }
    
    fn find_node_at_position(&self, pos: Point<Pixels>) -> Option<usize> {
        let pos_x: f32 = pos.x.into();
        let pos_y: f32 = pos.y.into();
        let node_radius = 30.0;  // Slightly larger hit area
        
        for (i, node) in self.layout.nodes.iter().enumerate() {
            let dx = pos_x - node.position.x;
            let dy = pos_y - node.position.y;
            let dist_sq = dx * dx + dy * dy;
            
            if dist_sq < node_radius * node_radius {
                return Some(i);
            }
        }
        None
    }

    fn update_graph(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        self.update_graph_initial(cx);
        cx.notify();
    }
    
    fn open_file(&self, path: PathBuf, worktree_id: Option<WorktreeId>, window: &mut Window, cx: &mut Context<Self>) {
        let Some(worktree_id) = worktree_id else {
            return;
        };
        
        // Get the relative path within the worktree
        let project = self.project.read(cx);
        let Some(worktree) = project.worktree_for_id(worktree_id, cx) else {
            return;
        };
        
        let worktree_root = worktree.read(cx).abs_path();
        let relative_path = path.strip_prefix(worktree_root.as_ref()).ok();
        
        let Some(relative_path) = relative_path else {
            return;
        };
        
        // Convert to RelPath
        let Ok(rel_path) = util::rel_path::RelPath::new(relative_path, util::paths::PathStyle::local()) else {
            return;
        };
        
        let project_path = ProjectPath {
            worktree_id,
            path: Arc::from(rel_path.as_ref()),
        };
        
        if let Some(workspace) = self.workspace.upgrade() {
            workspace.update(cx, |workspace, cx| {
                workspace.open_path(project_path, None, true, window, cx).detach_and_log_err(cx);
            });
        }
    }

    fn path_to_module_name(path: &Path, worktree_root: &Path) -> Option<String> {
        let relative = path.strip_prefix(worktree_root).ok()?;
        let module_path = relative.with_extension("");
        
        let components: Vec<_> = module_path
            .components()
            .filter_map(|c| c.as_os_str().to_str())
            .collect();
        
        if components.is_empty() {
            return None;
        }
        
        Some(components.join("."))
    }
    
    fn set_filter(&mut self, pattern: String, cx: &mut Context<Self>) {
        self.filter_pattern = pattern;
        self.compiled_filter = if self.filter_pattern.is_empty() {
            None
        } else {
            Regex::new(&self.filter_pattern).ok()
        };
        self.initialize_layout();
        self.start_simulation(cx);
        cx.notify();
    }
    
    fn get_filtered_node_count(&self) -> usize {
        if let Some(ref regex) = self.compiled_filter {
            self.graph.nodes.keys()
                .filter(|path| {
                    let path_str = path.to_string_lossy();
                    regex.is_match(&path_str)
                })
                .count()
        } else {
            self.graph.nodes.len()
        }
    }
    
    fn node_matches_filter(&self, path: &Path) -> bool {
        if let Some(ref regex) = self.compiled_filter {
            let path_str = path.to_string_lossy();
            regex.is_match(&path_str)
        } else {
            true
        }
    }
    
    fn handle_filter_input(&mut self, _action: &UpdateFilter, _window: &mut Window, _cx: &mut Context<Self>) {
        // This action is a placeholder - we'll use a different input mechanism
    }
    
    fn parse_imports(content: &str, language: Language) -> Vec<String> {
        match language {
            Language::Python => Self::parse_python_imports(content),
            Language::JavaScript | Language::TypeScript => Self::parse_js_ts_imports(content),
            Language::Rust => Self::parse_rust_imports(content),
            Language::Go => Self::parse_go_imports(content),
            Language::C | Language::Cpp => Self::parse_c_cpp_imports(content),
            Language::CSharp => Self::parse_csharp_imports(content),
            Language::Java | Language::Kotlin | Language::Scala => Self::parse_java_imports(content),
            Language::Ruby => Self::parse_ruby_imports(content),
            Language::Lua => Self::parse_lua_imports(content),
            Language::Swift => Self::parse_swift_imports(content),
            Language::Elixir => Self::parse_elixir_imports(content),
            Language::Css | Language::TailwindCss => Self::parse_css_imports(content),
            Language::Html => Self::parse_html_imports(content),
            Language::Vue => Self::parse_vue_imports(content),
            Language::Json => Self::parse_json_imports(content),
            Language::Yaml => Self::parse_yaml_imports(content),
            Language::Toml => Self::parse_toml_imports(content),
            Language::Markdown => Self::parse_markdown_imports(content),
            Language::GdScript => Self::parse_gdscript_imports(content),
            Language::R => Self::parse_r_imports(content),
            Language::Julia => Self::parse_julia_imports(content),
            Language::Terraform => Self::parse_terraform_imports(content),
            Language::Xml => Self::parse_xml_imports(content),
            Language::Diff | Language::Rego | Language::ReStructuredText => Vec::new(),
        }
    }
    
    fn parse_python_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // Handle "import module" statements
            if let Some(rest) = trimmed.strip_prefix("import ") {
                let module = rest.split_whitespace()
                    .next()
                    .unwrap_or("")
                    .split(',')
                    .next()
                    .unwrap_or("")
                    .split(" as ")
                    .next()
                    .unwrap_or("")
                    .trim();
                
                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
            
            // Handle "from module import ..." statements
            if let Some(rest) = trimmed.strip_prefix("from ") {
                if let Some(module) = rest.split(" import ").next() {
                    let module = module.trim();
                    if !module.is_empty() && !module.starts_with('.') {
                        imports.push(module.to_string());
                    }
                }
            }
        }
        
        imports
    }
    
    fn parse_js_ts_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // import ... from 'module'
            if trimmed.starts_with("import ") {
                if let Some(from_idx) = trimmed.find(" from ") {
                    let rest = &trimmed[from_idx + 7..];
                    let module = rest.trim_matches(|c| c == '\'' || c == '"' || c == ';' || c == ' ');
                    if !module.is_empty() && !module.starts_with('.') {
                        imports.push(module.to_string());
                    }
                }
            }
            
            // require('module')
            if let Some(start) = trimmed.find("require(") {
                let rest = &trimmed[start + 8..];
                if let Some(end) = rest.find(')') {
                    let module = rest[..end].trim_matches(|c| c == '\'' || c == '"');
                    if !module.is_empty() && !module.starts_with('.') {
                        imports.push(module.to_string());
                    }
                }
            }
            
            // export ... from 'module'
            if trimmed.starts_with("export ") && trimmed.contains(" from ") {
                if let Some(from_idx) = trimmed.find(" from ") {
                    let rest = &trimmed[from_idx + 7..];
                    let module = rest.trim_matches(|c| c == '\'' || c == '"' || c == ';' || c == ' ');
                    if !module.is_empty() && !module.starts_with('.') {
                        imports.push(module.to_string());
                    }
                }
            }
        }
        
        imports
    }
    
    fn parse_rust_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // use crate::module or use module::...
            if let Some(rest) = trimmed.strip_prefix("use ") {
                let module = rest.split("::")
                    .next()
                    .unwrap_or("")
                    .trim_matches(';')
                    .trim();
                
                if !module.is_empty() && module != "std" && module != "core" && module != "alloc" {
                    imports.push(module.to_string());
                }
            }
            
            // mod module;
            if let Some(rest) = trimmed.strip_prefix("mod ") {
                let module = rest.trim_matches(|c| c == ';' || c == ' ' || c == '{');
                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
        }
        
        imports
    }
    
    fn parse_go_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        let mut in_import_block = false;
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // import "package"
            if let Some(rest) = trimmed.strip_prefix("import ") {
                if rest.starts_with('(') {
                    in_import_block = true;
                    continue;
                }
                let module = rest.trim_matches(|c| c == '"' || c == ' ');
                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
            
            if in_import_block {
                if trimmed == ")" {
                    in_import_block = false;
                    continue;
                }
                let module = trimmed
                    .split_whitespace()
                    .last()
                    .unwrap_or("")
                    .trim_matches('"');
                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
        }
        
        imports
    }
    
    fn parse_c_cpp_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // #include <header> or #include "header"
            if let Some(rest) = trimmed.strip_prefix("#include") {
                let rest = rest.trim();
                if rest.starts_with('<') {
                    if let Some(end) = rest.find('>') {
                        let header = &rest[1..end];
                        imports.push(header.to_string());
                    }
                } else if rest.starts_with('"') {
                    if let Some(end) = rest[1..].find('"') {
                        let header = &rest[1..end + 1];
                        imports.push(header.to_string());
                    }
                }
            }
        }
        
        imports
    }
    
    fn parse_csharp_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // using namespace;
            if let Some(rest) = trimmed.strip_prefix("using ") {
                if !rest.starts_with('(') {
                    let namespace = rest.trim_matches(|c| c == ';' || c == ' ');
                    if !namespace.is_empty() && !namespace.starts_with("static ") {
                        imports.push(namespace.to_string());
                    }
                }
            }
        }
        
        imports
    }
    
    fn parse_java_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // import package.Class;
            if let Some(rest) = trimmed.strip_prefix("import ") {
                let import = rest.trim_start_matches("static ")
                    .trim_matches(|c| c == ';' || c == ' ');
                if !import.is_empty() {
                    imports.push(import.to_string());
                }
            }
        }
        
        imports
    }
    
    fn parse_ruby_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // require 'module' or require "module"
            if let Some(rest) = trimmed.strip_prefix("require ") {
                let module = rest.trim_matches(|c| c == '\'' || c == '"' || c == ' ');
                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
            
            // require_relative 'module'
            if let Some(rest) = trimmed.strip_prefix("require_relative ") {
                let module = rest.trim_matches(|c| c == '\'' || c == '"' || c == ' ');
                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
            
            // load 'file'
            if let Some(rest) = trimmed.strip_prefix("load ") {
                let module = rest.trim_matches(|c| c == '\'' || c == '"' || c == ' ');
                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
        }
        
        imports
    }
    
    fn parse_lua_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // require("module") or require "module"
            if let Some(start) = trimmed.find("require") {
                let rest = &trimmed[start + 7..];
                let rest = rest.trim_start_matches(|c| c == '(' || c == ' ');
                if let Some(end) = rest.find(|c| c == ')' || c == '\'' || c == '"') {
                    let module = rest[..end].trim_matches(|c| c == '\'' || c == '"');
                    if !module.is_empty() {
                        imports.push(module.to_string());
                    }
                } else {
                    let module = rest.trim_matches(|c| c == '\'' || c == '"' || c == ')');
                    if !module.is_empty() {
                        imports.push(module.to_string());
                    }
                }
            }
            
            // dofile("file")
            if let Some(start) = trimmed.find("dofile") {
                let rest = &trimmed[start + 6..];
                let rest = rest.trim_start_matches(|c| c == '(' || c == ' ');
                let module = rest.trim_matches(|c| c == '\'' || c == '"' || c == ')');
                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
        }
        
        imports
    }
    
    fn parse_swift_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // import Module
            if let Some(rest) = trimmed.strip_prefix("import ") {
                let module = rest.split_whitespace()
                    .next()
                    .unwrap_or("")
                    .trim();
                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
        }
        
        imports
    }
    
    fn parse_elixir_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // import Module
            if let Some(rest) = trimmed.strip_prefix("import ") {
                let module = rest.split(',')
                    .next()
                    .unwrap_or("")
                    .trim();
                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
            
            // alias Module
            if let Some(rest) = trimmed.strip_prefix("alias ") {
                let module = rest.split(',')
                    .next()
                    .unwrap_or("")
                    .trim();
                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
            
            // use Module
            if let Some(rest) = trimmed.strip_prefix("use ") {
                let module = rest.split(',')
                    .next()
                    .unwrap_or("")
                    .trim();
                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
            
            // require Module
            if let Some(rest) = trimmed.strip_prefix("require ") {
                let module = rest.split(',')
                    .next()
                    .unwrap_or("")
                    .trim();
                if !module.is_empty() {
                    imports.push(module.to_string());
                }
            }
        }
        
        imports
    }
    
    fn parse_css_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // @import url("file") or @import "file"
            if let Some(rest) = trimmed.strip_prefix("@import ") {
                let rest = rest.trim_start_matches("url(");
                let file = rest.trim_matches(|c| c == '\'' || c == '"' || c == ')' || c == ';' || c == ' ');
                if !file.is_empty() {
                    imports.push(file.to_string());
                }
            }
        }
        
        imports
    }
    
    fn parse_html_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        // Simple regex-like matching for src and href attributes
        for line in content.lines() {
            // script src="..."
            if let Some(start) = line.find("src=") {
                let rest = &line[start + 4..];
                let quote = rest.chars().next().unwrap_or(' ');
                if quote == '"' || quote == '\'' {
                    if let Some(end) = rest[1..].find(quote) {
                        let src = &rest[1..end + 1];
                        if !src.is_empty() {
                            imports.push(src.to_string());
                        }
                    }
                }
            }
            
            // link href="..." (for CSS)
            if line.contains("rel=") && line.contains("stylesheet") {
                if let Some(start) = line.find("href=") {
                    let rest = &line[start + 5..];
                    let quote = rest.chars().next().unwrap_or(' ');
                    if quote == '"' || quote == '\'' {
                        if let Some(end) = rest[1..].find(quote) {
                            let href = &rest[1..end + 1];
                            if !href.is_empty() {
                                imports.push(href.to_string());
                            }
                        }
                    }
                }
            }
        }
        
        imports
    }
    
    fn parse_vue_imports(content: &str) -> Vec<String> {
        // Vue files contain both JS imports and template references
        let mut imports = Self::parse_js_ts_imports(content);
        
        // Also check for component imports in template
        for line in content.lines() {
            // <ComponentName /> or <component-name>
            if let Some(start) = line.find('<') {
                let rest = &line[start + 1..];
                if let Some(end) = rest.find(|c: char| c.is_whitespace() || c == '>' || c == '/') {
                    let tag = &rest[..end];
                    // PascalCase components are likely imports
                    if !tag.is_empty() && tag.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        if !["DOCTYPE", "HTML", "HEAD", "BODY", "DIV", "SPAN", "SCRIPT", "STYLE", "TEMPLATE"].contains(&tag.to_uppercase().as_str()) {
                            imports.push(tag.to_string());
                        }
                    }
                }
            }
        }
        
        imports
    }
    
    fn parse_json_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        // Look for $ref in JSON Schema or OpenAPI
        for line in content.lines() {
            if let Some(start) = line.find("\"$ref\"") {
                let rest = &line[start + 6..];
                if let Some(colon) = rest.find(':') {
                    let rest = &rest[colon + 1..];
                    let rest = rest.trim();
                    if rest.starts_with('"') {
                        if let Some(end) = rest[1..].find('"') {
                            let reference = &rest[1..end + 1];
                            if !reference.is_empty() {
                                imports.push(reference.to_string());
                            }
                        }
                    }
                }
            }
        }
        
        imports
    }
    
    fn parse_yaml_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            // !include file.yaml
            if let Some(rest) = line.trim().strip_prefix("!include ") {
                let file = rest.trim();
                if !file.is_empty() {
                    imports.push(file.to_string());
                }
            }
            
            // $ref: "file.yaml"
            if let Some(start) = line.find("$ref:") {
                let rest = &line[start + 5..];
                let reference = rest.trim().trim_matches(|c| c == '\'' || c == '"');
                if !reference.is_empty() {
                    imports.push(reference.to_string());
                }
            }
        }
        
        imports
    }
    
    fn parse_toml_imports(_content: &str) -> Vec<String> {
        // TOML doesn't have a standard import mechanism
        // Could potentially look for path references
        Vec::new()
    }
    
    fn parse_markdown_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            // [text](link.md) - markdown links to other md files
            let mut rest = line;
            while let Some(start) = rest.find("](") {
                let after = &rest[start + 2..];
                if let Some(end_idx) = after.find(')') {
                    let link = &after[..end_idx];
                    if link.ends_with(".md") || link.ends_with(".markdown") {
                        imports.push(link.to_string());
                    }
                    rest = &after[end_idx.min(after.len())..];
                } else {
                    break;
                }
            }
            
            // ![alt](image.png) - image references
            if let Some(start) = line.find("![") {
                if let Some(paren) = line[start..].find("](") {
                    let after = &line[start + paren + 2..];
                    if let Some(end) = after.find(')') {
                        let link = &after[..end];
                        imports.push(link.to_string());
                    }
                }
            }
        }
        
        imports
    }
    
    fn parse_gdscript_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // preload("res://path/to/script.gd")
            if let Some(start) = trimmed.find("preload(") {
                let rest = &trimmed[start + 8..];
                if let Some(end) = rest.find(')') {
                    let path = rest[..end].trim_matches(|c| c == '\'' || c == '"');
                    if !path.is_empty() {
                        imports.push(path.to_string());
                    }
                }
            }
            
            // load("res://path/to/resource")
            if let Some(start) = trimmed.find("load(") {
                let rest = &trimmed[start + 5..];
                if let Some(end) = rest.find(')') {
                    let path = rest[..end].trim_matches(|c| c == '\'' || c == '"');
                    if !path.is_empty() {
                        imports.push(path.to_string());
                    }
                }
            }
            
            // extends "res://path/to/script.gd" or extends ClassName
            if let Some(rest) = trimmed.strip_prefix("extends ") {
                let class = rest.trim().trim_matches('"');
                if !class.is_empty() {
                    imports.push(class.to_string());
                }
            }
        }
        
        imports
    }
    
    fn parse_r_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // library(package)
            if let Some(start) = trimmed.find("library(") {
                let rest = &trimmed[start + 8..];
                if let Some(end) = rest.find(')') {
                    let package = rest[..end].trim_matches(|c| c == '\'' || c == '"');
                    if !package.is_empty() {
                        imports.push(package.to_string());
                    }
                }
            }
            
            // require(package)
            if let Some(start) = trimmed.find("require(") {
                let rest = &trimmed[start + 8..];
                if let Some(end) = rest.find(')') {
                    let package = rest[..end].trim_matches(|c| c == '\'' || c == '"');
                    if !package.is_empty() {
                        imports.push(package.to_string());
                    }
                }
            }
            
            // source("file.R")
            if let Some(start) = trimmed.find("source(") {
                let rest = &trimmed[start + 7..];
                if let Some(end) = rest.find(')') {
                    let file = rest[..end].trim_matches(|c| c == '\'' || c == '"');
                    if !file.is_empty() {
                        imports.push(file.to_string());
                    }
                }
            }
        }
        
        imports
    }
    
    fn parse_julia_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // using Module
            if let Some(rest) = trimmed.strip_prefix("using ") {
                for module in rest.split(',') {
                    let module = module.split(':').next().unwrap_or("").trim();
                    if !module.is_empty() {
                        imports.push(module.to_string());
                    }
                }
            }
            
            // import Module
            if let Some(rest) = trimmed.strip_prefix("import ") {
                for module in rest.split(',') {
                    let module = module.split(':').next().unwrap_or("").trim();
                    if !module.is_empty() {
                        imports.push(module.to_string());
                    }
                }
            }
            
            // include("file.jl")
            if let Some(start) = trimmed.find("include(") {
                let rest = &trimmed[start + 8..];
                if let Some(end) = rest.find(')') {
                    let file = rest[..end].trim_matches(|c| c == '\'' || c == '"');
                    if !file.is_empty() {
                        imports.push(file.to_string());
                    }
                }
            }
        }
        
        imports
    }
    
    fn parse_terraform_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            // source = "path/to/module"
            if let Some(rest) = trimmed.strip_prefix("source") {
                let rest = rest.trim().strip_prefix('=').unwrap_or(rest).trim();
                let source = rest.trim_matches(|c| c == '"' || c == '\'');
                if !source.is_empty() {
                    imports.push(source.to_string());
                }
            }
            
            // module "name" { source = "..." }
            if trimmed.starts_with("module ") {
                // Will be caught by source = above
            }
        }
        
        imports
    }
    
    fn parse_xml_imports(content: &str) -> Vec<String> {
        let mut imports = Vec::new();
        
        for line in content.lines() {
            // xsi:schemaLocation or schemaLocation
            if line.contains("schemaLocation") {
                if let Some(start) = line.find("schemaLocation") {
                    let rest = &line[start..];
                    if let Some(eq) = rest.find('=') {
                        let rest = &rest[eq + 1..];
                        let rest = rest.trim().trim_start_matches('"');
                        if let Some(end) = rest.find('"') {
                            let schema = &rest[..end];
                            for part in schema.split_whitespace() {
                                if part.ends_with(".xsd") {
                                    imports.push(part.to_string());
                                }
                            }
                        }
                    }
                }
            }
            
            // <xi:include href="file.xml" />
            if line.contains("xi:include") || line.contains("xinclude") {
                if let Some(start) = line.find("href=") {
                    let rest = &line[start + 5..];
                    let quote = rest.chars().next().unwrap_or(' ');
                    if quote == '"' || quote == '\'' {
                        if let Some(end) = rest[1..].find(quote) {
                            let href = &rest[1..end + 1];
                            imports.push(href.to_string());
                        }
                    }
                }
            }
        }
        
        imports
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Language {
    Python,
    JavaScript,
    TypeScript,
    Rust,
    Go,
    C,
    Cpp,
    CSharp,
    Java,
    Kotlin,
    Scala,
    Ruby,
    Lua,
    Swift,
    Elixir,
    Css,
    TailwindCss,
    Html,
    Vue,
    Json,
    Yaml,
    Toml,
    Markdown,
    GdScript,
    R,
    Julia,
    Terraform,
    Xml,
    Diff,
    Rego,
    ReStructuredText,
}

impl Language {
    fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "py" | "pyw" | "pyi" => Some(Language::Python),
            "js" | "mjs" | "cjs" | "jsx" => Some(Language::JavaScript),
            "ts" | "tsx" | "mts" | "cts" => Some(Language::TypeScript),
            "rs" => Some(Language::Rust),
            "go" => Some(Language::Go),
            "c" | "h" => Some(Language::C),
            "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" | "c++" | "h++" => Some(Language::Cpp),
            "cs" => Some(Language::CSharp),
            "java" => Some(Language::Java),
            "kt" | "kts" => Some(Language::Kotlin),
            "scala" | "sc" => Some(Language::Scala),
            "rb" | "rake" | "gemspec" => Some(Language::Ruby),
            "lua" => Some(Language::Lua),
            "swift" => Some(Language::Swift),
            "ex" | "exs" => Some(Language::Elixir),
            "css" | "scss" | "sass" | "less" => Some(Language::Css),
            "html" | "htm" => Some(Language::Html),
            "vue" => Some(Language::Vue),
            "json" | "jsonc" => Some(Language::Json),
            "yaml" | "yml" => Some(Language::Yaml),
            "toml" => Some(Language::Toml),
            "md" | "markdown" => Some(Language::Markdown),
            "gd" => Some(Language::GdScript),
            "r" => Some(Language::R),
            "jl" => Some(Language::Julia),
            "tf" | "tfvars" => Some(Language::Terraform),
            "xml" | "xsd" | "xsl" | "xslt" | "svg" => Some(Language::Xml),
            "diff" | "patch" => Some(Language::Diff),
            "rego" => Some(Language::Rego),
            "rst" => Some(Language::ReStructuredText),
            _ => None,
        }
    }
    
    fn file_extension_patterns() -> Vec<&'static str> {
        vec![
            "py", "pyw", "pyi",
            "js", "mjs", "cjs", "jsx",
            "ts", "tsx", "mts", "cts",
            "rs",
            "go",
            "c", "h",
            "cpp", "cc", "cxx", "hpp", "hh", "hxx",
            "cs",
            "java",
            "kt", "kts",
            "scala", "sc",
            "rb", "rake", "gemspec",
            "lua",
            "swift",
            "ex", "exs",
            "css", "scss", "sass", "less",
            "html", "htm",
            "vue",
            "json", "jsonc",
            "yaml", "yml",
            "toml",
            "md", "markdown",
            "gd",
            "r",
            "jl",
            "tf", "tfvars",
            "xml", "xsd", "xsl", "xslt", "svg",
            "diff", "patch",
            "rego",
            "rst",
        ]
    }
}

impl EventEmitter<PanelEvent> for GraphView {}

impl Focusable for GraphView {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Panel for GraphView {
    fn persistent_name() -> &'static str {
        "Graph View"
    }

    fn panel_key() -> &'static str {
        GRAPH_VIEW_KEY
    }

    fn position(&self, _: &Window, cx: &App) -> DockPosition {
        match GraphViewSettings::get_global(cx).dock {
            DockSide::Left => DockPosition::Left,
            DockSide::Right => DockPosition::Right,
        }
    }

    fn position_is_valid(&self, position: DockPosition) -> bool {
        matches!(position, DockPosition::Left | DockPosition::Right)
    }

    fn set_position(&mut self, _position: DockPosition, _: &mut Window, _cx: &mut Context<Self>) {
        // Panel position settings are not persisted for now
        // TODO: Add graph_view settings to SettingsContent
    }

    fn size(&self, _: &Window, cx: &App) -> Pixels {
        self.width
            .unwrap_or_else(|| GraphViewSettings::get_global(cx).default_width)
    }

    fn set_size(&mut self, size: Option<Pixels>, window: &mut Window, cx: &mut Context<Self>) {
        self.width = size;
        cx.notify();
        cx.defer_in(window, |this, _, cx| {
            this.serialize(cx);
        });
    }

    fn icon(&self, _: &Window, cx: &App) -> Option<IconName> {
        GraphViewSettings::get_global(cx)
            .button
            .then_some(IconName::GitBranch)
    }

    fn icon_tooltip(&self, _: &Window, _: &App) -> Option<&'static str> {
        Some("Graph View")
    }

    fn toggle_action(&self) -> Box<dyn Action> {
        Box::new(ToggleFocus)
    }

    fn starts_open(&self, _: &Window, _: &App) -> bool {
        false
    }

    fn set_active(&mut self, _active: bool, _: &mut Window, _cx: &mut Context<Self>) {}

    fn activation_priority(&self) -> u32 {
        4
    }
}

impl GraphView {
    fn render_file_node(
        node: &FileNode,
        incoming_edges: &HashMap<&Path, Vec<&ImportEdge>>,
    ) -> impl IntoElement {
        let file_name = node.path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        let imports_count = node.imports.len();
        let imported_by_count = incoming_edges
            .get(node.path.as_path())
            .map(|edges| edges.len())
            .unwrap_or(0);
        
        let imports = node.imports.clone();
        
        v_flex()
            .gap_1()
            .child(
                div()
                    .p_2()
                    .rounded_md()
                    .hover(|style| style.bg(gpui::black().opacity(0.05)))
                    .child(
                        v_flex()
                            .gap_1()
                            .child(
                                h_flex()
                                    .gap_2()
                                    .child(
                                        Label::new(file_name)
                                            .color(Color::Default),
                                    )
                                    .child(
                                        Label::new(format!("→{} ←{}", imports_count, imported_by_count))
                                            .size(LabelSize::XSmall)
                                            .color(Color::Muted),
                                    ),
                            )
                            .when(imports_count > 0, |this| {
                                this.child(
                                    div()
                                        .pl_2()
                                        .child(
                                            v_flex()
                                                .gap_px()
                                                .children(imports.iter().take(5).map(|import| {
                                                    Label::new(format!("  imports: {}", import))
                                                        .size(LabelSize::XSmall)
                                                        .color(Color::Muted)
                                                }))
                                                .when(imports.len() > 5, |this| {
                                                    this.child(
                                                        Label::new(format!("  ... and {} more", imports.len() - 5))
                                                            .size(LabelSize::XSmall)
                                                            .color(Color::Muted)
                                                    )
                                                }),
                                        ),
                                )
                            }),
                    ),
            )
    }
}

impl GraphView {
    fn render_graph(&self, cx: &Context<Self>) -> impl IntoElement {
        let edges = self.graph.edges.clone();
        let nodes = self.layout.nodes.clone();
        let nodes_for_canvas = nodes.clone();
        let nodes_for_labels = nodes.clone();
        let directories = self.layout.directories.clone();
        let directories_for_labels = self.layout.directories.clone();
        let theme_colors = cx.theme().colors();
        
        // Calculate canvas bounds - scale with node count for big projects
        let node_count = self.layout.nodes.len();
        let base_canvas_size = 1200.0;
        let size_per_node = 80.0;
        let min_canvas_size = base_canvas_size + (node_count as f32).sqrt() * size_per_node * 10.0;
        
        let mut max_x = 0.0f32;
        let mut max_y = 0.0f32;
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        for node in &self.layout.nodes {
            max_x = max_x.max(node.position.x + 150.0);
            max_y = max_y.max(node.position.y + 150.0);
            min_x = min_x.min(node.position.x - 50.0);
            min_y = min_y.min(node.position.y - 50.0);
        }
        let canvas_width = (max_x - min_x.min(0.0)).max(min_canvas_size);
        let canvas_height = (max_y - min_y.min(0.0)).max(min_canvas_size);
        
        let edge_color: Hsla = Hsla::from(rgb(0x5d3d3a));  // Cocoa color for edges
        let arrow_color: Hsla = Hsla::from(rgb(0x5d3d3a));  // Cocoa color for arrows
        let node_fill: Hsla = Hsla::from(rgb(0xdd5013));  // Sunset-orange for nodes
        let node_stroke: Hsla = Hsla::from(rgb(0xdd5013)).opacity(0.8);  // Sunset-orange for node stroke
        let text_color = theme_colors.text;
        let lasso_color: Hsla = Hsla::from(rgb(0xbdb7fc)).opacity(0.15);  // Lavender for directory lassos
        let lasso_stroke: Hsla = Hsla::from(rgb(0xbdb7fc)).opacity(0.5);  // Lavender for lasso stroke
        
        let dragging = self.dragging_node.is_some();
        
        div()
            .id("graph-canvas-inner")
            .relative()
            .w(px(canvas_width))
            .h(px(canvas_height))
            .on_mouse_move(cx.listener(move |this, event: &MouseMoveEvent, _window, cx| {
                if let Some(node_idx) = this.dragging_node {
                    let pos_x: f32 = event.position.x.into();
                    let pos_y: f32 = event.position.y.into();
                    
                    // Check if we've moved enough to count as a drag
                    let dx = pos_x - this.drag_start_pos.x;
                    let dy = pos_y - this.drag_start_pos.y;
                    if dx * dx + dy * dy > 25.0 {
                        this.has_dragged = true;
                    }
                    
                    if let Some(node) = this.layout.nodes.get_mut(node_idx) {
                        node.position = Point {
                            x: pos_x - this.drag_offset.x,
                            y: pos_y - this.drag_offset.y,
                        };
                        node.velocity = Point { x: 0.0, y: 0.0 };
                    }
                    cx.notify();
                }
            }))
            .on_mouse_up(MouseButton::Left, cx.listener(move |this, _event: &MouseUpEvent, window, cx| {
                if let Some(node_idx) = this.dragging_node {
                    let was_click = !this.has_dragged;
                    
                    if let Some(node) = this.layout.nodes.get_mut(node_idx) {
                        node.is_pinned = false;
                    }
                    
                    // Open file if it was a click (not a drag)
                    if was_click {
                        if let Some(node_layout) = this.layout.nodes.get(node_idx) {
                            let path = node_layout.path.clone();
                            let worktree_id = node_layout.worktree_id;
                            this.open_file(path, worktree_id, window, cx);
                        }
                    }
                    
                    this.dragging_node = None;
                    this.has_dragged = false;
                    this.layout.is_running = true;
                    this.start_simulation(cx);
                    cx.notify();
                }
            }))
            .when(dragging, |div| div.cursor_grabbing())
            .child(
                canvas(
                    move |_bounds, _window, _cx| {
                        // Prepaint: return data needed for painting
                        (edges.clone(), nodes_for_canvas.clone(), directories.clone(), edge_color, arrow_color, node_fill, node_stroke, text_color, lasso_color, lasso_stroke)
                    },
                    move |bounds, (edges, nodes, directories, edge_color, arrow_color, node_fill, node_stroke, _text_color, lasso_color, lasso_stroke), window, _cx| {
                        // Draw directory lassos using convex hull with bezier curves
                        for dir_group in directories.values() {
                            if dir_group.node_indices.len() < 2 {
                                continue;
                            }
                            
                            // Collect node positions for this directory
                            let node_points: Vec<Point<f32>> = dir_group.node_indices.iter()
                                .filter_map(|&idx| nodes.get(idx))
                                .map(|node| Point { x: node.position.x, y: node.position.y })
                                .collect();
                            
                            if node_points.len() < 2 {
                                continue;
                            }
                            
                            let centroid = dir_group.centroid;
                            let padding = 70.0f32;  // Increased padding to contain nodes
                            
                            // Compute convex hull
                            let hull = GraphView::convex_hull(&node_points);
                            
                            if hull.len() < 2 {
                                continue;
                            }
                            
                            // Expand hull points outward from centroid by padding amount
                            let expanded_hull: Vec<Point<Pixels>> = hull.iter()
                                .map(|p| {
                                    let dx = p.x - centroid.x;
                                    let dy = p.y - centroid.y;
                                    let dist = (dx * dx + dy * dy).sqrt().max(1.0);
                                    let expand_x = p.x + (dx / dist) * padding;
                                    let expand_y = p.y + (dy / dist) * padding;
                                    point(
                                        bounds.origin.x + px(expand_x),
                                        bounds.origin.y + px(expand_y)
                                    )
                                })
                                .collect();
                            
                            if expanded_hull.len() < 2 {
                                continue;
                            }
                            
                            let hull_len = expanded_hull.len();
                            
                            if hull_len >= 3 {
                                // Draw filled lasso with bezier curves
                                let mut builder = PathBuilder::fill();
                                
                                // Start at midpoint between last and first point
                                let first_mid = point(
                                    (expanded_hull[hull_len - 1].x + expanded_hull[0].x) / 2.0,
                                    (expanded_hull[hull_len - 1].y + expanded_hull[0].y) / 2.0
                                );
                                builder.move_to(first_mid);
                                
                                // Draw bezier curves: each hull point is a control point
                                // and we draw to the midpoint between consecutive hull points
                                for i in 0..hull_len {
                                    let ctrl = expanded_hull[i];
                                    let next = expanded_hull[(i + 1) % hull_len];
                                    let mid = point(
                                        (ctrl.x + next.x) / 2.0,
                                        (ctrl.y + next.y) / 2.0
                                    );
                                    
                                    // curve_to(to, ctrl) - draws quadratic bezier
                                    builder.curve_to(mid, ctrl);
                                }
                                
                                builder.close();
                                
                                if let Ok(path) = builder.build() {
                                    window.paint_path(path, lasso_color);
                                }
                                
                                // Draw lasso outline with bezier curves
                                let mut builder = PathBuilder::stroke(px(2.0));
                                
                                let first_mid = point(
                                    (expanded_hull[hull_len - 1].x + expanded_hull[0].x) / 2.0,
                                    (expanded_hull[hull_len - 1].y + expanded_hull[0].y) / 2.0
                                );
                                builder.move_to(first_mid);
                                
                                for i in 0..hull_len {
                                    let ctrl = expanded_hull[i];
                                    let next = expanded_hull[(i + 1) % hull_len];
                                    let mid = point(
                                        (ctrl.x + next.x) / 2.0,
                                        (ctrl.y + next.y) / 2.0
                                    );
                                    
                                    builder.curve_to(mid, ctrl);
                                }
                                
                                builder.close();
                                
                                if let Ok(path) = builder.build() {
                                    window.paint_path(path, lasso_stroke);
                                }
                            } else {
                                // Fallback for 2 or fewer points: draw a circle
                                let centroid_px = point(
                                    bounds.origin.x + px(centroid.x),
                                    bounds.origin.y + px(centroid.y)
                                );
                                
                                // Calculate radius from centroid to furthest node + padding
                                let max_dist = node_points.iter()
                                    .map(|p| {
                                        let dx = p.x - centroid.x;
                                        let dy = p.y - centroid.y;
                                        (dx * dx + dy * dy).sqrt()
                                    })
                                    .fold(0.0f32, f32::max);
                                let radius = px(max_dist + padding);
                                
                                // Draw filled circle
                                let mut builder = PathBuilder::fill();
                                let segments = 32;
                                for i in 0..=segments {
                                    let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
                                    let x = centroid_px.x + radius * angle.cos();
                                    let y = centroid_px.y + radius * angle.sin();
                                    let p = point(x, y);
                                    
                                    if i == 0 {
                                        builder.move_to(p);
                                    } else {
                                        builder.line_to(p);
                                    }
                                }
                                builder.close();
                                
                                if let Ok(path) = builder.build() {
                                    window.paint_path(path, lasso_color);
                                }
                                
                                // Draw circle outline
                                let mut builder = PathBuilder::stroke(px(2.0));
                                for i in 0..=segments {
                                    let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
                                    let x = centroid_px.x + radius * angle.cos();
                                    let y = centroid_px.y + radius * angle.sin();
                                    let p = point(x, y);
                                    
                                    if i == 0 {
                                        builder.move_to(p);
                                    } else {
                                        builder.line_to(p);
                                    }
                                }
                                builder.close();
                                
                                if let Ok(path) = builder.build() {
                                    window.paint_path(path, lasso_stroke);
                                }
                            }
                        }
                        
                        // Draw edges with arrows (from imported file -> importing file)
                        let node_radius = 25.0f32;
                        for edge in &edges {
                            if let (Some(importer_node), Some(imported_node)) = (
                                nodes.iter().find(|n| n.path == edge.from),
                                nodes.iter().find(|n| n.path == edge.to),
                            ) {
                                // Arrow goes FROM imported TO importer (shows "is used by")
                                let start_pos = point(
                                    bounds.origin.x + px(imported_node.position.x),
                                    bounds.origin.y + px(imported_node.position.y)
                                );
                                let end_pos = point(
                                    bounds.origin.x + px(importer_node.position.x),
                                    bounds.origin.y + px(importer_node.position.y)
                                );
                                
                                // Calculate direction
                                let start_x: f32 = start_pos.x.into();
                                let start_y: f32 = start_pos.y.into();
                                let end_x: f32 = end_pos.x.into();
                                let end_y: f32 = end_pos.y.into();
                                let dx = end_x - start_x;
                                let dy = end_y - start_y;
                                let dist = (dx * dx + dy * dy).sqrt();
                                
                                if dist > node_radius * 2.0 {
                                    let norm_x = dx / dist;
                                    let norm_y = dy / dist;
                                    
                                    // Shorten line to stop at node edges
                                    let line_start = point(
                                        px(start_x + norm_x * node_radius),
                                        px(start_y + norm_y * node_radius)
                                    );
                                    let line_end = point(
                                        px(end_x - norm_x * node_radius),
                                        px(end_y - norm_y * node_radius)
                                    );
                                    
                                    // Draw line
                                    let mut builder = PathBuilder::stroke(px(2.0));
                                    builder.move_to(line_start);
                                    builder.line_to(line_end);
                                    
                                    if let Ok(path) = builder.build() {
                                        window.paint_path(path, edge_color);
                                    }
                                    
                                    // Draw arrowhead at the end (pointing towards importer)
                                    let arrow_size = 10.0;
                                    let perp_x = -norm_y;
                                    let perp_y = norm_x;
                                    
                                    let tip_x: f32 = line_end.x.into();
                                    let tip_y: f32 = line_end.y.into();
                                    
                                    let arrow_tip = line_end;
                                    let arrow_left = point(
                                        px(tip_x - norm_x * arrow_size - perp_x * arrow_size * 0.5),
                                        px(tip_y - norm_y * arrow_size - perp_y * arrow_size * 0.5)
                                    );
                                    let arrow_right = point(
                                        px(tip_x - norm_x * arrow_size + perp_x * arrow_size * 0.5),
                                        px(tip_y - norm_y * arrow_size + perp_y * arrow_size * 0.5)
                                    );
                                    
                                    // Draw filled arrowhead
                                    let mut builder = PathBuilder::fill();
                                    builder.move_to(arrow_tip);
                                    builder.line_to(arrow_left);
                                    builder.line_to(arrow_right);
                                    builder.line_to(arrow_tip);
                                    
                                    if let Ok(path) = builder.build() {
                                        window.paint_path(path, arrow_color);
                                    }
                                }
                            }
                        }
                        
                        // Draw nodes
                        for node_layout in &nodes {
                            let center = point(
                                bounds.origin.x + px(node_layout.position.x),
                                bounds.origin.y + px(node_layout.position.y)
                            );
                            let radius = px(25.0);
                            
                            // Draw filled hexagon for node
                            let mut builder = PathBuilder::fill();
                            let sides = 6;
                            let angle_offset = std::f32::consts::PI / 6.0;  // Start with flat top
                            for i in 0..=sides {
                                let angle = angle_offset + (i as f32 / sides as f32) * std::f32::consts::TAU;
                                let x = center.x + radius * angle.cos();
                                let y = center.y + radius * angle.sin();
                                let p = point(x, y);
                                
                                if i == 0 {
                                    builder.move_to(p);
                                } else {
                                    builder.line_to(p);
                                }
                            }
                            builder.close();
                            
                            if let Ok(path) = builder.build() {
                                window.paint_path(path, node_fill);
                            }
                            
                            // Draw hexagon outline
                            let mut builder = PathBuilder::stroke(px(2.0));
                            for i in 0..=sides {
                                let angle = angle_offset + (i as f32 / sides as f32) * std::f32::consts::TAU;
                                let x = center.x + radius * angle.cos();
                                let y = center.y + radius * angle.sin();
                                let p = point(x, y);
                                
                                if i == 0 {
                                    builder.move_to(p);
                                } else {
                                    builder.line_to(p);
                                }
                            }
                            builder.close();
                            
                            if let Ok(path) = builder.build() {
                                window.paint_path(path, node_stroke);
                            }
                        }
                    },
                )
                .absolute()
                .size_full()
            )
            .children(nodes_for_labels.iter().enumerate().map(|(idx, node)| {
                let file_name = node.path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                
                let node_x = node.position.x;
                let node_y = node.position.y;
                
                div()
                    .id(ElementId::NamedInteger("graph-node".into(), idx as u64))
                    .absolute()
                    .top(px(node_y - 25.0))
                    .left(px(node_x - 25.0))
                    .w(px(50.0))
                    .h(px(50.0))
                    .cursor_grab()
                    .on_mouse_down(MouseButton::Left, cx.listener(move |this, event: &MouseDownEvent, _window, cx| {
                        let pos_x: f32 = event.position.x.into();
                        let pos_y: f32 = event.position.y.into();
                        
                        this.drag_offset = Point {
                            x: pos_x - node_x,
                            y: pos_y - node_y,
                        };
                        this.drag_start_pos = Point { x: pos_x, y: pos_y };
                        this.has_dragged = false;
                        this.dragging_node = Some(idx);
                        if let Some(node) = this.layout.nodes.get_mut(idx) {
                            node.is_pinned = true;
                        }
                        this.start_simulation(cx);
                        cx.notify();
                    }))
                    .child(
                        div()
                            .absolute()
                            .top(px(55.0))
                            .left(px(-15.0))
                            .w(px(80.0))
                            .flex()
                            .justify_center()
                            .child(
                                Label::new(file_name)
                                    .size(LabelSize::XSmall)
                                    .color(Color::Default)
                            )
                    )
            }))
            // Directory folder name labels
            .children(directories_for_labels.iter().filter_map(|(dir_path, dir_group)| {
                if dir_group.node_indices.len() < 2 {
                    return None;
                }
                
                let folder_name = dir_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("folder")
                    .to_string();
                
                let centroid = dir_group.centroid;
                
                // Find max distance from centroid for positioning the label (closer to lasso)
                let max_dist = dir_group.node_indices.iter()
                    .filter_map(|&idx| nodes_for_labels.get(idx))
                    .map(|node| {
                        let dx = node.position.x - centroid.x;
                        let dy = node.position.y - centroid.y;
                        (dx * dx + dy * dy).sqrt()
                    })
                    .fold(0.0f32, f32::max);
                
                // Position label just outside the lasso (padding + small offset)
                let label_offset = max_dist + 75.0;  // Closer to lasso edge
                
                Some(div()
                    .absolute()
                    .top(px(centroid.y - label_offset))
                    .left(px(centroid.x - 60.0))
                    .w(px(120.0))
                    .flex()
                    .justify_center()
                    .child(
                        div()
                            .px_2()
                            .py_0p5()
                            .rounded_md()
                            .bg(Hsla::from(rgb(0xbdb7fc)).opacity(0.3))  // Lavender background
                            .child(
                                Label::new(folder_name)
                                    .size(LabelSize::Small)
                                    .color(Color::Default)
                            )
                    ))
            }))
    }
}

impl Render for GraphView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let node_count = self.graph.nodes.len();
        let edge_count = self.graph.edges.len();
        let filtered_count = self.get_filtered_node_count();
        let filter_pattern = self.filter_pattern.clone();
        let has_filter_error = !self.filter_pattern.is_empty() && self.compiled_filter.is_none();

        v_flex()
            .id("graph-view")
            .size_full()
            .track_focus(&self.focus_handle)
            .key_context("GraphView")
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, _window, cx| {
                let key = &event.keystroke.key;
                
                // Handle backspace
                if key == "backspace" {
                    this.filter_pattern.pop();
                    this.compiled_filter = if this.filter_pattern.is_empty() {
                        None
                    } else {
                        Regex::new(&this.filter_pattern).ok()
                    };
                    this.initialize_layout();
                    this.start_simulation(cx);
                    cx.notify();
                    return;
                }
                
                // Handle escape to clear
                if key == "escape" {
                    this.filter_pattern.clear();
                    this.compiled_filter = None;
                    this.initialize_layout();
                    this.start_simulation(cx);
                    cx.notify();
                    return;
                }
                
                // Handle regular characters - use key_char if available, otherwise key
                if let Some(key_char) = &event.keystroke.key_char {
                    this.filter_pattern.push_str(key_char);
                    this.compiled_filter = Regex::new(&this.filter_pattern).ok();
                    this.initialize_layout();
                    this.start_simulation(cx);
                    cx.notify();
                } else if key.len() == 1 && !event.keystroke.modifiers.platform && !event.keystroke.modifiers.control {
                    // Single character key without modifiers
                    this.filter_pattern.push_str(key);
                    this.compiled_filter = Regex::new(&this.filter_pattern).ok();
                    this.initialize_layout();
                    this.start_simulation(cx);
                    cx.notify();
                }
            }))
            .bg(cx.theme().colors().panel_background)
            .child(
                div()
                    .p_4()
                    .border_b_1()
                    .border_color(cx.theme().colors().border)
                    .child(
                        v_flex()
                            .gap_2()
                            .child(
                                Label::new("Project Graph")
                                    .size(LabelSize::Large)
                                    .color(Color::Default),
                            )
                            .child(
                                h_flex()
                                    .gap_4()
                                    .child(
                                        Label::new(format!("Files: {}", if self.filter_pattern.is_empty() { node_count } else { filtered_count }))
                                            .size(LabelSize::Small)
                                            .color(Color::Muted),
                                    )
                                    .child(
                                        Label::new(format!("Total: {}", node_count))
                                            .size(LabelSize::Small)
                                            .color(Color::Muted),
                                    )
                                    .child(
                                        Label::new(format!("Imports: {}", edge_count))
                                            .size(LabelSize::Small)
                                            .color(Color::Muted),
                                    ),
                            )
                            .child(
                                h_flex()
                                    .gap_2()
                                    .w_full()
                                    .child(
                                        Label::new("Filter:")
                                            .size(LabelSize::Small)
                                            .color(Color::Muted),
                                    )
                                    .child(
                                        div()
                                            .flex_1()
                                            .px_2()
                                            .py_1()
                                            .rounded_md()
                                            .bg(cx.theme().colors().editor_background)
                                            .border_1()
                                            .border_color(if has_filter_error {
                                                cx.theme().status().error
                                            } else {
                                                cx.theme().colors().border
                                            })
                                            .child(
                                                div()
                                                    .id("filter-input")
                                                    .cursor_text()
                                                    .on_click(cx.listener(|this, _, window, cx| {
                                                        this.focus_handle.focus(window);
                                                        cx.notify();
                                                    }))
                                                    .child(
                                                        Label::new(if filter_pattern.is_empty() {
                                                            "regex pattern...".to_string()
                                                        } else {
                                                            filter_pattern
                                                        })
                                                        .size(LabelSize::Small)
                                                        .color(if self.filter_pattern.is_empty() {
                                                            Color::Muted
                                                        } else {
                                                            Color::Default
                                                        }),
                                                    )
                                            )
                                    )
                                    .when(!self.filter_pattern.is_empty(), |el| {
                                        el.child(
                                            div()
                                                .id("clear-filter")
                                                .cursor_pointer()
                                                .px_2()
                                                .py_1()
                                                .rounded_md()
                                                .bg(cx.theme().colors().element_background)
                                                .hover(|style| style.bg(cx.theme().colors().element_hover))
                                                .on_click(cx.listener(|this, _, _window, cx| {
                                                    this.filter_pattern.clear();
                                                    this.compiled_filter = None;
                                                    this.initialize_layout();
                                                    this.start_simulation(cx);
                                                    cx.notify();
                                                }))
                                                .child(
                                                    Label::new("Clear")
                                                        .size(LabelSize::Small)
                                                        .color(Color::Muted),
                                                )
                                        )
                                    }),
                            )
                            .when(has_filter_error, |el| {
                                el.child(
                                    Label::new("Invalid regex pattern")
                                        .size(LabelSize::XSmall)
                                        .color(Color::Error),
                                )
                            }),
                    ),
            )
            .child(
                div()
                    .id("graph-canvas")
                    .flex_1()
                    .overflow_scroll()
                    .child(self.render_graph(cx))
            )
    }
}