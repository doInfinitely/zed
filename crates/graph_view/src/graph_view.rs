use anyhow::{Context as _, Result};
use collections::HashMap;
use db::kvp::KEY_VALUE_STORE;
use gpui::{
    actions, Action, App, AsyncWindowContext, Context, Entity, EventEmitter,
    FocusHandle, Focusable, InteractiveElement, ParentElement, Pixels, Render, Styled,
    Subscription, Task, WeakEntity, Window, div, px,
};
use project::{Project, WorktreeId};
use serde::{Deserialize, Serialize};
use settings::{DockSide, RegisterSetting, Settings};
use std::path::{Path, PathBuf};
use ui::{prelude::*, IconName, Label};
use util::{ResultExt, TryFutureExt};
use workspace::{
    dock::{DockPosition, Panel, PanelEvent},
    Workspace,
};

const GRAPH_VIEW_KEY: &str = "GraphView";

actions!(graph_view, [ToggleFocus]);

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
    _subscriptions: Vec<Subscription>,
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
                _subscriptions: subscriptions,
            };
            this.update_graph_initial(cx);
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
        
        // First pass: collect all Python files
        for worktree in &worktrees {
            let worktree_id = worktree.read(cx).id();
            let snapshot = worktree.read(cx).snapshot();
            let worktree_root = snapshot.abs_path();
            
            for entry in snapshot.entries(false, 0) {
                if entry.is_file() {
                    if let Some(extension) = entry.path.extension() {
                        if extension == "py" {
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
            if let Ok(content) = std::fs::read_to_string(&path) {
                let imports = Self::parse_python_imports(&content);
                
                if let Some(node) = self.graph.nodes.get_mut(&path) {
                    node.imports = imports.clone();
                }
                
                // Build edges
                for import in imports {
                    if let Some(target_path) = self.graph.module_to_path.get(&import) {
                        self.graph.edges.push(ImportEdge {
                            from: path.clone(),
                            to: target_path.clone(),
                            import_name: import,
                        });
                    }
                }
            }
        }
    }

    fn update_graph(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        self.update_graph_initial(cx);
        cx.notify();
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

impl Render for GraphView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let node_count = self.graph.nodes.len();
        let edge_count = self.graph.edges.len();
        
        // Get nodes sorted by path for consistent display
        let mut nodes: Vec<_> = self.graph.nodes.values().collect();
        nodes.sort_by(|a, b| a.path.cmp(&b.path));
        
        // Build incoming edges map
        let mut incoming_edges: HashMap<&Path, Vec<&ImportEdge>> = HashMap::default();
        for edge in &self.graph.edges {
            incoming_edges.entry(edge.to.as_path())
                .or_insert_with(Vec::new)
                .push(edge);
        }

        v_flex()
            .id("graph-view")
            .size_full()
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
                                        Label::new(format!("Files: {}", node_count))
                                            .size(LabelSize::Small)
                                            .color(Color::Muted),
                                    )
                                    .child(
                                        Label::new(format!("Imports: {}", edge_count))
                                            .size(LabelSize::Small)
                                            .color(Color::Muted),
                                    ),
                            ),
                    ),
            )
            .child(
                div()
                    .id("graph-scroll")
                    .flex_1()
                    .overflow_y_scroll()
                    .child(
                        v_flex()
                            .p_2()
                            .gap_1()
                            .children(nodes.iter().map(|node| {
                                Self::render_file_node(node, &incoming_edges)
                            })),
                    ),
            )
    }
}