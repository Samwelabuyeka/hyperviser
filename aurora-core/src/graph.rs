//! Compute graph for operation fusion and optimization

use std::collections::{HashMap, HashSet};
use crate::tensor::Tensor;
use crate::kernel::KernelType;

/// Unique node identifier in a compute graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

impl NodeId {
    /// Create a new node ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Operation type for graph nodes
#[derive(Debug, Clone)]
pub enum OpType {
    /// Input tensor
    Input {
        /// Tensor reference
        tensor: Tensor,
    },
    /// Output tensor
    Output {
        /// Source node
        source: NodeId,
    },
    /// Kernel operation
    Kernel {
        /// Kernel type
        kernel_type: KernelType,
        /// Input nodes
        inputs: Vec<NodeId>,
        /// Attributes
        attrs: HashMap<String, AttrValue>,
    },
    /// Constant value
    Constant {
        /// Constant data
        data: Vec<u8>,
        /// Data type
        dtype: crate::tensor::DataType,
        /// Shape
        shape: crate::tensor::TensorShape,
    },
    /// Shape manipulation
    Reshape {
        /// Input node
        input: NodeId,
        /// New shape
        new_shape: Vec<usize>,
    },
    /// Transpose operation
    Transpose {
        /// Input node
        input: NodeId,
        /// Permutation
        perm: Vec<usize>,
    },
    /// Slice operation
    Slice {
        /// Input node
        input: NodeId,
        /// Start indices
        starts: Vec<usize>,
        /// End indices
        ends: Vec<usize>,
        /// Strides
        strides: Vec<usize>,
    },
    /// Concatenate tensors
    Concat {
        /// Input nodes
        inputs: Vec<NodeId>,
        /// Axis to concatenate along
        axis: usize,
    },
    /// Split tensor
    Split {
        /// Input node
        input: NodeId,
        /// Axis to split along
        axis: usize,
        /// Split sizes
        split_sizes: Vec<usize>,
    },
    /// Control flow: condition
    Condition {
        /// Condition node
        cond: NodeId,
        /// True branch
        true_branch: Vec<NodeId>,
        /// False branch
        false_branch: Vec<NodeId>,
    },
    /// Control flow: loop
    Loop {
        /// Trip count
        trip_count: NodeId,
        /// Loop body
        body: Vec<NodeId>,
    },
}

/// Attribute value for operation attributes
#[derive(Debug, Clone)]
pub enum AttrValue {
    /// Integer attribute
    Int(i64),
    /// Float attribute
    Float(f32),
    /// String attribute
    String(String),
    /// Integer list
    Ints(Vec<i64>),
    /// Float list
    Floats(Vec<f32>),
    /// String list
    Strings(Vec<String>),
    /// Tensor attribute
    Tensor(Vec<u8>),
}

/// Compute graph node
#[derive(Debug, Clone)]
pub struct Node {
    /// Node ID
    pub id: NodeId,
    /// Node name
    pub name: String,
    /// Operation type
    pub op: OpType,
    /// Output shape (computed)
    pub output_shape: Option<crate::tensor::TensorShape>,
    /// Output data type
    pub output_dtype: Option<crate::tensor::DataType>,
    /// Estimated compute cost (FLOPs)
    pub compute_cost: u64,
    /// Estimated memory footprint
    pub memory_cost: usize,
}

impl Node {
    /// Create a new node
    pub fn new(id: NodeId, name: impl Into<String>, op: OpType) -> Self {
        Self {
            id,
            name: name.into(),
            op,
            output_shape: None,
            output_dtype: None,
            compute_cost: 0,
            memory_cost: 0,
        }
    }
    
    /// Get input dependencies
    pub fn inputs(&self) -> Vec<NodeId> {
        match &self.op {
            OpType::Input { .. } => vec![],
            OpType::Output { source } => vec![*source],
            OpType::Kernel { inputs, .. } => inputs.clone(),
            OpType::Constant { .. } => vec![],
            OpType::Reshape { input, .. } => vec![*input],
            OpType::Transpose { input, .. } => vec![*input],
            OpType::Slice { input, .. } => vec![*input],
            OpType::Concat { inputs, .. } => inputs.clone(),
            OpType::Split { input, .. } => vec![*input],
            OpType::Condition { cond, true_branch, false_branch } => {
                let mut deps = vec![*cond];
                deps.extend(true_branch);
                deps.extend(false_branch);
                deps
            }
            OpType::Loop { trip_count, body } => {
                let mut deps = vec![*trip_count];
                deps.extend(body);
                deps
            }
        }
    }
    
    /// Check if this is a compute operation
    pub fn is_compute(&self) -> bool {
        matches!(self.op, OpType::Kernel { .. })
    }
    
    /// Check if this is a memory operation
    pub fn is_memory(&self) -> bool {
        matches!(self.op, OpType::Reshape { .. } | OpType::Transpose { .. } | 
                          OpType::Slice { .. } | OpType::Concat { .. } | OpType::Split { .. })
    }
}

/// Compute graph for operation scheduling
#[derive(Debug, Clone, Default)]
pub struct ComputeGraph {
    /// Graph nodes
    nodes: HashMap<NodeId, Node>,
    /// Input nodes
    inputs: Vec<NodeId>,
    /// Output nodes
    outputs: Vec<NodeId>,
    /// Next node ID
    next_id: u64,
}

impl ComputeGraph {
    /// Create a new empty compute graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            next_id: 1,
        }
    }
    
    /// Create a new node ID
    fn new_id(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        id
    }
    
    /// Add a node to the graph
    pub fn add_node(&mut self, name: impl Into<String>, op: OpType) -> NodeId {
        let id = self.new_id();
        let node = Node::new(id, name, op);
        self.nodes.insert(id, node);
        id
    }
    
    /// Add an input tensor
    pub fn add_input(&mut self, name: impl Into<String>, tensor: Tensor) -> NodeId {
        let id = self.add_node(name, OpType::Input { tensor });
        self.inputs.push(id);
        id
    }
    
    /// Add an output
    pub fn add_output(&mut self, name: impl Into<String>, source: NodeId) -> NodeId {
        let id = self.add_node(name, OpType::Output { source });
        self.outputs.push(id);
        id
    }
    
    /// Add a kernel operation
    pub fn add_kernel(
        &mut self,
        name: impl Into<String>,
        kernel_type: KernelType,
        inputs: Vec<NodeId>,
        attrs: HashMap<String, AttrValue>,
    ) -> NodeId {
        self.add_node(name, OpType::Kernel { kernel_type, inputs, attrs })
    }
    
    /// Get a node by ID
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }
    
    /// Get a mutable node by ID
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }
    
    /// Get input nodes
    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }
    
    /// Get output nodes
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }
    
    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<NodeId, Node> {
        &self.nodes
    }
    
    /// Get node count
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    
    /// Check if graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    
    /// Topological sort of the graph
    pub fn topological_sort(&self) -> Vec<NodeId> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut temp_mark = HashSet::new();
        
        fn visit(
            graph: &ComputeGraph,
            node_id: NodeId,
            visited: &mut HashSet<NodeId>,
            temp_mark: &mut HashSet<NodeId>,
            result: &mut Vec<NodeId>,
        ) {
            if visited.contains(&node_id) {
                return;
            }
            if temp_mark.contains(&node_id) {
                panic!("Cycle detected in compute graph");
            }
            
            temp_mark.insert(node_id);
            
            if let Some(node) = graph.get(node_id) {
                for input in node.inputs() {
                    visit(graph, input, visited, temp_mark, result);
                }
            }
            
            temp_mark.remove(&node_id);
            visited.insert(node_id);
            result.push(node_id);
        }
        
        // Visit all output nodes
        for &output in &self.outputs {
            visit(self, output, &mut visited, &mut temp_mark, &mut result);
        }
        
        result
    }
    
    /// Find fusible operations (simple fusion patterns)
    pub fn find_fusion_candidates(&self) -> Vec<Vec<NodeId>> {
        let mut candidates = Vec::new();
        let sorted = self.topological_sort();
        
        // Look for activation fusion patterns (e.g., Matmul + Bias + Activation)
        for window in sorted.windows(3) {
            if let [a, b, c] = window {
                if let (Some(node_a), Some(node_b), Some(node_c)) = 
                    (self.get(*a), self.get(*b), self.get(*c)) {
                    // Check for Matmul -> Add -> Activation pattern
                    if let (
                        OpType::Kernel { kernel_type: KernelType::Matmul, .. },
                        OpType::Kernel { kernel_type: KernelType::Binary(BinaryOp::Add), .. },
                        OpType::Kernel { kernel_type: KernelType::Unary(UnaryOp::Relu), .. }
                    ) = (&node_a.op, &node_b.op, &node_c.op) {
                        candidates.push(vec![*a, *b, *c]);
                    }
                }
            }
        }
        
        candidates
    }
    
    /// Estimate total FLOPs for the graph
    pub fn estimate_flops(&self) -> u64 {
        self.nodes.values().map(|n| n.compute_cost).sum()
    }
    
    /// Estimate peak memory usage
    pub fn estimate_memory(&self) -> usize {
        self.nodes.values().map(|n| n.memory_cost).max().unwrap_or(0)
    }
}

/// Graph executor trait
pub trait GraphExecutor {
    /// Execute a compute graph
    fn execute(&self, graph: &ComputeGraph) -> crate::error::Result<Vec<Tensor>>;
    
    /// Execute with profiling
    fn execute_profiled(&self, graph: &ComputeGraph) -> crate::error::Result<(Vec<Tensor>, ExecutionProfile)>;
}

/// Execution profile
#[derive(Debug, Clone, Default)]
pub struct ExecutionProfile {
    /// Total execution time in microseconds
    pub total_time_us: u64,
    /// Per-node execution times
    pub node_times: HashMap<NodeId, u64>,
    /// Memory usage over time
    pub memory_profile: Vec<(u64, usize)>,
    /// Kernel launch overhead
    pub launch_overhead_us: u64,
}

/// Binary operations (re-export for graph)
pub use crate::kernel::BinaryOp;
/// Unary operations (re-export for graph)
pub use crate::kernel::UnaryOp;
