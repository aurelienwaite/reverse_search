use bit_set::BitSet;
use std::cmp::Ordering;
use std::collections::hash_map::Iter as MapIter;
use std::collections::HashMap;
use std::usize;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TreeIndex {
    pub polytope_id: usize,
    pub vertex_id: usize,
}

impl TreeIndex{
    pub fn new(polytope_id: usize, vertex_id: usize) -> Self{
        TreeIndex{
            polytope_id,
            vertex_id
        }
    }
}

pub type NodePath = Vec<TreeIndex>;

pub struct Node {
    path: NodePath,
    // Cached value from the reverse search. If an adjacent state
    // is non-listed in either then it has not been tested
    nonadjacent: HashMap<usize, BitSet>,
    // Adjacent states are cached in the children
    children: HashMap<TreeIndex, Node>,
}

struct StackItem<'a> {
    node: &'a Node,
    reward: f32,
    iterator: MapIter<'a, TreeIndex, Node>,
}

fn tree_index_iter<'a>(
    decomp: &'a [usize],
    node_id: &'a [TreeIndex],
) -> impl Iterator<Item = usize> + 'a {
    let mut overrides: HashMap<usize, usize> = HashMap::new();
    for ti in node_id.iter() {
        overrides.insert(ti.polytope_id, ti.vertex_id);
    }

    struct TreeIndexIter<'b> {
        overrides: HashMap<usize, usize>,

        wrapped_iter: Box<dyn Iterator<Item = (usize, &'b usize)> + 'b>,
    }

    impl Iterator for TreeIndexIter<'_> {
        type Item = usize;

        fn next(&mut self) -> Option<Self::Item> {
            self.wrapped_iter
                .next()
                .map(|(p_index, v_index)| *self.overrides.get(&p_index).unwrap_or(v_index))
        }
    }

    TreeIndexIter {
        overrides: overrides,
        wrapped_iter: Box::new(decomp.iter().enumerate()),
    }
}

pub fn accuracy(decomp: &[usize], labels: &[usize]) -> f32 {
    let matches: f32 = decomp
        .iter()
        .zip_eq(labels)
        .map(|(h, l)| if h == l { 1_f32 } else { 0_f32 })
        .sum();
    // Connfident that these won't overflow
    let len_downsize: i16 = decomp.len().try_into().unwrap();
    let len_float: f32 = len_downsize.try_into().unwrap();
    (matches / len_float) * 100.
}


impl Node {
    pub fn new() -> Self {
        Node {
            path: Vec::new(),
            nonadjacent: HashMap::new(),
            children: HashMap::new(),
        }
    }

    pub fn make_child(&mut self, polytope_id: usize, vertex_id: usize) -> &Self {
        let mut path = self.path.clone();
        let index = TreeIndex {
            polytope_id,
            vertex_id,
        };
        path.push(index.clone());
        let child_node = Node {
            path,
            nonadjacent: HashMap::new(),
            children: HashMap::new(),
        };
        self.children.insert(index.clone(), child_node);
        self.children.get(&index).unwrap()
    }

    pub fn mark_nonadjacent(&mut self, polytope_id: usize, vertex_id: usize) {
        if !self.nonadjacent.contains_key(&vertex_id) {
            self.nonadjacent.insert(vertex_id, BitSet::new());
        }
        let polytopes: &mut BitSet = self.nonadjacent.get_mut(&vertex_id).unwrap();
        polytopes.insert(polytope_id);
    }

    // Find best candidate for next search state
    pub fn score<'a>(
        &'a self,
        decomposition: &'a [usize],
        scoring_fn: Box<dyn Fn(Box<dyn Iterator<Item = usize> + 'a>) -> usize>,
    ) -> Option<&'a NodePath> {
        let mut paths_by_score: Vec<(usize, usize, &NodePath)> = Vec::new();
        let mut stack: Vec<StackItem> = Vec::new();
        stack.push(StackItem {
            node: self,
            reward: 0.,
            iterator: self.children.iter(),
        });
        while !stack.is_empty() {
            let mut stack_item = stack.pop().unwrap();
            let iter_next = stack_item.iterator.next();
            match iter_next {
                Some((_, next_node)) => {
                    stack.push(stack_item);
                    stack.push(StackItem {
                        node: next_node,
                        reward: 0.,
                        iterator: next_node.children.iter(),
                    });
                }
                None => {
                    let index_iter = tree_index_iter(decomposition, &stack_item.node.path);
                    let node_score = scoring_fn(Box::new(index_iter));
                    let explored = stack_item.node.children.len()
                        + stack_item
                            .node
                            .nonadjacent
                            .values()
                            .map(|polytopes| polytopes.len())
                            .sum::<usize>();
                    paths_by_score.push((node_score, explored, &stack_item.node.path));
                }
            }
        }
        paths_by_score.sort_by(|left, right| match left.0.cmp(&right.0) {
            Ordering::Equal => right.1.cmp(&left.1),
            ordering => ordering,
        });
        paths_by_score.last().map(|(_s, _u, id)| *id)
    }
}
