use crate::{EssentialPolytope, ReverseSearchOut, StepResult};
use crate::{Polytope, Searcher, TreeIndex};
use anyhow::{anyhow, Result};
use bit_set::BitSet;
use core::future::Future;
use std::rc::Rc;
use itertools::Itertools;
use rand::prelude::SeedableRng;
use rand::prelude::*;
use std::cmp::Ord;
use std::cmp::Ordering;
use std::collections::hash_map::Iter as MapIter;
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::usize;
use log::{debug, info};
use std::cell::Cell;

pub type Executor<'a> =
    dyn Fn(Vec<TreeIndex>) -> Pin<Box<dyn Future<Output = Result<StepResult>>>>;

#[derive(Default)]
pub struct Node {
    path: Vec<TreeIndex>,
    // Cached value from the reverse search. If an adjacent state
    // is non-listed in either then it has not been tested
    nonadjacent: HashMap<usize, BitSet>,
    // Adjacent states are cached in the children
    children: HashMap<TreeIndex, Node>,
    // Transient collection holding the indices being investigated
    testing: HashSet<TreeIndex>,
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

impl Node {
    pub fn new() -> Self {
        Node {
            path: Vec::new(),
            nonadjacent: HashMap::new(),
            children: HashMap::new(),
            testing: HashSet::new(),
        }
    }

    fn get_node_mut(&mut self, path: &[TreeIndex]) -> Option<&mut Node> {
        let begin = Some(self);
        path.iter().fold(begin, |accum, index| {
            accum.and_then(|node| node.children.get_mut(index))
        })
    }

    fn get_node(&self, path: &[TreeIndex]) -> Option<&Node> {
        let begin = Some(self);
        path.iter().fold(begin, |accum, index| {
            accum.and_then(|node| node.children.get(index))
        })
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
            testing: HashSet::new(),
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

    pub fn is_nonadjacent(&self, index: &TreeIndex) -> bool {
        self.nonadjacent
            .get(&index.vertex_id)
            .map(|bit_set| bit_set.contains(index.polytope_id))
            .unwrap_or(false)
    }


    // Find best candidate for next search state
    pub fn score<'a>(
        &'a self,
        decomposition: &'a [usize],
        scorer: impl Fn(&mut (dyn Iterator<Item = usize>)) -> usize,
    ) -> Vec<&[TreeIndex]> {
        let mut paths_by_score: Vec<(usize, usize, &[TreeIndex])> = Vec::new();
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
                    let mut index_iter = tree_index_iter(decomposition, &stack_item.node.path);
                    let node_score = scorer(&mut index_iter);
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
        paths_by_score.sort_by(|left: &(usize, usize, &[TreeIndex]), right| match left.0.cmp(&right.0) {
            Ordering::Equal => right.1.cmp(&left.1),
            ordering => ordering,
        });
        paths_by_score.iter().map(|(_s, _e, path)| *path).collect_vec()
    }
}

pub fn search_wrapper(searcher: & mut Searcher, path: Vec<TreeIndex>) -> Result<StepResult> {
    let to_test = path
        .last()
        .ok_or(anyhow!("Cannot search with empty path"))?;
    let mut history: Vec<TreeIndex> = Vec::new();
    for index in &path[..path.len() - 1] {
        history.push(TreeIndex::new(
            index.polytope_id,
            searcher.minkowski_decomp[index.polytope_id],
        ));
        searcher.minkowski_decomp[index.polytope_id] = index.vertex_id;
    }
    let prev = TreeIndex::new(
        to_test.polytope_id,
        searcher.minkowski_decomp[to_test.polytope_id],
    );
    let step_result = searcher.step(&prev, to_test.vertex_id);
    for index in history.iter().rev() {
        searcher.minkowski_decomp[index.polytope_id] = index.vertex_id;
    }
    step_result
}

pub struct Guide {
    minkowski_decomp: Vec<usize>,
    polys: Vec<EssentialPolytope>,
    labels: Vec<usize>,
    rng: Cell<Option<SmallRng>>,
    root: Cell<Node>,
    //executor: Rc<Box<dyn Fn(Vec<TreeIndex>) -> Pin<Box<dyn Future<Output = Result<StepResult>> + 'a>> +'a>>
}

impl<'a> Guide {
    pub fn new(
        searcher: &Searcher,
        labels: &[usize],
        seed: Option<u64>,
    ) -> Result<Self> {
        let rng = seed.map_or(SmallRng::from_entropy(), |seed| {
            SmallRng::seed_from_u64(seed)
        });
        Ok(Guide {
            minkowski_decomp: searcher.minkowski_decomp.to_owned(),
            polys: searcher.polys.to_owned(),
            labels: labels.to_owned(),
            rng: Cell::new(Some(rng)),
            root: Cell::new(Node::new()),
            //executor
        })
    }


    // Run a single iteration of guided search. Note that this is not thread safe!
    pub async fn guided_search(& self, executor: Rc<Box<dyn Fn(Vec<TreeIndex>) -> Pin<Box<dyn Future<Output = Result<StepResult>> + 'a>> +'a>>)  -> Result<Option<ReverseSearchOut>> {
        let scoring_fn = |to_score: &mut dyn Iterator<Item = usize>| {
            let res: usize = to_score
                .zip_eq(&self.labels)
                .map(|(a, b)| if a == *b { 1usize } else { 0usize })
                .sum();
            res
        };
        let mut to_test: Vec<TreeIndex>;
        {
            let root = self.root.take();
            // Do all the immutable processing in it's own scope
            let mut scored_nodes =
                root
                .score(&self.minkowski_decomp, scoring_fn);

            let make_candidates = |best_nodepath: Option<&[TreeIndex]>| {
                let best_node = root.get_node(best_nodepath.ok_or(anyhow!("No scored nodes!"))?).ok_or(anyhow!("Path doesn't find node. Should never happen"))?;
                let candidates = || {
                    tree_index_iter(&self.minkowski_decomp, &best_node.path)
                        .enumerate()
                        .flat_map(|(poly_index, vertex)| {
                            self.polys[poly_index].adjacency[vertex]
                                .iter()
                                .map(move |neighbour| TreeIndex::new(poly_index, *neighbour))
                        })
                        .filter(|index| {
                            !best_node.children.contains_key(index)
                                && !best_node.is_nonadjacent(index)
                                && !best_node.testing.contains(index)
                        })
                };

                let filtered_by_labels = candidates()
                    .filter(|index| self.labels[index.polytope_id] == index.vertex_id)
                    .collect::<Vec<_>>();
                Ok::<_, anyhow::Error>(if filtered_by_labels.is_empty() {
                    (best_node, candidates().collect::<Vec<_>>())
                } else {
                    (best_node, filtered_by_labels)
                })
            };

            let (mut best_node, mut to_search) = make_candidates(scored_nodes.pop())?;
            while to_search.len() == 0 {
                (best_node, to_search) = make_candidates(scored_nodes.pop())?;
            }

            let mut rng = self.rng.take().unwrap();
            let selection_index = (rng.next_u32() as usize) % to_search.len();
            self.rng.set(Some(rng));
            let selection = &to_search[selection_index];
            to_test = best_node.path.clone();
            to_test.push(selection.clone());
            self.root.set(root);
        }
        let mut root = self.root.take();
        let mut_node = 
            root
            .get_node_mut(&to_test[..to_test.len() - 1])
            .ok_or(anyhow!("Can't find best node!"))?;

        let test_selection = to_test.last().unwrap();
        mut_node.testing.insert(test_selection.clone());
        let step_result = (executor)(to_test.clone()).await;
        mut_node.testing.remove(test_selection);

        let mut mark_nonadj = || {
            mut_node.mark_nonadjacent(test_selection.polytope_id, test_selection.vertex_id);
            Option::<ReverseSearchOut>::None
        };

        let result = match step_result? {
            StepResult::Child(param) => {
                mut_node.make_child(test_selection.polytope_id, test_selection.vertex_id);
                Some(ReverseSearchOut::encode(
                    param,
                    tree_index_iter(&self.minkowski_decomp, &to_test),
                )?)
            }
            StepResult::Nonadjacent => mark_nonadj(),
            StepResult::NotAChild => mark_nonadj(),
            StepResult::LoopComplete => mark_nonadj(),
        };
        self.root.set(root);

        Ok(result)
    }
}
