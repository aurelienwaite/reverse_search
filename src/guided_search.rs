use crate::{EssentialPolytope, ReverseSearchOut, StepResult};
use crate::{Searcher, TreeIndex};
use anyhow::{anyhow, Result};
use bit_set::BitSet;
use core::future::Future;
use itertools::Itertools;
use log::{debug, info};
use rand::prelude::SeedableRng;
use rand::prelude::*;
use std::cell::Cell;
use std::cmp::Ord;
use std::cmp::Ordering;
use std::collections::hash_map::Iter as MapIter;
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::rc::Rc;
use std::usize;
use ndarray_stats::QuantileExt;

static TOP_K: usize = 100;

pub type Executor<'a> = dyn Fn(Vec<TreeIndex>) -> Pin<Box<dyn Future<Output = Result<StepResult>>>>;

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
    iterator: MapIter<'a, TreeIndex, Node>,
}

pub trait Scorer {
    fn score<'a>(&self, to_score: Box<dyn Iterator<Item = usize> + 'a>) -> f64;
}

/*
 * Treat NaNs as less than everything and two NaNs as equal
 */
fn float_compare(left: f64, right: f64) -> Ordering {
    let max_score = f64::max(left, right);
    if left == right {
        Ordering::Equal
    } else if left == max_score {
        Ordering::Greater
    } else if right == max_score {
        debug_assert!(right == max_score);
        Ordering::Less
    } else {
        // Treat two nans as equal
        assert!(left != left && right != right);
        Ordering::Equal
    }
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
}

pub fn search_wrapper(searcher: &mut Searcher, path: Vec<TreeIndex>) -> Result<StepResult> {
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
    scorer: Rc<dyn Scorer>,
    rng: Cell<Option<SmallRng>>,
    root: Cell<Option<Node>>,
    mappings: Vec<HashMap<usize, usize>>
}

impl<'a> Guide {
    pub fn new(searcher: &Searcher, scorer: Rc<dyn Scorer>, seed: Option<u64>) -> Result<Self> {
        let rng = seed.map_or(SmallRng::from_entropy(), |seed| {
            SmallRng::seed_from_u64(seed)
        });
        let mappings = searcher.polys.iter().map(|p| p.mapping.clone()).collect_vec();
        Ok(Guide {
            minkowski_decomp: searcher.minkowski_decomp.to_owned(),
            polys: searcher.polys.to_owned(),
            scorer: scorer,
            rng: Cell::new(Some(rng)),
            root: Cell::new(Some(Node::new())),
            mappings,
        })
    }

    pub fn map_polytope_indices<'b>(&'b self, decomp_iter: impl Iterator<Item=usize> + 'b) -> impl Iterator<Item=usize> + 'b{
        decomp_iter.zip(self.mappings.iter()).map(|(v, mapping)| mapping[&v])
    }

    // Run a single iteration of guided search. Note that this is not thread safe!
    pub async fn guided_search(
        &self,
        executor: Rc<
            Box<
                dyn Fn(Vec<TreeIndex>) -> Pin<Box<dyn Future<Output = Result<StepResult>> + 'a>>
                    + 'a,
            >,
        >,
    ) -> Result<Option<ReverseSearchOut>> {
        let to_test: Vec<TreeIndex>;
        {
            // Do all the immutable processing in it's own scope
            let mut scored_nodes = self.score(&self.minkowski_decomp);

            let root = self.root.take().unwrap();
            let make_candidates = |best_nodepath: Vec<TreeIndex>| {
                let best_node_res = root
                    .get_node(&best_nodepath)
                    .ok_or(anyhow!("Path doesn't find node. Should never happen"));
                let best_node = match best_node_res {
                    Ok(best_node) => best_node,
                    Err(err) => return Err(err),
                };
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
                Ok((best_node, candidates().collect::<Vec<_>>()))
            };

            let mut num_searched = 0usize;
            let mut merged_candidates: Vec<(Vec<TreeIndex>, f64)> = Vec::new();
            while let Some(scored_node) = scored_nodes.pop() {
                if !merged_candidates.is_empty() && num_searched >= TOP_K {
                    break;
                }
                num_searched += 1;
                let (best_node, to_search) = match make_candidates(scored_node) {
                    Ok(candidates) => candidates,
                    Err(err) => {
                        self.root.set(Some(root));
                        return Err(err);
                    }
                };
                let mut test_path = best_node.path.to_owned();
                for tree_index in to_search {
                    test_path.push(tree_index);
                    let mapped_iter = self.map_polytope_indices(tree_index_iter(&self.minkowski_decomp, &test_path));
                    let index_iter = Box::new(mapped_iter);
                    let node_score = self.scorer.score(index_iter);
                    merged_candidates.push((test_path.to_owned(), node_score));
                    test_path.pop();
                }
            }
            merged_candidates.sort_by(|left, right| float_compare(left.1, right.1));
            let best_candidate = merged_candidates.pop().ok_or(anyhow!("No candidates!"))?;
            let best_score = best_candidate.1;
            let mut best_candidates = vec![best_candidate.0];
            while let Some((candidate, score)) = merged_candidates.pop() {
                if score < best_score {
                    break;
                }
                best_candidates.push(candidate);
            }

            let mut rng = self.rng.take().unwrap();
            let selection_index = (rng.next_u32() as usize) % best_candidates.len();
            to_test = best_candidates[selection_index].to_owned();
            self.rng.set(Some(rng));
            self.root.set(Some(root));
        }

        let test_selection = to_test.last().unwrap();
        {
            // Moving each root take into a separate scope to make it clear that we're borrowing
            let mut root = self.root.take().unwrap();
            let mut_node = root
                .get_node_mut(&to_test[..to_test.len() - 1])
                .ok_or(anyhow!("Can't find best node!"))?;
            mut_node.testing.insert(test_selection.clone());
            self.root.set(Some(root));
        }
        let step_result = (executor)(to_test.clone()).await;
        debug!("Got result, continuing processing");
        let result;
        {
            // Need a new scope because this executes after the await
            let mut root = self.root.take().unwrap();
            let mut_node = root
                .get_node_mut(&to_test[..to_test.len() - 1])
                .ok_or(anyhow!("Can't find best node!"))?;
            mut_node.testing.remove(test_selection);

            let mut mark_nonadj = || {
                mut_node.mark_nonadjacent(test_selection.polytope_id, test_selection.vertex_id);
                Option::<ReverseSearchOut>::None
            };

            result = match step_result? {
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
            self.root.set(Some(root));
        }
        let root = self.root.take();
        assert!(root.is_some());
        debug!("Root ok");
        self.root.set(root);
        Ok(result)
    }

    // Find best candidate for next search state
    pub fn score(&self, decomposition: &'a [usize]) -> Vec<Vec<TreeIndex>> {
        let mut paths_by_score: Vec<(f64, usize, Vec<TreeIndex>)> = Vec::new();
        {
            let mut stack: Vec<StackItem> = Vec::new();
            let root = self.root.take().unwrap();
            stack.push(StackItem {
                node: &root,
                iterator: (&root).children.iter(),
            });
            while !stack.is_empty() {
                let mut stack_item = stack.pop().unwrap();
                let iter_next = stack_item.iterator.next();
                match iter_next {
                    Some((_, next_node)) => {
                        stack.push(stack_item);
                        stack.push(StackItem {
                            node: next_node,
                            iterator: next_node.children.iter(),
                        });
                    }
                    None => {
                        let mapped_iter = self.map_polytope_indices(tree_index_iter(decomposition, &stack_item.node.path));
                        let index_iter =
                            Box::new(mapped_iter);
                        let node_score = self.scorer.score(index_iter);
                        let explored = stack_item.node.children.len()
                            + stack_item
                                .node
                                .nonadjacent
                                .values()
                                .map(|polytopes| polytopes.len())
                                .sum::<usize>();
                        paths_by_score.push((
                            node_score,
                            explored,
                            stack_item.node.path.to_owned(),
                        ));
                    }
                }
            }
            self.root.set(Some(root));
        }
        paths_by_score.sort_by(|left: &(f64, usize, Vec<TreeIndex>), right| {
            let score_ordering = float_compare(left.0, right.0);
            match score_ordering {
                Ordering::Equal => right.1.cmp(&left.1),
                ordering => ordering,
            }
        });
        paths_by_score
            .into_iter()
            .map(|(_s, _e, path)| path)
            .collect_vec()
    }
}
