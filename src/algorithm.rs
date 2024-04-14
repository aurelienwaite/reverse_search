use anyhow::{anyhow, Result};
use good_lp::{
    constraint, default_solver, variables, Expression, ResolutionError, Solution, SolverModel,
};

use itertools::Itertools;
use log::{debug, info};
use ndarray::prelude::*;
use ndarray_linalg::norm::normalize;
use ndarray_linalg::NormalizeAxis;
use ndarray_rand::rand_distr::num_traits::{float, zero};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::{DeviationExt, QuantileExt};
use serde::{Deserialize, Serialize};
use serde_json::map::Iter;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env::var;
use std::io::{stdout, Error, Write};
use std::ops::{Index, Sub};
use std::os::macos::raw::stat;
use std::rc::Rc;

type Vector = [f64];
type Matrix = Vec<Rc<Vector>>;

const EMPTY_MATRIX: Matrix = Vec::new();

// Used for the LP separation problem. Due to the formalisation of the LP the number will always
// be in the range [0,1]. Having a fixed epsilon is probably ok.
const EPSILON: f64 = 1e-6;

enum EssentialSolution {
    Essential(Vec<f64>), // Contains the certificate of the solution
    Redundant,
}

enum AdjacencySolution {
    Adjacent(Array1<f64>), // Contains the certificate of the solution
    Infeasible,
}

fn inner_product_lp(coefficients: &Vector, variables: &[good_lp::Variable]) -> Expression {
    return coefficients
        .iter()
        .zip_eq(variables)
        .map(|item| *item.0 * *item.1)
        .sum();
}

// Frustratingly, I can't seem to find a way to make this generic with inner_product_lp
// The issue is that into_iter for an native array needs a mutable reference :(
// There must be a way...
fn inner_product_lp_ndarray(
    coefficients: &ArrayView1<f64>,
    variables: &[good_lp::Variable],
) -> Expression {
    return coefficients
        .iter()
        .zip_eq(variables)
        .map(|item| *item.0 * *item.1)
        .sum();
}

// Used to detecting extreme points in individual polytopes
fn lp_separation(
    to_test: &Vector,
    vertices: &Matrix,
    linearity: &Matrix,
) -> Result<EssentialSolution, ResolutionError> {
    let mut vars = variables!();
    let dim = to_test.len();
    let mut x: Vec<good_lp::Variable> = Vec::with_capacity(dim);
    for _i in 0..dim {
        x.push(vars.add_variable());
    }
    let d = vars.add_variable();

    let objective = d + inner_product_lp(&to_test, &x);
    let mut problem = vars
        .maximise(&objective)
        .using(default_solver)
        .with(constraint!(d + inner_product_lp(&to_test, &x) <= 1));

    for vertex in vertices {
        problem = problem.with(constraint!(d + inner_product_lp(&vertex, &x) <= 0.));
    }
    for generator in linearity {
        problem = problem.with(constraint!(d + inner_product_lp(&generator, &x) == 0.));
    }

    let solution = problem.solve()?;
    let maxima = solution.eval(&objective);
    let sep_solution = if maxima > EPSILON {
        let mut cert: Vec<f64> = Vec::with_capacity(x.len());
        cert.push(solution.value(d));
        cert.extend(x.iter().map(|variable| solution.value(*variable)));
        EssentialSolution::Essential(cert)
    } else {
        EssentialSolution::Redundant
    };
    return Ok(sep_solution);
}

// Finds a maximiser for a normal cone
fn lp_normal_cone(edges: &Array2<f64>) -> Result<Array1<f64>> {
    let mut vars = variables!();
    let num_edges = edges.shape()[0];
    let ones: Array2<f64> = Array2::ones((num_edges, 1));
    let constraints = ndarray::concatenate(Axis(1), &[ones.view(), edges.view()])?;
    let dim = constraints.shape()[1];
    let mut x: Vec<good_lp::Variable> = Vec::with_capacity(dim + 1);
    for _i in 0..dim + 1 {
        x.push(vars.add_variable());
    }

    let objective = x.first().ok_or(anyhow!("No variables!"))?;
    let mut problem = vars.maximise(objective).using(default_solver);
    for edge in constraints.axis_iter(Axis(0)) {
        let constraint = inner_product_lp_ndarray(&edge, &x);
        // Copied from CDD, I think the 1 is to ensure that the point is interior
        problem = problem.with(constraint!(constraint <= 1.));
    }
    // To be honest, I'm not completely sure why this is needed. I think
    // it's to keep the solution bounded but using a 2 is a bit arbitrary.
    problem = problem.with(constraint!(*objective <= 2.));

    let solution = problem.solve()?;
    let maximiser = x.iter().map(|v| solution.value(*v)).collect::<Vec<_>>();
    return Ok(arr1(&maximiser));
}

// Identifies whether an edge identifies a boundary to an adjacent cone
fn lp_adjacency(edges: &Array2<f64>, to_test: usize) -> Result<AdjacencySolution> {
    let test_vec = edges
        .select(Axis(0), &[to_test])
        .into_shape(edges.shape()[1])?;

    let dim = edges.shape()[1];
    let mut vars = variables!();
    let mut x: Vec<good_lp::Variable> = Vec::with_capacity(dim);
    for _i in 0..dim {
        x.push(vars.add_variable());
    }
    let objective = inner_product_lp_ndarray(&test_vec.view(), &x);
    let mut problem = vars.maximise(objective.clone()).using(default_solver);
    // Minksum adds a 1 to the objective and adds it as constraint. Assuming that this
    // is a bounding constraint and reproducing here.
    let bounding_constraint = objective.clone() * -1;
    problem = problem.with(constraint!(bounding_constraint <= 1.));

    // The edges have been normalised, making it easy to test for parallism
    let parallel = test_vec.dot(edges).map(|d| (d.abs() - 1.).abs() < EPSILON);
    for (i, (edge, is_parallel)) in edges.axis_iter(Axis(0)).zip(parallel).enumerate() {
        // Don't add parallel edges. This is different to the minksum
        // implementation that returns false if the polytope or neighbour
        // is indexed lower than the given indices.
        if !is_parallel || i == to_test {
            let constraint = inner_product_lp_ndarray(&edge, &x);
            problem = problem.with(constraint!(constraint <= 0.));
        }
    }
    let solution = problem.solve()?;
    let cert: Array1<f64> = x
        .iter()
        .map(|v| solution.value(*v))
        .collect::<Vec<_>>()
        .into();
    let result = if cert.dot(&test_vec) > EPSILON {
        AdjacencySolution::Adjacent(cert)
    } else {
        AdjacencySolution::Infeasible
    };
    return Ok(result);
}

fn make_other_matrix(vertices: &Matrix, to_remove: usize) -> Matrix {
    let mut other: Matrix = Vec::with_capacity(vertices.len() - 1);
    other.extend_from_slice(&vertices[0..to_remove]);
    other.extend_from_slice(&vertices[to_remove + 1..vertices.len()]);
    return other;
}

fn adjacency(
    essential_vertices: &Matrix,
    indices: &BTreeSet<usize>,
) -> Result<HashMap<usize, Vec<usize>>> {
    let index_map = Vec::from_iter(indices); // Maps the essential index back to the original indices
    let mut res: HashMap<usize, Vec<usize>> = HashMap::with_capacity(index_map.len());

    for (index, vertex) in essential_vertices.iter().enumerate() {
        let linearity = Vec::from([Rc::clone(&essential_vertices[index])]);
        eprintln!("Finding adjacent");
        let adjacent = find_essential(essential_vertices, &linearity)?;
        res.insert(
            *index_map[index],
            Vec::from_iter(adjacent.0.iter().map(|adj| *index_map[*adj])),
        );
    }
    eprintln!("{:?}", res);
    return Ok(res);
}

fn find_essential(
    vertices: &Matrix,
    linearity: &Matrix,
) -> Result<(BTreeSet<usize>, HashMap<usize, Vec<f64>>)> {
    let mut essential: BTreeSet<usize> = BTreeSet::new();
    let mut certs: HashMap<usize, Vec<f64>> = HashMap::new();
    for (index, vertex) in (vertices).iter().enumerate() {
        let other = make_other_matrix(&vertices, index);
        let lp_solution = lp_separation(vertex, &other, linearity)?;
        let is_essential = match lp_solution {
            EssentialSolution::Essential(cert) => {
                certs.insert(index, cert);
                essential.insert(index)
            }
            EssentialSolution::Redundant => false,
        };
        if is_essential {
            eprint!("+");
        } else {
            eprint!("-");
        }
        std::io::stdout().flush()?;
        //eprintln!("{:#?}", essential);
    }
    eprintln!();
    eprintln!("{:?}", essential);
    return Ok((essential, certs));
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FullPolytope {
    vertices: Matrix,
    essential_indices: Option<BTreeSet<usize>>,
    adjacency: Option<HashMap<usize, Vec<usize>>>,
    essential_certs: Option<HashMap<usize, Vec<f64>>>,
}

// A polytope with essential data only
struct Polytope {
    vertices: Array2<f64>, // Transposed - dim x vertices
    adjacency: Vec<(usize, usize)>,
    edges: Array2<f64>,
}

impl Polytope {
    fn neighbours(self: &Polytope, vertex: usize) -> Vec<&(usize, usize)> {
        self.adjacency
            .iter()
            .filter(|(first, _)| *first == vertex)
            .collect::<Vec<_>>()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReverseSearchState {
    param: Array1<f64>,
    max_vertices: Vec<usize>, 
    neighbour_index: usize,
    polytope_index: usize,
}

impl ReverseSearchState{

    fn vertex(self: &ReverseSearchState) -> usize {
        return self.max_vertices[self.polytope_index];
    }
}

struct AdjacencyTuple {
    polytope_index: usize,
    vertex: usize,
    neighbour: usize,
}

impl FullPolytope {
    fn fill_essential(&mut self) -> Result<()> {
        let essential = find_essential(&self.vertices, &EMPTY_MATRIX)?;
        self.essential_indices = Option::Some(essential.0);
        self.essential_certs = Option::Some(essential.1);
        return Ok(());
    }

    fn essential_vertices(&self) -> Option<Matrix> {
        return self.essential_indices.as_ref().map(|indices| {
            Vec::from_iter(
                indices
                    .iter()
                    .map(|index| Rc::clone(&(self.vertices[*index]))),
            )
        });
    }

    fn fill_adjacency(&mut self) -> Result<()> {
        return self
            .essential_indices
            .as_ref()
            .and_then(|essential| {
                self.essential_vertices()
                    .map(|vertices| (vertices, essential))
            })
            .map_or(Ok(()), |(vertices, indices)| {
                let adjacent = adjacency(&vertices, indices)?;
                self.adjacency = Option::Some(adjacent);
                Ok(())
            });
    }

    fn essential_polytope(&self) -> Result<Option<Polytope>> {
        let essential = self.essential_indices.as_ref().and_then(|indices| {
            self.adjacency.as_ref().map(|adjacency| {
                let mut mapping: HashMap<usize, usize> = HashMap::new();
                for (essential, raw) in indices.iter().enumerate() {
                    mapping.insert(*raw, essential);
                }
                let dim = self.vertices.first().map_or(0, |f| f.len());
                let mut essential_vertices = Array2::<f64>::zeros((dim, indices.len()));
                for i in indices {
                    for j in 0..dim {
                        essential_vertices[[j, mapping[i]]] = self.vertices[*i][j];
                    }
                }
                let mut essential_adjacency: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
                for (vertex, neighbours) in adjacency {
                    essential_adjacency.insert(
                        mapping[&vertex],
                        neighbours.iter().map(|i| mapping[i]).collect(),
                    );
                }
                (essential_vertices, essential_adjacency)
            })
        });
        let poly: Option<Result<Polytope>> = essential.map(|(vertices, adjacency)| {
            let mut edges: Vec<Array1<f64>> = Vec::new();
            let mut flattened_adjacency: Vec<(usize, usize)> = Vec::new();
            for (current_vertex, neighbours) in adjacency {
                for neighbour in neighbours {
                    let edge = &vertices.index_axis(Axis(1), neighbour)
                        - &vertices.index_axis(Axis(1), current_vertex);
                    edges.push(edge);
                    flattened_adjacency.push((current_vertex, neighbour));
                }
            }
            let expanded_dims = edges
                .iter()
                .map(|e| {
                    let dim = e.shape()[0];
                    return e.view().into_shape((1, dim));
                })
                .collect::<Result<Vec<_>, _>>()?;
            let edge_array: Array2<f64> = ndarray::concatenate(Axis(0), &expanded_dims)?;
            let polytope = Polytope {
                vertices: vertices,
                adjacency: flattened_adjacency,
                edges: normalize(edge_array, NormalizeAxis::Row).0,
            };
            return Ok(polytope);
        });
        return poly.transpose();
    }
}

fn inc_mink_sum_setup(
    poly_list: &mut [FullPolytope],
) -> Result<(Vec<Polytope>, ReverseSearchState)> {
    for poly in poly_list.as_mut() {
        poly.fill_essential()?;
        poly.fill_adjacency()?;
    }

    let essential_polys = poly_list
        .iter()
        .map(|p| p.essential_polytope())
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|p| p.ok_or(anyhow!("No item")))
        .collect::<Result<Vec<_>, _>>()?;

    let dim = essential_polys
        .first()
        .map(|p| p.vertices.shape()[0])
        .ok_or(anyhow!("No polytopes!"))?;

    let mut state: ReverseSearchState = ReverseSearchState {
        param: Array1::ones(dim),
        max_vertices: Vec::with_capacity(poly_list.as_ref().len()),
        polytope_index: 0,
        neighbour_index: 0,
    };

    // Checking to see if any vertices in a polytope have the same score
    // If they do the reverse search will fail. We break times by perturbing the parameter
    // vector
    // Probably a degenerate condition that will never happen.
    let mut ties = true;
    while ties {
        ties = false;
        for (i, polytope) in essential_polys.iter().enumerate() {
            debug!("Testing polytope {} for ties with initial param", i);
            let scores = state.param.dot(&polytope.vertices);
            let mut prev_best = *scores.first().unwrap_or(&0.0);
            let mut best_count = 1;
            for score in scores.to_vec()[1..].iter() {
                if *score > prev_best {
                    prev_best = *score;
                    best_count = 1;
                } else if *score == prev_best {
                    best_count += 1;
                }
            }
            ties = best_count > 1;

            if ties {
                info!(
                    "Found tie: {} {}, perturbing starting parameter",
                    prev_best, best_count
                );
                debug! {"Scores\n {:?}", &scores};
                debug!("Vertices\n {:?}", &polytope.vertices.t());
                let perturb = Array::random(dim, Uniform::new(-1.0 * EPSILON, EPSILON));
                state.param = state.param + perturb;
                break;
            }
        }
    }
    let max_vertices = essential_polys
        .iter()
        .map(|p| state.param.dot(&p.vertices).argmin())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    state.max_vertices = max_vertices;
    debug!("Max vertices {:?}", &state.max_vertices);
    info!("Setup done");
    return Ok((essential_polys, state));
}

// Find the parent of a cone. We shoot a ray towards the initial
// cone, and the first hyperplane intersected by the ray is the
// boundary to the parent cone.
// Edges should be normalised, although normalising isn't done in
// CDD code
fn parent(
    edges: &Array2<f64>,
    interior_point: &Array1<f64>,
    direction: &Array1<f64>,
) -> Result<usize> {
    debug_assert!(interior_point.len() > 0);
    debug_assert_eq!(interior_point.len(), direction.len());
    debug_assert_eq!(
        *interior_point.first().unwrap(),
        1_f64,
        "Points should start with a 1 to store offsets"
    );
    debug_assert_eq!(
        *direction.first().unwrap(),
        0_f64,
        "Directions are rays and should start with a 0"
    );

    let inner_product_point = interior_point.dot(edges);
    let inner_product_direction = direction.dot(edges);
    let alpha = inner_product_direction / inner_product_point;
    let parent = alpha.argmin()?;
    debug_assert_eq!(
        alpha.count_eq(&array![*alpha.get(parent).unwrap()])?,
        1,
        "Found ties when searching for a parent"
    );
    let min_index = alpha.argmin()?;
    return Ok(min_index);
}

pub fn reverse_search(poly_list: &mut [FullPolytope]) -> Result<Vec<Rc<ReverseSearchState>>> {
    let (polys, initial_state) = inc_mink_sum_setup(poly_list)?;
    let initial_state_ref = Rc::new(initial_state);
    let num_edges: usize = polys.iter().map(|p| p.edges.len()).sum();
    // Allocate these arrays in advance
    let mut edges: Vec<ArrayView2<f64>> = Vec::with_capacity(num_edges);
    let mut edge_indices: Vec<AdjacencyTuple> = Vec::with_capacity(num_edges);

    let mut stack: Vec<Rc<ReverseSearchState>> = Vec::new();
    stack.push(initial_state_ref.clone());

    let mut res: Vec<Rc<ReverseSearchState>> = Vec::new();

    while !stack.is_empty() {
        if res.len() >= 10{ //Delete this!
            break;
        }
        let prev_state = stack
            .pop()
            .ok_or(anyhow!("Empty stack! This should never happen"))?;
        let mut state = (*prev_state).clone();
        edges.clear();
        edge_indices.clear();

        let mut polytope = &polys[state.polytope_index];
        let mut vertex = state.vertex();
        state.neighbour_index += 1;
        let mut neighbours = polytope.neighbours(vertex);
        if state.neighbour_index >= neighbours.len() {
            state.polytope_index += 1;
            if state.polytope_index >= polys.len() {
                continue;
            }
            polytope = &polys[state.polytope_index];
            vertex = state.vertex();

            state.neighbour_index = 0;
            neighbours = polytope.neighbours(vertex);
        }

        state.max_vertices[state.polytope_index] = neighbours[state.neighbour_index].1;

        let mut test_edge: Option<(usize, ArrayView1<f64>)> = None;
        let mut edge_counter: usize = 0;
        for (inner_polytope_index, (poly, inner_vertex)) in
            polys.iter().zip(&state.max_vertices).enumerate()
        {
            for (index, edge) in poly.adjacency.iter().zip_eq(poly.edges.axis_iter(Axis(0))) {
                if inner_polytope_index == state.polytope_index
                    && vertex == index.0
                    && *inner_vertex == index.1
                {
                    test_edge = Some((edge_counter, edge));
                }
                edges.push(edge.into_shape((1, edge.shape()[0]))?);
                edge_indices.push(AdjacencyTuple {
                    polytope_index: inner_polytope_index,
                    vertex: index.0,
                    neighbour: index.1,
                });
                edge_counter += 1;
            }
        }
        info!("edges len {} ", edges.len());
        let (test_edge_index, test_edge) = test_edge.ok_or(anyhow!("No edge found!"))?;
        // Test if edge is leaving objective
        let score = state.param.dot(&test_edge);
        if score < -1. * EPSILON {
            stack.push(prev_state);
            continue;  
        }


        let edge_array: Array2<f64> = ndarray::concatenate(Axis(0), &edges)?;

        let is_adjacent =
            match lp_adjacency(&edge_array,  test_edge_index)?{
                AdjacencySolution::Adjacent(_cert) => true,
                AdjacencySolution::Infeasible => false,
            };
        if !is_adjacent {
            stack.push(prev_state);
            continue;
        }

        let maximiser = lp_normal_cone(&edge_array)?;
        debug_assert!(maximiser.dot(&test_edge) > EPSILON, "Because we have tested for adjacency, we should always get a maximiser");
        
        let parent_edge = parent(&edge_array, &maximiser, &initial_state_ref.param)?;
        let parent = edge_indices[parent_edge].neighbour;
        if parent == state.max_vertices[state.polytope_index]{
            let state_ref = Rc::new(state);
            stack.push(state_ref);
            res.push(state_ref);
        }else{
            stack.push(prev_state);
        }
    }
    return Ok(res);
}
