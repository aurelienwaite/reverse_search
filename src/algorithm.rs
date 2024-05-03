use anyhow::{anyhow, Result};
use good_lp::{
    constraint, default_solver, variables, Expression, ResolutionError, Solution, SolverModel,
};

use itertools::Itertools;
use log::{debug, info};
use ndarray::{concatenate, prelude::*};
use ndarray_linalg::norm::normalize;
use ndarray_linalg::{Norm, NormalizeAxis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::rc::Rc;
use std::usize;

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

enum InteriorPointSolution {
    Feasible(Array1<f64>), // Contains the certificate of the solution
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
        .with(constraint!(d + inner_product_lp(&to_test, &x) <= 1.));

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

// Shamelessly copied from CDD for LP feasibility problems
fn lp_interior_point(
    matrix_a: &Array2<f64>,
    vector_b: &Array1<f64>,
) -> Result<InteriorPointSolution> {
    debug_assert_eq!(matrix_a.shape()[0], vector_b.shape()[0]);
    let mut vars = variables!();
    let num_rows = matrix_a.shape()[0];
    let ones: Array2<f64> = Array2::ones((num_rows, 1));
    let constraints = ndarray::concatenate(Axis(1), &[ones.view(), matrix_a.view()])?;
    let dim = constraints.shape()[1];
    let mut x: Vec<good_lp::Variable> = Vec::with_capacity(dim);
    for _i in 0..dim {
        x.push(vars.add_variable());
    }

    let objective = x.first().ok_or(anyhow!("No variables!"))?;
    let mut problem = vars.maximise(objective).using(default_solver);
    for (row, b_val) in constraints.axis_iter(Axis(0)).zip_eq(vector_b) {
        let constraint = inner_product_lp_ndarray(&row, &x);
        problem = problem.with(constraint!(constraint <= (*b_val - EPSILON)));
    }

    // To be honest, I'm not completely sure why this is needed. I think
    // it's to keep the solution bounded but using a 2 is a bit arbitrary.
    let b_ceil = 1f64.max(*vector_b.max()?) * 2.;
    problem = problem.with(constraint!(*objective <= b_ceil));

    let solution = problem.solve()?;
    if solution.value(*objective) < EPSILON {
        return Ok(InteriorPointSolution::Infeasible);
    }
    let maximiser = x[1..]
        .iter()
        .map(|v| solution.value(*v))
        .collect::<Vec<_>>();
    return Ok(InteriorPointSolution::Feasible(arr1(&maximiser)));
}

// Finds a maximiser for a normal cone
fn lp_normal_cone(edges: &Array2<f64>) -> Result<Array1<f64>> {
    // Copied from CDD, I think the 1 is to ensure that the point is interior
    let vector_b: Array1<f64> = Array1::ones(edges.shape()[0]);
    let solution = lp_interior_point(&(-1.0 * edges), &vector_b)?;
    let res = match solution {
        InteriorPointSolution::Feasible(cert) => Ok(cert),
        InteriorPointSolution::Infeasible =>Result::Err(anyhow!("We should always get a feasible solution because the adjacency oracle will return feasible cones"))
    };
    return res;
}

// Identifies whether an edge identifies a boundary to an adjacent cone
fn lp_adjacency(edges: &Array2<f64>, to_test: usize) -> Result<InteriorPointSolution> {
    let dim = edges.shape()[1];
    let test_vec = edges
        .select(Axis(0), &[to_test])
        .into_shape(edges.shape()[1])?;
    let objective_vec = &test_vec * 1.;

    // The edges have been normalised, making it easy to test for parallelism
    let parallel = test_vec
        .dot(&edges.t())
        .map(|d| (d.abs() - 1.).abs() < EPSILON);
    let negated = -1. * edges;
    let edges_view = negated.view();
    let mut filtered = edges_view
        .axis_iter(Axis(0))
        .zip(parallel)
        .filter_map(|(e, is_parallel)| {
            if is_parallel {
                None
            } else {
                Some(e.into_shape((1, dim)))
            }
        })
        .collect::<core::result::Result<Vec<_>, _>>()?;
    filtered.push(objective_vec.view().into_shape((1, dim))?);
    let filtered_array = concatenate(Axis(0), &filtered)?;
    let vector_b = Array1::zeros(filtered.len());
    let solution = lp_interior_point(&filtered_array, &vector_b)?;
    let result = match solution {
        InteriorPointSolution::Feasible(cert) => {
            let score = cert.dot(&test_vec);
            if score < EPSILON {
                InteriorPointSolution::Feasible(cert)
            } else {
                InteriorPointSolution::Infeasible
            }
        }
        InteriorPointSolution::Infeasible => InteriorPointSolution::Infeasible,
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

    for (index, _vertex) in essential_vertices.iter().enumerate() {
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

    fn neighbouring_edges<'a>(
        self: &'a Polytope,
        vertex: usize,
    ) -> Box<dyn Iterator<Item = (&(usize, usize), ArrayView1<f64>)> + 'a> {
        let iterator = self
            .adjacency
            .iter()
            .zip_eq(self.edges.axis_iter(Axis(0)))
            .filter(move |tup| tup.0 .0 == vertex);
        return Box::new(iterator);
    }

    fn maximised_vertex(self: &Polytope, param: &Array1<f64>) -> Result<usize> {
        return Ok(param.dot(&self.vertices).argmax()?);
    }

    fn scores(self: &Polytope, param: &Array1<f64>) -> Vec<(usize, f64)>{
        let scores = param.dot(&self.vertices);
        return scores.iter().enumerate().sorted_by(|(_ai, a_score), (_bi, b_score) |
            if *a_score - *b_score < 0. {
                Ordering::Less
            }else{
                Ordering::Greater
            }
        ).map(|(i, s)| (i, *s)).collect::<Vec<_>>()
    }
}

#[derive(Clone)]
struct ReverseSearchState{
    param: Array1<f64>,
    minkowski_decomp: Vec<usize>,
    neighbour_index: usize,
    polytope_index: usize,
    polys: Rc<Vec<Polytope>>
}

#[derive(Serialize, Deserialize)]
pub struct ReverseSearchOut{
    param: Array1<f64>,
    minkowski_decomp: Vec<usize>,
}

impl ReverseSearchState {
    fn vertex(&self) -> usize {
        return self.minkowski_decomp[self.polytope_index];
    }

    fn polytope(&self) -> &Polytope {
        return &self.polys[self.polytope_index];
    }

    fn neighbours(&self) -> Vec<&(usize, usize)> {
        return self.polytope().neighbours(self.vertex());
    }

    fn complete(&self) -> bool {
        return self.polytope_index >= self.polys.len();
    }

    fn incr_state(&mut self) -> () {
        self.neighbour_index += 1;
        if self.neighbour_index >= self.neighbours().len() {
            self.polytope_index += 1;
            self.neighbour_index = 0;
        }
    }
    
}

#[derive(Debug)]
struct AdjacencyTuple {
    polytope_index: usize,
    _vertex: usize,
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
            debug!("Vertices shape {:?}", vertices.shape());
            for (current_vertex, neighbours) in adjacency {
                for neighbour in neighbours {
                    let edge = &vertices.index_axis(Axis(1), current_vertex)
                        - &vertices.index_axis(Axis(1), neighbour);
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

    // Map the essential indices back to full polytope raw indices before exporting
    fn map_essential_index(self: &FullPolytope, index: usize) -> Option<usize> {
        return self.essential_indices.as_ref().and_then(|m| {
            m.iter()
                .enumerate()
                .filter(|(essential, _r)| *essential == index)
                .map(|(_e, raw)| *raw)
                .take(1)
                .collect::<Vec<_>>()
                .first()
                .map(|i| *i)
        });
    }
}

fn mink_sum_setup<'a>(
    poly_list: &mut [FullPolytope],
) -> Result<ReverseSearchState> {
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

    let polys_ref = Rc::new(essential_polys); 

    let dim = polys_ref
        .first()
        .map(|p| p.vertices.shape()[0])
        .ok_or(anyhow!("No polytopes!"))?;

    let mut state: ReverseSearchState = ReverseSearchState {
        param: Array1::ones(dim),
        minkowski_decomp: Vec::with_capacity(poly_list.as_ref().len()),
        polytope_index: 0,
        neighbour_index: 0,
        polys: polys_ref
    };

    // Checking to see if any vertices in a polytope have the same score
    // If they do the reverse search will fail. We break times by perturbing the parameter
    // vector
    // Probably a degenerate condition that will never happen.
    let mut ties = true;
    while ties {
        ties = false;
        for (i, polytope) in state.polys.iter().enumerate() {
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
    let max_vertices = state.polys
        .iter()
        .map(|p| state.param.dot(&p.vertices).argmax())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    state.minkowski_decomp = max_vertices;
    debug!("Minkowski decomposition {:?}", &state.minkowski_decomp);
    info!("Setup done");
    return Ok(state);
}

// Find the parent of a cone. We shoot a ray towards the initial
// cone, and the first hyperplane intersected by the ray is the
// boundary to the parent cone.
// Following CDD convention, the interior point should have a 1 as its
// first element, and the direction a 0. However, because all the edges
// have no offset we can ignore the first element.
//
fn parent(
    edges: &Array2<f64>,
    interior_point: &Array1<f64>,
    initial_point: &Array1<f64>,
) -> Result<usize> {
    debug_assert!(interior_point.len() > 0);
    debug_assert_eq!(interior_point.len(), initial_point.len());
    let direction = interior_point - initial_point;
    let edge_t = edges.t();
    let inner_product_point = interior_point.dot(&edge_t);
    let inner_product_direction = direction.dot(&edge_t);
    let alpha = &inner_product_direction / &inner_product_point;
    let parent = alpha.argmax()?;
    let min = alpha[parent];
    debug_assert_eq!(
        alpha.iter().filter(|a| (*a - min).abs() < EPSILON).count(),
        1,
        "Found ties when searching for a parent"
    );
    return Ok(parent);
}

pub fn reverse_search<'a>(poly_list: &mut [FullPolytope], mut writer: Box<dyn FnMut(ReverseSearchOut) -> Result<()> + 'a>) -> Result<()> {
    let initial_state = mink_sum_setup(poly_list)?;
    let polys = initial_state.polys.clone();
    let num_edges: usize = polys.iter().map(|p| p.edges.len()).sum();
    info!("Total edges: {}", num_edges);
    // Allocate these arrays in advance
    let mut edges: Vec<ArrayView2<f64>> = Vec::with_capacity(num_edges);
    let mut edge_indices: Vec<AdjacencyTuple> = Vec::with_capacity(num_edges);

    let mut stack: Vec<ReverseSearchState> = Vec::new();
    stack.push(initial_state.clone());

    while !stack.is_empty() {
        let mut state = stack
            .pop()
            .ok_or(anyhow!("Empty stack! This should never happen"))?;
        if state.complete() {
            continue;
        }
        debug!(
            "Starting state: polytope {} vertex {} neighbour counter {}",
            state.polytope_index,
            state.vertex(),
            state.neighbour_index
        );
        edges.clear();


        let neighbours = state.neighbours();
        let test_vertex = neighbours[state.neighbour_index].1;

        let mut test_edge: Option<(usize, ArrayView1<f64>)> = None;
        let mut edge_counter: usize = 0;
        for (inner_polytope_index, (poly, inner_vertex)) in
            polys.iter().zip(&state.minkowski_decomp).enumerate()
        {
            for ((_vertex, neighbouring_vertex), edge) in poly.neighbouring_edges(*inner_vertex) {
                if inner_polytope_index == state.polytope_index && test_vertex == *neighbouring_vertex {
                    debug_assert_eq!(test_edge, None);
                    test_edge = Some((edge_counter, edge));
                }
                edges.push(edge.into_shape((1, edge.shape()[0]))?);
                edge_counter += 1;
            }
        }
        let (test_edge_index, test_edge) = test_edge.ok_or(anyhow!("No edge found!"))?;
        // Test if edge is leaving objective
        let score = state.param.dot(&test_edge);
        if score < -1. * EPSILON {
            debug!(
                "Skipping because edge is not leading away from objective. Score: {}",
                score
            );
            state.incr_state();
            stack.push(state);
            continue;
        }

        let edge_array: Array2<f64> = ndarray::concatenate(Axis(0), &edges)?;

        let is_adjacent = match lp_adjacency(&edge_array, test_edge_index)? {
            InteriorPointSolution::Feasible(_cert) => true,
            InteriorPointSolution::Infeasible => false,
        };
        if !is_adjacent {
            debug!("Skipping because edge is not adjacent");
            state.incr_state();
            stack.push(state);
            continue;
        }

        let mut test_decomp = state.minkowski_decomp.clone();
        test_decomp[state.polytope_index] = neighbours[state.neighbour_index].1;
        debug!("Test decomposition {:?}", &test_decomp);
        if test_decomp == initial_state.minkowski_decomp {
            debug!("Reached the initial state. Loop complete");
            continue;
        }

        //rebuild edges for new cone
        edges.clear();
        edge_indices.clear();
        for (inner_polytope_index, (poly, inner_vertex)) in
            polys.iter().zip(&test_decomp).enumerate()
        {
            for (adj, edge) in poly.neighbouring_edges(*inner_vertex) {
                edges.push(edge.into_shape((1, edge.shape()[0]))?);
                edge_indices.push(AdjacencyTuple {
                    polytope_index: inner_polytope_index,
                    _vertex: adj.0,
                    neighbour: adj.1,
                });
            }
        }
        debug_assert_eq!(edges.len(), edge_indices.len());
        let test_edges_array: Array2<f64> = ndarray::concatenate(Axis(0), &edges)?;

        let maximiser = lp_normal_cone(&test_edges_array)?;
        let maximiser_norm = &maximiser / maximiser.norm();
        
        debug!("Maximiser: {:?}", maximiser_norm);
        fn test_edges(edges: &Array2<f64>, maximiser: &Array1<f64>) -> Result<bool> {
            let scores = maximiser.dot(&edges.t()).map(|score| *score > EPSILON);
            let test_op = scores.iter().map(|t| *t).reduce(|acc, test| acc && test);
            let test = test_op.ok_or(anyhow!("No scores! - edge array must be empty"))?;
            if ! test {
                for (edge_id, score) in maximiser.dot(&edges.t()).iter().enumerate(){
                    if *score < EPSILON {
                        debug!("Score {} for edge {} is below epsilon", *score, edge_id);
                    }
                }
            }
            return Ok(test);
        }

        fn test_maximiser(
            polys: &[Polytope],
            decomp: &[usize],
            maximiser: &Array1<f64>,
        ) -> Result<bool> {
            let maximised = polys
                .iter()
                .map(|p| p.maximised_vertex(maximiser))
                .collect::<Result<Vec<_>>>()?;
            let equal_decomp = maximised
                .iter()
                .zip_eq(decomp)
                .map(|(left, right)| *left == *right)
                .reduce(|accm, test| accm && test)
                .ok_or(anyhow!("Empty decomp!"))?;
            if ! equal_decomp {
                for ((decomp_val, maximised_val), poly) in decomp.iter().zip_eq(maximised).zip_eq(polys){
                    if *decomp_val != maximised_val {
                        let scores = poly.scores(maximiser);
                        for (vertex, score) in scores {
                            debug!("{}, {}", vertex, score);
                        }
                    }
                }
            }
            return Ok(equal_decomp);
        }

        debug_assert!(
            // Some distance from edges are very small. The normalisation pushes them below epsilon. Use the raw maximiser instead
            test_edges(&test_edges_array, &maximiser)?,
            "Because we have tested for adjacency, we should always get a maximiser"
        );

        debug_assert!(
            test_maximiser(&polys, &test_decomp, &maximiser_norm)?,
            "Maximiser does not retrieve the decomposition"
        );

        let parent_edge = parent(&test_edges_array, &maximiser_norm, &initial_state.param)?;
        let parent_edge_index = &edge_indices[parent_edge];
        let mut parent = test_decomp.clone();
        parent[parent_edge_index.polytope_index] = parent_edge_index.neighbour;

        debug!("Parent decomp: {:?}", &parent);
        debug!("  Test decomp: {:?}", &test_decomp);
        debug!(" State decomp: {:?}", &state.minkowski_decomp);
        if parent == state.minkowski_decomp {
            // Reverse traverse
            let child_state = ReverseSearchState {
                polytope_index: 0,
                neighbour_index: 0,
                param: maximiser_norm,
                minkowski_decomp: test_decomp,
                polys: state.polys.clone()
            };
            let mut output = ReverseSearchOut{
                param: child_state.param.clone(),
                minkowski_decomp: child_state.minkowski_decomp.clone()
            };
            for (vertex, full_poly) in output.minkowski_decomp.iter_mut().zip(poly_list.as_ref()) {
                *vertex = full_poly
                    .map_essential_index(*vertex)
                    .ok_or(anyhow!("Essential vertex not found!"))?;
            }
            writer(output)?;
            state.incr_state();
            stack.push(state);
            stack.push(child_state);
        } else {
            state.incr_state();
            stack.push(state);
        }
    }
    return Ok(());
}
