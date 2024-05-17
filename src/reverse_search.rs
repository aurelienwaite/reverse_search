use anyhow::{anyhow, Result};
use good_lp::{
    constraint, default_solver, variables, Expression, ResolutionError, Solution, SolverModel,
};
use itertools::Itertools;
use log::{debug, info};
use ndarray::{concatenate, prelude::*};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::rc::Rc;
use std::usize;

// Used for the LP separation problem. Due to the formalisation of the LP the number will always
// be in the range [0,1]. Having a fixed epsilon is probably ok.
const EPSILON: f64 = 1e-6;

enum EssentialSolution {
    Essential(Array1<f64>), // Contains the certificate of the solution
    Redundant,
}

enum InteriorPointSolution {
    Feasible(Array1<f64>), // Contains the certificate of the solution
    Infeasible,
}

// Cannot use linalg on wasm because of BLAS. Build our own l2_norm function.
fn l2_norm(to_norm: &[f64]) -> f64 {
    to_norm
        .iter()
        .map(|v| v.powi(2))
        .fold(0., |l, r| l + r)
        .sqrt()
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
    to_test: &ArrayView1<f64>,
    vertices: &[ArrayView2<f64>], // For efficiency we often want to pass slices of a single matrix. Saves array copies
    linearity: &ArrayView2<f64>,
) -> Result<EssentialSolution, ResolutionError> {
    let mut vars = variables!();
    let dim = to_test.len();
    let mut x: Vec<good_lp::Variable> = Vec::with_capacity(dim);
    for _i in 0..dim {
        x.push(vars.add_variable());
    }
    let d = vars.add_variable();

    let objective = d + inner_product_lp_ndarray(&to_test, &x);
    let mut problem = vars
        .maximise(&objective)
        .using(default_solver)
        .with(constraint!(
            d + inner_product_lp_ndarray(&to_test, &x) <= 1.
        ));

    for slice in vertices {
        for vertex in (*slice).axis_iter(Axis(0)) {
            problem = problem.with(constraint!(d + inner_product_lp_ndarray(&vertex, &x) <= 0.));
        }
    }
    for generator in linearity.axis_iter(Axis(0)) {
        problem = problem.with(constraint!(
            d + inner_product_lp_ndarray(&generator, &x) == 0.
        ));
    }

    let solution = problem.solve()?;
    let maxima = solution.eval(&objective);
    let sep_solution = if maxima > EPSILON {
        let mut cert: Vec<f64> = Vec::with_capacity(x.len());
        cert.push(solution.value(d));
        cert.extend(x.iter().map(|variable| solution.value(*variable)));
        EssentialSolution::Essential(Array1::<f64>::from(cert))
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

fn adjacency(
    essential_vertices: &Array2<f64>,
    indices: &BTreeSet<usize>,
) -> Result<HashMap<usize, Vec<usize>>> {
    let index_map = Vec::from_iter(indices); // Maps the essential index back to the original indices
    let mut res: HashMap<usize, Vec<usize>> = HashMap::with_capacity(index_map.len());

    for (index, _vertex) in essential_vertices.axis_iter(Axis(0)).enumerate() {
        let linearity: ArrayView2<f64> = essential_vertices.slice(s![index..index + 1, ..]);
        debug!("Finding adjacent");
        let adjacent = find_essential(&essential_vertices.view(), &linearity)?;
        res.insert(
            *index_map[index],
            Vec::from_iter(adjacent.0.iter().map(|adj| *index_map[*adj])),
        );
    }
    debug!("{:?}", res);
    return Ok(res);
}

fn find_essential(
    vertices: &ArrayView2<f64>,
    linearity: &ArrayView2<f64>,
) -> Result<(BTreeSet<usize>, HashMap<usize, Array1<f64>>)> {
    let mut essential: BTreeSet<usize> = BTreeSet::new();
    let mut certs: HashMap<usize, Array1<f64>> = HashMap::new();
    for (index, vertex) in vertices.axis_iter(Axis(0)).enumerate() {
        let before = vertices.slice(s![0..index, ..]);
        let after = vertices.slice(s![index + 1.., ..]);
        let other = [before, after];
        let lp_res = lp_separation(&vertex, &other, &linearity.view());
        if lp_res.as_ref().is_err() {
            let error_msg = lp_res
                .as_ref()
                .map_err(|err| err.to_string())
                .err()
                .unwrap_or(String::from("No message"));
            info!("Error finding essential vertex, skipping. {}", error_msg);
            continue;
        }
        let lp_solution = lp_res?;
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

#[derive(Debug)]
pub struct Polytope {
    pub vertices: Array2<f64>,
    essential_indices: Option<BTreeSet<usize>>,
    adjacency: Option<HashMap<usize, Vec<usize>>>,
    essential_certs: Option<HashMap<usize, Array1<f64>>>,
}

// A polytope with essential data only
struct EssentialPolytope {
    vertices: Array2<f64>, // Transposed - dim x vertices
    flattened_adjacency: Vec<(usize, usize)>,
    edges: Array2<f64>,
    adjacency: Vec<Vec<usize>>,
}

impl EssentialPolytope {
    fn neighbouring_edges<'a>(
        self: &'a EssentialPolytope,
        vertex: usize,
    ) -> Box<dyn Iterator<Item = (&(usize, usize), ArrayView1<f64>)> + 'a> {
        let iterator = self
            .flattened_adjacency
            .iter()
            .zip_eq(self.edges.axis_iter(Axis(0)))
            .filter(move |tup| tup.0 .0 == vertex);
        return Box::new(iterator);
    }

    fn maximised_vertex(self: &EssentialPolytope, param: &Array1<f64>) -> Result<usize> {
        return Ok(self.vertices.dot(param).argmax()?);
    }

    fn _scores(self: &EssentialPolytope, param: &Array1<f64>) -> Vec<(usize, f64)> {
        let scores = param.dot(&self.vertices);
        return scores
            .iter()
            .enumerate()
            .sorted_by(|(_ai, a_score), (_bi, b_score)| {
                if *a_score - *b_score < 0. {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .map(|(i, s)| (i, *s))
            .collect::<Vec<_>>();
    }
}

#[derive(Clone)]
struct ReverseSearchState {
    param: Array1<f64>,
    previous_polytope_index: usize,
    previous_vertex: usize,
    neighbour_index: usize,
    polytope_index: usize,
    // Vertex is a function of polytope index and the minkowski decomp.
    // We cache it in the state to make keeping track of diffs easier.
    // It must be kept in sync with the minkowski decomp
    vertex: usize,
    polys: Rc<Vec<EssentialPolytope>>,
}

pub struct ReverseSearchOut {
    pub param: Array1<f64>,
    pub minkowski_decomp: Vec<usize>,
}

impl ReverseSearchState {
    fn polytope(&self) -> &EssentialPolytope {
        return &self.polys[self.polytope_index];
    }

    fn neighbours(&self) -> &Vec<usize> {
        return &self.polytope().adjacency[self.vertex];
    }

    fn complete(&self) -> bool {
        return self.polytope_index >= self.polys.len();
    }

    fn test_vertex(&self) -> usize {
        self.neighbours()[self.neighbour_index]
    }

    fn incr_state(&mut self, minkowski_decomp: &Vec<usize>) -> () {
        self.neighbour_index += 1;
        if self.neighbour_index >= self.neighbours().len() {
            self.polytope_index += 1;
            self.neighbour_index = 0;
            if !self.complete() {
                self.vertex = minkowski_decomp[self.polytope_index];
            }
        }
    }

    fn make_child(&self, vertex: usize, param: Array1<f64>) -> ReverseSearchState {
        ReverseSearchState {
            previous_polytope_index: self.polytope_index,
            previous_vertex: self.vertex,
            param,
            polys: self.polys.clone(),
            neighbour_index: 0,
            polytope_index: 0,
            vertex,
        }
    }

    fn make_minkowski_decomp(&self) -> Result<Vec<usize>> {
        let decomp = self
            .polys
            .iter()
            .map(|p| p.vertices.dot(&self.param).argmax())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        debug!("Minkowski decomposition {:?}", &decomp);
        return Ok(decomp);
    }
}

fn make_state(param: Array1<f64>, polys: Rc<Vec<EssentialPolytope>>) -> Result<ReverseSearchState> {
    let vertex = polys[0].maximised_vertex(&param)?;
    let state = ReverseSearchState {
        param,
        polytope_index: 0,
        vertex,
        neighbour_index: 0,
        previous_polytope_index: 0,
        previous_vertex: vertex,
        polys: polys.clone(),
    };
    Ok(state)
}

#[derive(Debug)]
struct AdjacencyTuple {
    polytope_index: usize,
    _vertex: usize,
    neighbour: usize,
}

#[derive(Debug)]
struct DecompositionIndex {
    polytope_index: usize,
    vertex: usize,
}

impl DecompositionIndex {
    fn new(polytope_index: usize, vertex: usize) -> Self {
        Self {
            polytope_index,
            vertex,
        }
    }
}

impl Polytope {
    pub fn new(vertices: Array2<f64>) -> Polytope {
        Polytope {
            vertices,
            essential_indices: None,
            essential_certs: None,
            adjacency: None,
        }
    }

    fn fill_essential(&mut self) -> Result<()> {
        let empty = Array2::<f64>::zeros((0, 0));
        let essential = find_essential(&self.vertices.view(), &empty.view())?;
        self.essential_indices = Option::Some(essential.0);
        self.essential_certs = Option::Some(essential.1);
        return Ok(());
    }

    fn essential_vertices(&self) -> Option<Array2<f64>> {
        return self.essential_indices.as_ref().map(|indices| {
            let as_slice = indices.iter().map(|i| *i).collect::<Vec<_>>();
            self.vertices.select(Axis(0), &as_slice)
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

    fn essential_polytope(&self) -> Result<Option<EssentialPolytope>> {
        let essential = self.essential_indices.as_ref().and_then(|indices| {
            self.adjacency.as_ref().map(|adjacency| {
                let mut mapping: HashMap<usize, usize> = HashMap::new();
                for (essential, raw) in indices.iter().enumerate() {
                    mapping.insert(*raw, essential);
                }
                let indices_as_slice = indices.iter().map(|i| *i).collect::<Vec<_>>();
                let essential_vertices = self.vertices.select(Axis(0), &indices_as_slice);
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
        let poly: Option<Result<EssentialPolytope>> = essential.map(|(vertices, adjacency)| {
            let mut edges: Vec<Array1<f64>> = Vec::new();
            let mut flattened_adjacency: Vec<(usize, usize)> = Vec::new();
            debug!("Adjacency {:?}", flattened_adjacency);
            debug!("Vertices shape {:?}", vertices.shape());
            for (current_vertex, neighbours) in &adjacency {
                for neighbour in neighbours {
                    let edge = &vertices.index_axis(Axis(0), *current_vertex)
                        - &vertices.index_axis(Axis(0), *neighbour);
                    edges.push(edge);
                    flattened_adjacency.push((*current_vertex, *neighbour));
                }
            }
            let expanded_dims = edges
                .iter()
                .map(|e| {
                    let dim = e.shape()[0];
                    return e.view().into_shape((1, dim));
                })
                .collect::<Result<Vec<_>, _>>()?;
            debug!("edges {:?}", expanded_dims);
            let mut edge_array: Array2<f64> = ndarray::concatenate(Axis(0), &expanded_dims)?;
            let vec_adjacency = adjacency
                .iter()
                .map(|(_vertex, neighbours)| neighbours.iter().map(|n| *n).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            for mut row in edge_array.rows_mut() {
                let row_slice = row
                    .as_slice()
                    .ok_or(anyhow!("Incorrect memory layout in edges"))?;
                let norm = l2_norm(row_slice);
                row.mapv_inplace(|v| v / norm);
            }
            let polytope = EssentialPolytope {
                adjacency: vec_adjacency,
                vertices: vertices,
                flattened_adjacency,
                edges: edge_array,
            };

            fn test_polytope(poly: &EssentialPolytope) -> bool {
                let mut correct = true;
                let mut flat_index: usize = 0;
                for (vertex, neighbours) in poly.adjacency.iter().enumerate() {
                    for neighbour in neighbours {
                        let flat = &poly.flattened_adjacency[flat_index];
                        correct = correct && flat.0 == vertex && flat.1 == *neighbour;
                        flat_index += 1;
                    }
                }
                correct
            }
            debug_assert!(test_polytope(&polytope));
            return Ok(polytope);
        });
        return poly.transpose();
    }

    // Map the essential indices back to full polytope raw indices before exporting
    fn map_essential_index(self: &Polytope, index: usize) -> Option<usize> {
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

fn mink_sum_setup<'a>(poly_list: &mut [Polytope]) -> Result<ReverseSearchState> {
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
        .map(|p| p.vertices.shape()[1])
        .ok_or(anyhow!("No polytopes!"))?;

    let initial_param = Array1::<f64>::ones(dim);
    let mut state: ReverseSearchState = make_state(initial_param, polys_ref)?;

    // Checking to see if any vertices in a polytope have the same score
    // If they do the reverse search will fail. We break times by perturbing the parameter
    // vector
    // Probably a degenerate condition that will never happen.
    let mut ties = true;
    while ties {
        ties = false;
        for (i, polytope) in state.polys.iter().enumerate() {
            debug!("Testing polytope {} for ties with initial param", i);
            let scores = state.param.dot(&polytope.vertices.t());
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

fn decomp_iter<'a>(
    decomp: impl Iterator<Item = &'a usize> + 'a,
    decomp_index: &'a DecompositionIndex,
) -> impl Iterator<Item = &'a usize> + 'a {
    decomp.enumerate().map(|(index, vertex)| {
        if decomp_index.polytope_index == index {
            &decomp_index.vertex
        } else {
            vertex
        }
    })
}

pub fn reverse_search<'a>(
    poly_list: &mut [Polytope],
    mut writer: Box<dyn FnMut(ReverseSearchOut) -> Result<()> + 'a>,
) -> Result<()> {
    let initial_state = mink_sum_setup(poly_list)?;
    let polys = initial_state.polys.clone();
    let num_edges: usize = polys.iter().map(|p| p.edges.len()).sum();
    info!("Total edges: {}", num_edges);
    // Allocate these arrays in advance
    let mut edges: Vec<ArrayView2<f64>> = Vec::with_capacity(num_edges);
    let mut edge_indices: Vec<AdjacencyTuple> = Vec::with_capacity(num_edges);

    //Use a single decomposition vector. These could get quite large
    let initial_decomp = initial_state.make_minkowski_decomp()?;
    let mut minkowski_decomp = initial_decomp.clone();

    let mut stack: Vec<ReverseSearchState> = Vec::new();
    stack.push(initial_state.clone());

    while !stack.is_empty() {
        let mut state = stack
            .pop()
            .ok_or(anyhow!("Empty stack! This should never happen"))?;
        if state.complete() {
            debug!(
                "Popping, restoring element {} to {}",
                state.previous_polytope_index, state.previous_vertex
            );
            minkowski_decomp[state.previous_polytope_index] = state.previous_vertex;
            continue;
        }
        debug!(
            "Starting state: polytope {} vertex {} neighbour counter {}",
            state.polytope_index, state.vertex, state.neighbour_index
        );
        edges.clear();

        let test_vertex = state.test_vertex();
        let mut test_edge: Option<(usize, ArrayView1<f64>)> = None;
        let mut edge_counter: usize = 0;
        for (inner_polytope_index, (poly, inner_vertex)) in
            polys.iter().zip(&minkowski_decomp).enumerate()
        {
            for ((_vertex, neighbouring_vertex), edge) in poly.neighbouring_edges(*inner_vertex) {
                if inner_polytope_index == state.polytope_index
                    && test_vertex == *neighbouring_vertex
                {
                    debug_assert_eq!(test_edge, None);
                    test_edge = Some((edge_counter, edge));
                }
                edges.push(edge.into_shape((1, edge.shape()[0]))?);
                edge_counter += 1;
            }
        }
        let (test_edge_index, _test_edge) = test_edge.ok_or(anyhow!("No edge found!"))?;

        let edge_array: Array2<f64> = ndarray::concatenate(Axis(0), &edges)?;

        let is_adjacent = match lp_adjacency(&edge_array, test_edge_index)? {
            InteriorPointSolution::Feasible(_cert) => true,
            InteriorPointSolution::Infeasible => false,
        };
        if !is_adjacent {
            debug!("Skipping because edge is not adjacent");
            state.incr_state(&minkowski_decomp);
            stack.push(state);
            continue;
        }

        let test_decomp_index = DecompositionIndex::new(state.polytope_index, test_vertex);
        let test_iter = || decomp_iter(minkowski_decomp.iter(), &test_decomp_index);

        if test_iter().eq(initial_decomp.iter()) {
            debug!("Reached the initial state. Loop complete");
            state.incr_state(&minkowski_decomp);
            stack.push(state);
            continue;
        }

        //rebuild edges for new cone
        edges.clear();
        edge_indices.clear();
        for (inner_polytope_index, (poly, inner_vertex)) in
            polys.iter().zip(test_iter()).enumerate()
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
        let norm = l2_norm(maximiser.as_slice().ok_or(anyhow!("Incorrect memory"))?);
        let maximiser_norm = &maximiser / norm;

        debug!("Maximiser: {:?}", maximiser_norm);
        fn test_edges(edges: &Array2<f64>, maximiser: &Array1<f64>) -> Result<bool> {
            let scores = maximiser.dot(&edges.t()).map(|score| *score > EPSILON);
            let test_op = scores.iter().map(|t| *t).reduce(|acc, test| acc && test);
            let test = test_op.ok_or(anyhow!("No scores! - edge array must be empty"))?;
            if !test {
                for (edge_id, score) in maximiser.dot(&edges.t()).iter().enumerate() {
                    if *score < EPSILON {
                        debug!("Score {} for edge {} is below epsilon", *score, edge_id);
                    }
                }
            }
            return Ok(test);
        }

        fn test_maximiser(
            polys: &[EssentialPolytope],
            decomp: impl Iterator<Item = usize>,
            maximiser: &Array1<f64>,
        ) -> Result<bool> {
            let maximised = polys
                .iter()
                .map(|p| p.maximised_vertex(maximiser))
                .collect::<Result<Vec<_>>>()?;
            let equal_decomp = maximised
                .iter()
                .zip_eq(decomp)
                .map(|(left, right)| *left == right)
                .reduce(|accm, test| accm && test)
                .ok_or(anyhow!("Empty decomp!"))?;
            // Commenting out at the moment because the decomp iterator can't be called twice.
            // Will figure out later. Probably safe to delete this test
            /* if !equal_decomp {
                for ((decomp_val, maximised_val), poly) in
                    decomp.iter().zip_eq(maximised).zip_eq(polys)
                {
                    if *decomp_val != maximised_val {
                        let scores = poly.scores(maximiser);
                        for (vertex, score) in scores {
                            debug!("{}, {}", vertex, score);
                        }
                    }
                }
            }*/
            return Ok(equal_decomp);
        }

        debug_assert!(
            // Some distance from edges are very small. The normalisation pushes them below epsilon. Use the raw maximiser instead
            test_edges(&test_edges_array, &maximiser)?,
            "Because we have tested for adjacency, we should always get a maximiser"
        );

        debug_assert!(
            test_maximiser(&polys, test_iter().map(|v| *v), &maximiser_norm)?,
            "Maximiser does not retrieve the decomposition"
        );

        let parent_edge = parent(&test_edges_array, &maximiser_norm, &initial_state.param)?;
        let parent_edge_index = &edge_indices[parent_edge];
        let parent_decomp_index = DecompositionIndex::new(
            parent_edge_index.polytope_index,
            parent_edge_index.neighbour,
        );

        debug!(
            "Parent decomp: {:?}",
            decomp_iter(test_iter(), &parent_decomp_index).collect::<Vec<_>>()
        );
        debug!("  Test decomp: {:?}", test_iter().collect::<Vec<_>>());
        debug!(" State decomp: {:?}", &minkowski_decomp);
        if test_decomp_index.polytope_index == parent_decomp_index.polytope_index
            && minkowski_decomp[parent_decomp_index.polytope_index] == parent_decomp_index.vertex
        {
            // Reverse traverse
            debug!("Reverse traverse");
            let child_vertex = test_iter()
                .next()
                .ok_or(anyhow!("Test decomposition is empty"))?;
            let child_state = state.make_child(*child_vertex, maximiser_norm);
            state.incr_state(&minkowski_decomp);
            minkowski_decomp[test_decomp_index.polytope_index] = test_decomp_index.vertex;
            stack.push(state);

            // TODO: Fix the writing code to take references to decompositions, allowing for deletion of this clone
            let output_decomp = minkowski_decomp.clone();
            let mut output = ReverseSearchOut {
                param: child_state.param.clone(),
                minkowski_decomp: output_decomp,
            };
            for (vertex, full_poly) in output.minkowski_decomp.iter_mut().zip(poly_list.as_ref()) {
                *vertex = full_poly
                    .map_essential_index(*vertex)
                    .ok_or(anyhow!("Essential vertex not found!"))?;
            }

            writer(output)?;

            stack.push(child_state);
        } else {
            state.incr_state(&minkowski_decomp);
            stack.push(state);
        }
    }
    return Ok(());
}
