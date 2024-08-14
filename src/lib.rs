use anyhow::{anyhow, Result};
use good_lp::{
    constraint, default_solver, variables, Expression, ResolutionError, Solution, SolverModel,
};
use instant::Instant;
use itertools::Itertools;
use log::{debug, info};
use ndarray::{concatenate, prelude::*};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{Read, Write};
use std::usize;
use flate2::Compression;
use flate2::write::ZlibEncoder;
use flate2::read::ZlibDecoder;
use bytes::Buf;
use bytes::buf::Reader;

pub mod guided_search;

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

#[derive(Clone)]
pub enum StepResult {
    Nonadjacent,
    NotAChild,
    Child(Array1<f64>),
    LoopComplete,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TreeIndex {
    pub polytope_id: usize,
    pub vertex_id: usize,
}

impl TreeIndex {
    pub fn new(polytope_id: usize, vertex_id: usize) -> Self {
        TreeIndex {
            polytope_id,
            vertex_id,
        }
    }
}

// Cannot use linalg on wasm because of BLAS. Build our own l2_norm function.
fn l2_norm(to_norm: &[f64]) -> f64 {
    to_norm
        .iter()
        .map(|v| v.powi(2))
        .fold(0., |l, r| l + r)
        .sqrt()
}

fn inner_product_lp(coefficients: &ArrayView1<f64>, variables: &[good_lp::Variable]) -> Expression {
    return coefficients
        .iter()
        .zip_eq(variables)
        .filter_map(|item| {
            if item.0.abs() > EPSILON {
                Some(*item.0 * *item.1)
            } else 
            {
                None
            } 
        })
        .sum();
}

fn minkowski_decomp(polys: &[EssentialPolytope], param: &Array1<f64>) -> Result<Vec<usize>> {
    let decomp = polys
        .iter()
        .map(|p| p.vertices.dot(param).argmax())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    debug!("Minkowski decomposition {:?}", &decomp);
    Ok(decomp)
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

    let objective = d + inner_product_lp(&to_test, &x);
    let mut problem = vars
        .maximise(&objective)
        .using(default_solver)
        .with(constraint!(d + inner_product_lp(&to_test, &x) <= 1.));

    for slice in vertices {
        for vertex in (*slice).axis_iter(Axis(0)) {
            problem = problem.with(constraint!(d + inner_product_lp(&vertex, &x) <= 0.));
        }
    }
    for generator in linearity.axis_iter(Axis(0)) {
        problem = problem.with(constraint!(d + inner_product_lp(&generator, &x) == 0.));
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
        let constraint = inner_product_lp(&row, &x);
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
        match lp_solution {
            EssentialSolution::Essential(cert) => {
                certs.insert(index, cert);
                essential.insert(index)
            }
            EssentialSolution::Redundant => false,
        };
    }
    return Ok((essential, certs));
}

#[derive(Clone)]
struct State<'a> {
    param: Array1<f64>,
    neighbour_index: usize,
    polys: &'a [EssentialPolytope],
    node: TreeNode,
}

#[derive(Clone)]
struct TreeNode {
    // The Vertex in the tree index is a function of polytope index and the minkowski decomp.
    // We cache it in the state to make keeping track of diffs easier.
    // It must be kept in sync with the minkowski decomp
    current: TreeIndex,
    previous: TreeIndex,
}

impl State<'_> {
    fn new<'a>(
        param: Array1<f64>,
        polys: &'a [EssentialPolytope],
    ) -> Result<State<'a>> {
        let vertex = polys[0].maximised_vertex(&param)?;
        let state = State {
            param,
            neighbour_index: 0,
            polys: polys,
            node: TreeNode {
                current: TreeIndex::new(0, vertex),
                previous: TreeIndex::new(0, vertex),
            },
        };
        Ok(state)
    }

    fn polytope(&self) -> &EssentialPolytope {
        return &self.polys[self.node.current.polytope_id];
    }

    fn neighbours(&self) -> &Vec<usize> {
        return &self.polytope().adjacency[self.node.current.vertex_id];
    }

    fn complete(&self) -> bool {
        return self.node.current.polytope_id >= self.polys.len();
    }

    fn test_vertex(&self) -> usize {
        self.neighbours()[self.neighbour_index]
    }

    fn incr_state(mut self, minkowski_decomp: &Vec<usize>) -> Self {
        self.neighbour_index += 1;
        if self.neighbour_index >= self.neighbours().len() {
            self.node.current.polytope_id += 1;
            self.neighbour_index = 0;
            if !self.complete() {
                self.node.current.vertex_id =
                    minkowski_decomp[self.node.current.polytope_id];
            }
        };
        self
    }
}

pub struct ReverseSearchConfig {
    // If supplied with training labels, only
    // test for adjacency for neighbours that will
    // improve accuracy
    pub greedy_search: bool,
}

#[derive(Debug)]
pub struct Polytope {
    pub vertices: Array2<f64>,
    essential_indices: Option<BTreeSet<usize>>,
    adjacency: Option<HashMap<usize, Vec<usize>>>,
    essential_certs: Option<HashMap<usize, Array1<f64>>>,
}

// A polytope with essential data only
#[derive(Clone)]
struct EssentialPolytope {
    vertices: Array2<f64>, // Transposed - dim x vertices
    flattened_adjacency: Vec<(usize, usize)>,
    edges: Array2<f64>,
    adjacency: Vec<Vec<usize>>,
    mapping: HashMap<usize, usize>, // Maps essential to raw indices
}

impl EssentialPolytope {
    fn neighbouring_edges(
        self: &EssentialPolytope,
        vertex: usize,
    ) -> impl Iterator<Item = (&(usize, usize), ArrayView1<f64>)> {
        self.flattened_adjacency
            .iter()
            .zip_eq(self.edges.axis_iter(Axis(0)))
            .filter(move |tup| tup.0 .0 == vertex)
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

pub struct ReverseSearchOut {
    pub param: Array1<f64>,
    pub compressed_decomp: Vec<u8>,
}

impl ReverseSearchOut {

    pub fn encode(param: Array1<f64>, minkowski_decomp: impl Iterator<Item = usize>) -> Result<Self>{
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        for vertex_id in minkowski_decomp{
            let bytes_representation = vertex_id.to_ne_bytes();
            encoder.write(&bytes_representation)?;
        }
        Ok(ReverseSearchOut{
            param,
            compressed_decomp: encoder.finish()?
        })
    }

    pub fn minkowski_decomp_iter(& self) -> MinkDecompIter{
        MinkDecompIter{
            decoder: ZlibDecoder::new(self.compressed_decomp.reader())
        } 
    }

}

pub struct MinkDecompIter<'a> {
    decoder: ZlibDecoder<Reader<&'a [u8]>>,
}

impl Iterator for MinkDecompIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = [0u8; std::mem::size_of::<usize>()];
        let read_res = self.decoder.read(&mut buf);
        read_res.ok().and_then(|num_read|{
            if num_read != buf.len() {
                None
            }else{
                Some(usize::from_ne_bytes(buf))
            }
        })
    }
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

    pub fn fill(&mut self) -> Result<()> {
        self.fill_essential()?;
        self.fill_adjacency()
    }

    fn essential_polytope(&self) -> Option<EssentialPolytope> {
        let essential = self.essential_indices.as_ref().and_then(|indices| {
            self.adjacency.as_ref().map(|adjacency| {
                let mut inverse_mapping: HashMap<usize, usize> = HashMap::new();
                for (essential, raw) in indices.iter().enumerate() {
                    inverse_mapping.insert(*raw, essential);
                }
                let indices_as_slice = indices.iter().map(|i| *i).collect::<Vec<_>>();
                let essential_vertices = self.vertices.select(Axis(0), &indices_as_slice);
                let mut essential_adjacency: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
                for (vertex, neighbours) in adjacency {
                    essential_adjacency.insert(
                        inverse_mapping[&vertex],
                        neighbours.iter().map(|i| inverse_mapping[i]).collect(),
                    );
                }
                (essential_vertices, essential_adjacency, inverse_mapping)
            })
        });
        let poly: Option<EssentialPolytope> = essential.map(|(vertices, adjacency, inverse_mapping)| {
            let mut edges: Vec<Array1<f64>> = Vec::new();
            let mut flattened_adjacency: Vec<(usize, usize)> = Vec::new();
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
                    return e.view().into_shape((1, dim)).expect("Cannot reshape array");
                })
                .collect::<Vec<_>>();
            let mut edge_array: Array2<f64> = ndarray::concatenate(Axis(0), &expanded_dims).expect("Cannot concat edges");
            let vec_adjacency = adjacency
                .iter()
                .map(|(_vertex, neighbours)| neighbours.iter().map(|n| *n).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            for mut row in edge_array.rows_mut() {
                let row_slice = row
                    .as_slice()
                    .expect("Incorrect memory layout in edges");
                let norm = l2_norm(row_slice);
                row.mapv_inplace(|v| v / norm);
            }
            let mut mapping: HashMap<usize, usize> = HashMap::new();
            for (raw, essential) in inverse_mapping.iter(){
                mapping.insert(*essential, *raw);
            }
            let polytope = EssentialPolytope {
                adjacency: vec_adjacency,
                vertices: vertices,
                flattened_adjacency,
                edges: edge_array,
                mapping
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
            return polytope;
        });
        return poly;
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

fn setup(poly_list: &[Polytope]) -> Result<(Vec<EssentialPolytope>, Array1<f64>)> {
    let essential_polys = poly_list
        .iter()
        .map(|p| p.essential_polytope())
        .collect::<Vec<_>>()
        .into_iter()
        .map(|p| p.ok_or(anyhow!("No item")))
        .collect::<Result<Vec<_>, _>>()?;

    //let polys_ref = Rc::new(essential_polys);

    let dim = essential_polys
        .first()
        .map(|p| p.vertices.shape()[1])
        .ok_or(anyhow!("No polytopes!"))?;

    let mut initial_param = Array1::<f64>::zeros(dim);
    // As a convention set the parameter score to be the last value
    initial_param[dim - 1] = 1.0;

    // Checking to see if any vertices in a polytope have the same score
    // If they do the reverse search will fail. We break times by perturbing the parameter
    // vector
    // Probably a degenerate condition that will never happen.
    let mut ties = true;
    while ties {
        ties = false;
        for (i, polytope) in essential_polys.iter().enumerate() {
            debug!("Testing polytope {} for ties with initial param", i);
            let scores = initial_param.dot(&polytope.vertices.t());
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
                initial_param = initial_param + perturb;
                break;
            }
        }
    }
    info!("Setup done");
    return Ok((essential_polys, initial_param));
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

#[derive(Clone)]
pub struct Searcher {
    polys: Vec<EssentialPolytope>,
    initial_decomp: Vec<usize>,
    pub minkowski_decomp: Vec<usize>,
    initial_param: Array1<f64>,
}

impl Searcher {
    pub fn setup_reverse_search(poly_list: &[Polytope]) -> Result<Self> {
        let (polys, initial_param) = setup(poly_list)?;
        let initial_decomp = minkowski_decomp(&polys, &initial_param)?;

        Ok(Searcher {
            polys,
            minkowski_decomp: initial_decomp.clone(),
            initial_decomp,
            initial_param,
        })
    }

    fn step(
        &self,
        prev_index: &TreeIndex,
        test_vertex: usize,
    ) -> Result<StepResult> {
        let mut edges: Vec<ArrayView2<f64>> = Vec::new();
        let mut test_edge: Option<(usize, ArrayView1<f64>)> = None;
        let mut edge_counter: usize = 0;
        for (inner_polytope_index, (poly, inner_vertex)) in
            self.polys.iter().zip(&self.minkowski_decomp).enumerate()
        {
            for ((_vertex, neighbouring_vertex), edge) in poly.neighbouring_edges(*inner_vertex) {
                if inner_polytope_index == prev_index.polytope_id
                    && test_vertex == *neighbouring_vertex
                {
                    debug_assert_eq!(test_edge, None);
                    test_edge = Some((edge_counter, edge));
                }
                edges.push(edge.into_shape((1, edge.shape()[0]))?);
                edge_counter += 1;
            }
        }
        debug!("Num edges {}", edges.len());
        let (test_edge_index, _test_edge) = test_edge.ok_or(anyhow!("No edge found!"))?;

        let edge_array: Array2<f64> = ndarray::concatenate(Axis(0), &edges)?;

        let now = Instant::now();
        let is_adjacent = match lp_adjacency(&edge_array, test_edge_index)? {
            InteriorPointSolution::Feasible(_cert) => true,
            InteriorPointSolution::Infeasible => false,
        };
        debug!("Adjacency computation took {}ms", now.elapsed().as_millis());
        if !is_adjacent {
            debug!("Skipping because edge is not adjacent");
            return Ok(StepResult::Nonadjacent);
        }

        let test_decomp_index = DecompositionIndex::new(prev_index.polytope_id, test_vertex);
        let test_iter = || decomp_iter(self.minkowski_decomp.iter(), &test_decomp_index);

        if test_iter().eq(self.initial_decomp.iter()) {
            debug!("Reached the initial state. Loop complete");
            return Ok(StepResult::LoopComplete);
        }

        //rebuild edges for new cone
        edges.clear();
        let mut edge_indices: Vec<AdjacencyTuple> = Vec::with_capacity(edges.capacity());
        for (inner_polytope_index, (poly, inner_vertex)) in
            self.polys.iter().zip(test_iter()).enumerate()
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
            test_maximiser(&self.polys, test_iter().map(|v| *v), &maximiser_norm)?,
            "Maximiser does not retrieve the decomposition"
        );

        let parent_edge = parent(&test_edges_array, &maximiser_norm, &self.initial_param)?;
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
        debug!(" State decomp: {:?}", &self.minkowski_decomp);
        let step_result = if test_decomp_index.polytope_index == parent_decomp_index.polytope_index
            && self.minkowski_decomp[parent_decomp_index.polytope_index]
                == parent_decomp_index.vertex
        {
            StepResult::Child(maximiser_norm)
        } else {
            StepResult::Nonadjacent
        };

        Ok(step_result)
    }
}

pub fn reverse_search<'a>(
    poly_list: &mut [Polytope],
    mut writer: Box<dyn FnMut(ReverseSearchOut) -> Result<()> + 'a>,
) -> Result<()> {
    debug!("poly_list len {}", poly_list.len());
    for poly in poly_list.as_mut() {
        poly.fill_essential()?;
        poly.fill_adjacency()?;
    }
    let mut search = Searcher::setup_reverse_search(poly_list)?;
    let initial_state: State =
        State::new(search.initial_param.clone(), &search.polys)?;

    let mut stack: Vec<State> = Vec::new();
    stack.push(initial_state.clone());

    while !stack.is_empty() {
        {
            let state = stack
                .pop()
                .ok_or(anyhow!("Empty stack! This should never happen"))?;
            if state.complete() {
                debug!(
                    "Popping, restoring element {} to {}",
                    state.node.previous.polytope_id,
                    state.node.previous.vertex_id
                );
                search.minkowski_decomp[state.node.previous.polytope_id] =
                    state.node.previous.vertex_id;
                continue;
            }
            debug!(
                "Starting state: polytope {} vertex {} neighbour counter {}",
                state.node.current.polytope_id,
                state.node.current.vertex_id,
                state.neighbour_index
            );

            let test_vertex = state.test_vertex();

            match search.step(&state.node.current, test_vertex)? {
                StepResult::Nonadjacent => {
                    debug!("Skipping because edge is not adjacent");
                    stack.push(state.incr_state(&search.minkowski_decomp));
                }
                StepResult::LoopComplete => {
                    debug!("Reached the initial state. Loop complete");
                    stack.push(state.incr_state(&search.minkowski_decomp));
                }
                StepResult::NotAChild => {
                    debug!("Adjacency passed but previous node is not a parent");
                    stack.push(state.incr_state(&search.minkowski_decomp));
                }
                StepResult::Child(param) => {
                    // Reverse traverse
                    let initial_vertex = if state.node.current.polytope_id == 0 {
                        state.test_vertex()
                    } else {
                        search.minkowski_decomp[0]
                    };
                    let child_state = State {
                        param,
                        polys: state.polys,
                        neighbour_index: 0,
                        node: TreeNode {
                            current: TreeIndex::new(0, initial_vertex),
                            previous: state.node.current.clone(),
                        },
                    };
                    let test_polytope_index = state.node.current.polytope_id;
                    let incr = state.incr_state(&search.minkowski_decomp);
                    search.minkowski_decomp[test_polytope_index] = test_vertex;
                    stack.push(incr);

                    let mapped_iter = search.minkowski_decomp.iter().zip(poly_list.as_ref()).map(
                        |(vertex, full_poly)| {
                        full_poly
                            .map_essential_index(*vertex)
                            .expect("Essential vertex not found!")
                        }
                    );
                    let output = ReverseSearchOut::encode(
                        child_state.param.clone(),
                        mapped_iter,
                    )?;
                    writer(output)?;
                    stack.push(child_state);
                }
            }
        }
    }
    Ok(())
}
