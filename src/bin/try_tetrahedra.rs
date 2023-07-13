use molassembler::shapes::SQUAREANTIPRISM;
use molassembler::dg::*;
use molassembler::dg::refinement::{Chiral, SerialRefinement, Bounds};
use statrs::statistics::Statistics;

const REPETITIONS: usize = 100;

fn main() {
    let shape = &SQUAREANTIPRISM;
    let bounds = modeling::solitary_shape::shape_into_bounds(shape);
    let tetrahedra = shape.find_tetrahedra();
    let iterations: Vec<f64> = (0..REPETITIONS).filter_map(|_| {
        let distances = DistanceMatrix::try_from_distance_bounds(bounds.clone(), MetrizationPartiality::Complete).expect("Successful metrization");
        let metric = MetricMatrix::from(distances);
        let coords = metric.embed();
        let chirals: Vec<Chiral<f64>> = tetrahedra.iter()
            .map(|&tetr| modeling::solitary_shape::chiral_from_tetrahedron(tetr, shape, 0.1))
            .collect();
        let refinement_bounds = Bounds::new(bounds.clone(), chirals);
        let problem = SerialRefinement::new(refinement_bounds);
        refinement::refine(problem, coords).map(|results| results.steps as f64).ok()
    }).collect();

    println!("{} runs: mean iterations: {}", iterations.len(), iterations.mean());
}
