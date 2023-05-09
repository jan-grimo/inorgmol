use molassembler::shapes::TETRAHEDRON;
use molassembler::dg::*;
use molassembler::dg::refinement::{Chiral, SerialRefinement};

fn main() {
    let shape = &TETRAHEDRON;
    let bounds = modeling::solitary_shape::shape_into_bounds(shape);
    let distances = DistanceMatrix::try_from_distance_bounds(bounds.clone(), MetrizationPartiality::Complete).expect("Successful metrization");
    let metric = MetricMatrix::from_distance_matrix(distances);
    let coords = metric.embed();
    let chirals: Vec<Chiral> = shape.find_tetrahedra().into_iter()
        .map(|tetr| modeling::solitary_shape::chiral_from_tetrahedron(tetr, shape, 0.1))
        .collect();
    let problem = SerialRefinement::new(bounds, chirals);
    if let Ok(results) = refinement::refine(problem, coords) {
        println!("coords: {}", results.coords);
        println!("iterations: {}", results.steps);
    }
}
