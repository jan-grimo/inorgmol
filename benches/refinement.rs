use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use molassembler::dg::refinement::RefinementErrorFunction;
use molassembler::*;
extern crate nalgebra as na;

fn refinement(c: &mut Criterion) {
    let shape = &shapes::SQUAREANTIPRISM;
    let bounds = dg::modeling::solitary_shape::shape_into_bounds(shape);
    let tetrahedra = shape.find_tetrahedra();
    let distances = dg::DistanceMatrix::try_from_distance_bounds(bounds.clone(), dg::MetrizationPartiality::Complete).expect("Successful metrization");
    let metric = dg::MetricMatrix::from_distance_matrix(distances);
    let coords = metric.embed();
    let n = coords.len();
    let linear_positions = coords.reshape_generic(na::Dyn(n), na::Const::<1>);
    let chirals: Vec<dg::refinement::Chiral> = tetrahedra.iter()
        .map(|&tetr| dg::modeling::solitary_shape::chiral_from_tetrahedron(tetr, shape, 0.1))
        .collect();

    let mut bench_group = c.benchmark_group("refinement");
    let refinement_n = (shape.size() + 1) as u64;
    bench_group.throughput(Throughput::Elements(refinement_n));

    let refinement_bounds = dg::refinement::Bounds::new(bounds, chirals);

    let serial = dg::refinement::SerialRefinement {bounds: refinement_bounds.clone(), stage: dg::refinement::Stage::FixChirals};
    bench_group.bench_with_input(
        BenchmarkId::new("serial", refinement_n),
        &refinement_n,
        |b, _| b.iter(|| serial.gradient(black_box(&linear_positions)))
    );

    let parallel = dg::refinement::ParallelRefinement {bounds: refinement_bounds.clone(), stage: dg::refinement::Stage::FixChirals};
    bench_group.bench_with_input(
        BenchmarkId::new("parallel", refinement_n),
        &refinement_n,
        |b, _| b.iter(|| parallel.gradient(black_box(&linear_positions)))
    );

    let gpu = dg::gpu::GpuRefinement::new(refinement_bounds);
    bench_group.bench_with_input(
        BenchmarkId::new("gpu", refinement_n),
        &refinement_n,
        |b, _| b.iter(|| gpu.gradient(black_box(&linear_positions)))
    );
}

criterion_group!(benches, refinement);
criterion_main!(benches);
