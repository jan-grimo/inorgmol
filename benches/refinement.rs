use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use inorgmol::dg::refinement::RefinementErrorFunction;
use inorgmol::*;
extern crate nalgebra as na;

// This benchmark setup essentially only benchmarks the distance gradient
// calculation, which is assumed to be the bottleneck in large molecule
// refinements.

#[cfg(feature = "gpu")]
fn gpu_bench<T>(
    bench_group: &mut criterion::BenchmarkGroup<T>,
    refinement_n: u64,
    refinement_bounds: dg::refinement::Bounds<f64>,
    linear_positions: &na::DVector<f64>
) 
    where T: criterion::measurement::Measurement 
{
    let gpu = dg::refinement::gpu::Gpu::new(refinement_bounds);
    bench_group.bench_with_input(
        BenchmarkId::new("gpu", refinement_n),
        &refinement_n,
        |b, _| b.iter(|| gpu.gradient(black_box(linear_positions)))
    );
}

#[cfg(not(feature = "gpu"))]
fn gpu_bench<T>(
    _bench_group: &mut criterion::BenchmarkGroup<T>,
    _refinement_n: u64,
    _refinement_bounds: dg::refinement::Bounds<f64>,
    _linear_positions: &na::DVector<f64>
) where T: criterion::measurement::Measurement {}

fn refinement(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("refinement");
    let plot_config = criterion::PlotConfiguration::default()
        .summary_scale(criterion::AxisScale::Logarithmic);
    bench_group.plot_config(plot_config);

    let max_shape_size = &shapes::SHAPES.iter().map(|s| s.num_vertices()).max().unwrap();
    for shape_size in 8..=*max_shape_size {
        let shape = &shapes::SHAPES.iter().find(|s| s.num_vertices() == shape_size).unwrap();
        let bounds = dg::modeling::solitary_shape::shape_into_bounds(shape);
        let tetrahedra = shape.find_tetrahedra();
        let distances = dg::DistanceMatrix::try_from_distance_bounds(bounds.clone(), dg::MetrizationPartiality::Complete).expect("Successful metrization");
        let coords = dg::MetricMatrix::from(distances).embed();
        let n = coords.len();
        let linear_positions = coords.reshape_generic(na::Dyn(n), na::Const::<1>);
        let chirals: Vec<dg::refinement::Chiral<f64>> = tetrahedra.iter()
            .map(|&tetr| dg::modeling::solitary_shape::chiral_from_tetrahedron(tetr, shape, 0.1))
            .collect();

        let refinement_n = (shape.num_vertices() + 1) as u64;
        bench_group.throughput(Throughput::Elements(refinement_n));

        let refinement_bounds = dg::refinement::Bounds::new(bounds, chirals);

        let serial = dg::refinement::Serial::new(refinement_bounds.clone());
        bench_group.bench_with_input(
            BenchmarkId::new("serial", refinement_n),
            &refinement_n,
            |b, _| b.iter(|| serial.gradient(black_box(&linear_positions)))
        );

        let parallel = dg::refinement::Parallel::new(refinement_bounds.clone());
        bench_group.bench_with_input(
            BenchmarkId::new("parallel", refinement_n),
            &refinement_n,
            |b, _| b.iter(|| parallel.gradient(black_box(&linear_positions)))
        );

        #[allow(clippy::unnecessary_mut_passed)]
        gpu_bench(&mut bench_group, refinement_n, refinement_bounds, &linear_positions);
    }
}

criterion_group!(benches, refinement);
criterion_main!(benches);
