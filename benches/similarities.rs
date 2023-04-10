use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use molassembler::*;
use strong::matrix::StrongPoints;
use strong::bijection::Bijection;

use itertools::Itertools;

// TODO
// - Better line colors

fn similarities(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("similarities");
    let plot_config = criterion::PlotConfiguration::default()
        .summary_scale(criterion::AxisScale::Logarithmic);
    bench_group.plot_config(plot_config);

    let mut shapes = Vec::new();
    for (size, shape_group) in &shapes::SHAPES.iter().group_by(|s| s.size()) {
        if size < 4 {
            continue;
        }

        let most_symmetric = shape_group
            .max_by_key(|s| s.generate_rotations().len())
            .unwrap();

        shapes.push(*most_symmetric);
    }

    for shape in shapes {
        let size = shape.size();

        if size < 4 {
            continue;
        }

        let shape_coordinates = shape.coordinates.clone().insert_column(size, 0.0);
        let rotation = quaternions::random_rotation().to_rotation_matrix();
        let shape_rotated: StrongPoints<shapes::Vertex> = StrongPoints::new(shapes::similarity::unit_sphere_normalize(rotation * shape_coordinates));
        let bijection: Bijection<shapes::Vertex, shapes::Column> = {
            let mut p = permutation::Permutation::random(shape.size());
            p.sigma.push(p.set_size());
            strong::bijection::Bijection::new(p)
        };
        let cloud = shape_rotated.apply_bijection(&bijection);

        bench_group.throughput(Throughput::Elements(size as u64));

        if size <= 7 {
            bench_group.bench_with_input(
                BenchmarkId::new("reference", size as u64),
                &(size as u64),
                |b, _| b.iter(|| shapes::similarity::polyhedron_reference_base::<false>(black_box(&cloud.matrix), black_box(shape.name)))
            );
        }

        if size <= 8 {
            bench_group.bench_with_input(
                BenchmarkId::new("skip", size as u64),
                &(size as u64),
                |b, _| b.iter(|| shapes::similarity::polyhedron_reference(black_box(&cloud.matrix), black_box(shape.name)))
            );
        }

        if size > 5 && size <= 10 {
            bench_group.bench_with_input(
                BenchmarkId::new("prematch", size as u64),
                &(size as u64),
                |b, _| b.iter(|| shapes::similarity::polyhedron_base::<5, false, false>(black_box(&cloud.matrix), black_box(shape.name)))
            );
        }

        if size > 5 {
            bench_group.bench_with_input(
                BenchmarkId::new("prematch, skip", size as u64),
                &(size as u64),
                |b, _| b.iter(|| shapes::similarity::polyhedron_base::<5, true, false>(black_box(&cloud.matrix), black_box(shape.name)))
            );
        }

        if size > 7 {
            bench_group.bench_with_input(
                BenchmarkId::new("prematch, skip, jv", size as u64),
                &(size as u64),
                |b, _| b.iter(|| shapes::similarity::polyhedron(black_box(&cloud.matrix), black_box(shape.name)))
            );
        }
    }
}

criterion_group!(benches, similarities);
criterion_main!(benches);
