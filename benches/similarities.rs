use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use molassembler::*;
use strong::matrix::StrongPoints;
use strong::bijection::Bijection;

use itertools::Itertools;

fn similarities(c: &mut Criterion) {
    let size_limit = { if cfg!(debug_assertions) { 7 } else { 9 } };

    let mut bench_group = c.benchmark_group("similarities");
    for shape in shapes::SHAPES.iter().unique_by(|s| s.size()) {
        let size = shape.size();
        if size < 5 || size > size_limit {
            // Up to and including size == 5 the algorithms are identical
            continue;
        }

        let shape_coordinates = shape.coordinates.clone().insert_column(size, 0.0);
        let rotation = quaternions::random_rotation().to_rotation_matrix();
        let shape_rotated: StrongPoints<shapes::Vertex> = StrongPoints::new(shapes::similarity::unit_sphere_normalize(rotation * shape_coordinates));
        let bijection: Bijection<shapes::Vertex, shapes::Column> = {
            let mut p = permutation::Permutation::random(shape.size());
            p.sigma.push(p.set_size() as u8);
            strong::bijection::Bijection::new(p)
        };
        let cloud = shape_rotated.apply_bijection(&bijection);

        bench_group.throughput(Throughput::Elements(size as u64));
        bench_group.bench_with_input(
            BenchmarkId::new("reference", size as u64),
            &(size as u64),
            |b, _| b.iter(|| shapes::similarity::polyhedron_reference(black_box(&cloud.matrix), black_box(shape.name)))
        );
        bench_group.bench_with_input(
            BenchmarkId::new("shortcut", size as u64),
            &(size as u64),
            |b, _| b.iter(|| shapes::similarity::polyhedron(black_box(&cloud.matrix), black_box(shape.name)))
        );
    }

}

criterion_group!(benches, similarities);
criterion_main!(benches);
