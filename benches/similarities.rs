use criterion::{black_box, criterion_group, criterion_main, Criterion};
use molassembler::*;
use strong::matrix::StrongPoints;
use strong::bijection::Bijection;

fn criterion_benchmark(c: &mut Criterion) {
    let shape = &shapes::OCTAHEDRON;
    let shape_coordinates = shape.coordinates.clone().insert_column(shape.size(), 0.0);
    let rotation = quaternions::random_rotation().to_rotation_matrix();
    let shape_rotated: StrongPoints<shapes::Vertex> = StrongPoints::new(shapes::similarity::unit_sphere_normalize(rotation * shape_coordinates));
    let bijection: Bijection<shapes::Vertex, shapes::Column> = {
        let mut p = permutation::Permutation::random(shape.size());
        p.sigma.push(p.set_size() as u8);
        strong::bijection::Bijection::new(p)
    };
    let cloud = shape_rotated.apply_bijection(&bijection);

    c.bench_function("reference", |b| b.iter(|| shapes::similarity::polyhedron_reference(black_box(&cloud.matrix), black_box(shape.name))));
    c.bench_function("shortcut", |b| b.iter(|| shapes::similarity::polyhedron(black_box(&cloud.matrix), black_box(shape.name))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
