use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use molassembler::*;

fn linear_assignment(c: &mut Criterion) {
    let mut bench_group = c.benchmark_group("linear assignment");
    let plot_config = criterion::PlotConfiguration::default()
        .summary_scale(criterion::AxisScale::Logarithmic);
    bench_group.plot_config(plot_config);

    for i in 1..=6 {
        let cost_matrix = nalgebra::DMatrix::new_random(i, i);
        let cost_fn = |i, j| cost_matrix[(i, j)];

        bench_group.throughput(Throughput::Elements(i as u64));
        bench_group.bench_with_input(
            BenchmarkId::new("brute force", i as u64),
            &(i as u64),
            |b, _| b.iter(|| shapes::similarity::linear_assignment::brute_force(black_box(i), black_box(&cost_fn)))
        );
        bench_group.bench_with_input(
            BenchmarkId::new("jonker volgenant", i as u64),
            &(i as u64),
            |b, _| b.iter(|| shapes::similarity::linear_assignment::jonker_volgenant(black_box(i), black_box(&cost_fn)))
        );
    }

}

criterion_group!(benches, linear_assignment);
criterion_main!(benches);
