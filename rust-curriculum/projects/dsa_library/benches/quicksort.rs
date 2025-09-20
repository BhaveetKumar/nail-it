use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use dsa_library::quicksort;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn bench_quicksort(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort");
    for &n in &[100usize, 1_000, 10_000] {
        group.bench_function(format!("quicksort_{}", n), |b| {
            b.iter_batched(
                || {
                    let mut rng = StdRng::seed_from_u64(42);
                    (0..n).map(|_| rng.gen::<i32>()).collect::<Vec<_>>()
                },
                |mut v| quicksort(&mut v[..]),
                BatchSize::SmallInput,
            )
        });
        group.bench_function(format!("std_sort_{}", n), |b| {
            b.iter_batched(
                || {
                    let mut rng = StdRng::seed_from_u64(42);
                    (0..n).map(|_| rng.gen::<i32>()).collect::<Vec<_>>()
                },
                |mut v| v.sort(),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_quicksort);
criterion_main!(benches);