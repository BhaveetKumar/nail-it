use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use dsa_library::dijkstra;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::HashMap;

fn random_graph(n: usize, m: usize, seed: u64) -> HashMap<usize, Vec<(usize, i64)>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut g: HashMap<usize, Vec<(usize, i64)>> = HashMap::new();
    for _ in 0..m {
        let u = rng.gen_range(0..n);
        let v = rng.gen_range(0..n);
        if u == v { continue; }
        let w = rng.gen_range(1..10);
        g.entry(u).or_default().push((v, w));
    }
    g
}

fn bench_dijkstra(c: &mut Criterion) {
    let mut group = c.benchmark_group("dijkstra");
    for &(n, m) in &[(100usize, 500usize), (1000, 5000)] {
        group.bench_function(format!("dijkstra_n{}_m{}", n, m), |b| {
            b.iter_batched(
                || random_graph(n, m, 42),
                |g| { let _ = dijkstra(&g, 0); },
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dijkstra);
criterion_main!(benches);