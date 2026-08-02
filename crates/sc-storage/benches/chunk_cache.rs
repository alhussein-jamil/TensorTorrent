//! Criterion benches for chunk-cache hit overhead.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sc_storage::ChunkCache;

fn bench_cache_hits(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_cache_hit");
    for entries in [16, 256, 4096] {
        let cache = ChunkCache::new(u64::MAX);
        for index in 0..entries {
            cache.insert(format!("chunk-{index}"), vec![0]);
        }
        let key = "chunk-0";
        group.bench_with_input(BenchmarkId::from_parameter(entries), &entries, |b, _| {
            b.iter(|| {
                let data = cache.get(black_box(key)).unwrap();
                black_box(data);
                cache.release(key);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cache_hits);
criterion_main!(benches);
