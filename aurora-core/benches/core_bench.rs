use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn format_bytes_bench(c: &mut Criterion) {
    c.bench_function("format_bytes", |b| {
        b.iter(|| aurora_core::format_bytes(black_box(16 * 1024 * 1024 * 1024)))
    });
}

criterion_group!(benches, format_bytes_bench);
criterion_main!(benches);
