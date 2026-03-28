use aurora_cpu::ScalarOps;
use aurora_cpu::VectorOps;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn add_kernel_bench(c: &mut Criterion) {
    let ops = ScalarOps;
    let a = vec![1.0f32; 4096];
    let b = vec![2.0f32; 4096];
    let mut out = vec![0.0f32; 4096];

    c.bench_function("scalar_add_f32", |bench| {
        bench.iter(|| ops.add_f32(black_box(&a), black_box(&b), black_box(&mut out)))
    });
}

criterion_group!(benches, add_kernel_bench);
criterion_main!(benches);
