//! Correctness and lifetime tests for the TuckCore Metal path.
//!
//! GPU tests skip with a loud note when no Metal device is present and fail
//! instead when `TUCKMILL_REQUIRE_METAL` is set (the repository gate on
//! owned hardware). Comparisons against the CPU scalar reference are
//! tolerance-registered: GPU reduction order differs from the CPU sequence,
//! so exact bit equality is not claimed anywhere.

use super::*;

/// Skip helper: `None` means "no GPU"; with TUCKMILL_REQUIRE_METAL set that
/// is a test failure, never a silent pass.
fn metal_context() -> Option<MetalContext> {
    let probe = probe();
    if !probe.available {
        assert!(
            std::env::var("TUCKMILL_REQUIRE_METAL").is_err(),
            "TUCKMILL_REQUIRE_METAL is set but no Metal device is available ({})",
            probe.info
        );
        eprintln!("skipping: no Metal device ({})", probe.info);
        return None;
    }
    Some(MetalContext::open().expect("metal context"))
}

// ------------------------------------------------------------ references --

/// Deterministic xorshift64* stream (the tests never use OS randomness).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// IEEE binary16 round-to-nearest-even, the semantics the Metal `half`
/// conversion is required to match.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if ((bits >> 23) & 0xff) == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = mantissa | 0x80_0000;
        let shift = (14 - exp) as u32;
        let half = 1u32 << (shift - 1);
        let mut sub = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        if rem > half || (rem == half && (sub & 1) != 0) {
            sub += 1;
        }
        return sign | sub;
    }
    let mut half = (((exp as u32) << 10) | (mantissa >> 13)) as u16;
    let rem = mantissa & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) != 0) {
        half += 1;
    }
    sign | half
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if mantissa == 0 {
            sign
        } else {
            // Subnormal half: value = mantissa * 2^-24; normalize into f32.
            let p = 31 - mantissa.leading_zeros(); // floor(log2(mantissa))
            let m = (mantissa << (23 - p)) & 0x7f_ffff;
            sign | ((p + 103) << 23) | m
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exp + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(out)
}

/// CPU scalar reference for the packed Q8_0 gemv (sequential f32
/// accumulation, exact same dequantization: d * q).
fn cpu_q8_0_gemv(w: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let blocks = cols / 32;
    let mut y = vec![0f32; rows];
    for (r, slot) in y.iter_mut().enumerate() {
        let mut acc = 0f32;
        for b in 0..blocks {
            let off = (r * blocks + b) * 34;
            let d = f16_bits_to_f32(u16::from_le_bytes([w[off], w[off + 1]]));
            for (l, xv) in x[b * 32..(b + 1) * 32].iter().enumerate() {
                let q = w[off + 2 + l] as i8 as f32;
                acc += d * q * xv;
            }
        }
        *slot = acc;
    }
    y
}

/// Q8_0 weights with boundary scales and quantized extremes.
fn boundary_weights(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks = cols / 32;
    let mut rng = Rng(seed);
    let scales = [
        1.0f32,
        -0.5,
        2f32.powi(-12),
        65504.0,
        -65504.0,
        5.96e-8,
        0.0,
        -0.0,
        448.0,
        6e-5,
    ];
    let mut bytes = Vec::with_capacity(rows * blocks * 34);
    for _ in 0..rows * blocks {
        let scale = scales[(rng.next() % scales.len() as u64) as usize];
        bytes.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        for l in 0..32 {
            let q: i8 = match (rng.next() % 8, l) {
                (0, _) => -128,
                (1, _) => 127,
                (2, _) => 0,
                _ => (rng.next() % 256) as u8 as i8,
            };
            bytes.push(q as u8);
        }
    }
    bytes
}

fn boundary_input(cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..cols)
        .map(|i| match i % 7 {
            0 => 0.0,
            1 => -1.0,
            2 => 1e3,
            3 => -1e3,
            _ => ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0,
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ---------------------------------------------------------------- tests ---

#[test]
fn probe_reports_device_presence() {
    let probe = probe();
    assert!(!probe.info.is_empty());
    if !probe.available {
        assert!(probe.info.contains("no Metal"));
    } else {
        // The figure is Metal's recommended work-set size, not the
        // machine's total unified memory; the label must say so.
        assert!(
            probe.info.contains("recommended GPU working set"),
            "info must label the working-set figure honestly: {}",
            probe.info
        );
    }
    eprintln!(
        "metal probe: available={} info={}",
        probe.available, probe.info
    );
}

#[test]
fn context_creation_reports_bad_msl_with_a_typed_error() {
    if !probe().available {
        if std::env::var("TUCKMILL_REQUIRE_METAL").is_ok() {
            panic!("TUCKMILL_REQUIRE_METAL is set but no Metal device is available");
        }
        eprintln!("skipping: no Metal device");
        return;
    }
    let err = match MetalContext::open_with_source("kernel void broken( [[ buffer(0) ]] float x {")
    {
        Ok(_) => panic!("malformed MSL must fail context creation"),
        Err(e) => e,
    };
    let text = format!("{err}");
    assert!(
        text.contains("MSL compilation failed") || text.contains("compilation"),
        "the error must name the compilation failure: {text}"
    );
}

#[test]
fn f32_utility_ops_match_the_scalar_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let mut rng = Rng(7);
    for n in [1usize, 3, 32, 33, 4096, 100_003] {
        let x: Vec<f32> = (0..n)
            .map(|_| ((rng.next() % 200_001) as f32 - 100_000.0) / 1000.0)
            .collect();
        let a: Vec<f32> = (0..n)
            .map(|_| ((rng.next() % 200_001) as f32 - 100_000.0) / 1000.0)
            .collect();
        let factor = -3.25f32;
        let alpha = 0.125f32;

        let xb = ctx.buffer(n * 4, true).unwrap();
        let ab = ctx.buffer(n * 4, true).unwrap();
        let yb = ctx.buffer(n * 4, true).unwrap();
        xb.upload(0, &f32_bytes(&x)).unwrap();
        ab.upload(0, &f32_bytes(&a)).unwrap();

        ctx.f32_scale(&xb, &yb, factor, n as u32).unwrap();
        ctx.f32_axpy(&ab, &yb, alpha, n as u32).unwrap();
        ctx.fence().unwrap();

        let mut out = vec![0u8; n * 4];
        yb.download(0, &mut out).unwrap();
        for (i, chunk) in out.chunks_exact(4).enumerate() {
            let got = f32::from_le_bytes(chunk.try_into().unwrap());
            let want = x[i] * factor + alpha * a[i];
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                "n={n} i={i}: got {got}, want {want}"
            );
        }
    }
}

#[test]
fn f16_conversions_match_ieee_round_to_nearest_even() {
    let Some(ctx) = metal_context() else {
        return;
    };
    // Values chosen across representable ranges, subnormals, and overflow.
    let values: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        65504.0,
        65520.0,
        100_000.0,
        -100_000.0,
        5.960_464_5e-8, // min positive subnormal
        2.980_232_2e-8, // half of it: rounds to zero (ties to even)
        1.0e-7,
        -6.0e-8,
        44.856,
        std::f32::consts::PI,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let n = values.len();
    let xb = ctx.buffer(n * 4, true).unwrap();
    let hb = ctx.buffer(n * 2, true).unwrap();
    let back = ctx.buffer(n * 4, true).unwrap();
    xb.upload(0, &f32_bytes(&values)).unwrap();
    ctx.cast_f32_to_f16(&xb, &hb, n as u32).unwrap();
    ctx.fence().unwrap();

    let mut half_bytes = vec![0u8; n * 2];
    hb.download(0, &mut half_bytes).unwrap();
    for (i, chunk) in half_bytes.chunks_exact(2).enumerate() {
        let got = u16::from_le_bytes(chunk.try_into().unwrap());
        let want = f32_to_f16_bits(values[i]);
        assert_eq!(got, want, "f32->f16 mismatch for {}", values[i]);
    }

    // And back: f16 -> f32 must reproduce the reference decode exactly.
    ctx.cast_f16_to_f32(&hb, &back, n as u32).unwrap();
    ctx.fence().unwrap();
    let mut out = vec![0u8; n * 4];
    back.download(0, &mut out).unwrap();
    for (i, chunk) in out.chunks_exact(4).enumerate() {
        let got = f32::from_le_bytes(chunk.try_into().unwrap());
        let want = f16_bits_to_f32(f32_to_f16_bits(values[i]));
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "f16->f32 mismatch for input {}",
            values[i]
        );
    }
}

#[test]
fn q8_0_gemv_minimal_ones_case() {
    // One row, one block, all q = 1, d = 1.0, x = 2.0: y[0] must be 64.
    let Some(ctx) = metal_context() else {
        return;
    };
    let mut w = Vec::new();
    w.extend_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
    w.extend_from_slice(&[1u8; 32]);
    let x = vec![2.0f32; 32];
    let wb = ctx.buffer(w.len(), true).unwrap();
    let xb = ctx.buffer(32 * 4, true).unwrap();
    let yb = ctx.buffer(4, true).unwrap();
    wb.upload(0, &w).unwrap();
    xb.upload(0, &f32_bytes(&x)).unwrap();
    ctx.q8_0_gemv(&wb, &xb, &yb, 1, 32).unwrap();
    ctx.fence().unwrap();
    let mut out = [0u8; 4];
    yb.download(0, &mut out).unwrap();
    let got = f32::from_le_bytes(out);
    assert_eq!(got, 64.0, "gmm ones case");
}

#[test]
fn q8_0_gemv_matches_the_scalar_reference_across_real_shapes() {
    let Some(ctx) = metal_context() else {
        return;
    };
    // Real Qwen3-0.6B shapes plus small tail/Odd row counts; cols always a
    // multiple of 32 by the format, rows exercise 1, primes and large grids.
    let shapes: [(usize, usize, u64); 7] = [
        (1, 32, 1),
        (3, 32, 2),
        (17, 64, 3),
        (1024, 1024, 4),    // o_proj / embed-like
        (2048, 1024, 5),    // q_proj
        (3072, 1024, 6),    // ffn gate/up/down
        (151_936, 1024, 7), // token embedding / lm head
    ];
    for (rows, cols, seed) in shapes {
        let w = boundary_weights(rows, cols, seed);
        let x = boundary_input(cols, seed ^ 0x9e37);
        let wb = ctx.buffer(w.len(), true).unwrap();
        let xb = ctx.buffer(cols * 4, true).unwrap();
        let yb = ctx.buffer(rows * 4, true).unwrap();
        wb.upload(0, &w).unwrap();
        xb.upload(0, &f32_bytes(&x)).unwrap();
        ctx.q8_0_gemv(&wb, &xb, &yb, rows as u32, cols as u32)
            .unwrap();
        ctx.fence().unwrap();

        let mut out = vec![0u8; rows * 4];
        yb.download(0, &mut out).unwrap();
        let gpu: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let cpu = cpu_q8_0_gemv(&w, &x, rows, cols);
        let scale = cpu.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
        let check_rows: Vec<usize> = if rows <= 32 {
            (0..rows).collect()
        } else {
            vec![0, 1, 2, rows / 2, rows - 3, rows - 2, rows - 1]
        };
        for &r in &check_rows {
            let diff = (gpu[r] - cpu[r]).abs();
            assert!(
                diff <= 1e-4 * scale,
                "shape ({rows},{cols}) row {r}: gpu {}, cpu {}, diff {diff} over scale {scale}",
                gpu[r],
                cpu[r]
            );
        }
    }
}

#[test]
fn q8_0_gemv_rejects_malformed_arguments() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let wb = ctx.buffer(2 * 34, true).unwrap(); // one row of 32 cols
    let xb = ctx.buffer(32 * 4, true).unwrap();
    let yb = ctx.buffer(4, true).unwrap();
    // cols not a multiple of 32.
    assert!(ctx.q8_0_gemv(&wb, &xb, &yb, 2, 30).is_err());
    // zero rows.
    assert!(ctx.q8_0_gemv(&wb, &xb, &yb, 0, 32).is_err());
    // Output buffer too small for two rows.
    let err = ctx.q8_0_gemv(&wb, &xb, &yb, 2, 32).unwrap_err();
    assert_eq!(err.code(), ErrorCode::InvalidRequest);
    // Weight buffer too small.
    let big_w = ctx.buffer(34, true).unwrap();
    let big_y = ctx.buffer(8, true).unwrap();
    assert!(ctx.q8_0_gemv(&big_w, &xb, &big_y, 2, 32).is_err());
    // Zero-length allocation is refused outright.
    assert!(ctx.buffer(0, true).is_err());
}

#[test]
fn buffers_are_context_scoped_and_accounted() {
    let Some(ctx) = metal_context() else {
        return;
    };
    assert_eq!(ctx.allocated_bytes(), 0);
    let a = ctx.buffer(1024, true).unwrap();
    let b = ctx.buffer(2048, false).unwrap();
    assert_eq!(ctx.allocated_bytes(), 3072);
    assert_eq!(a.len(), 1024);
    assert_eq!(b.len(), 2048);

    // A buffer from another context is refused by every operation.
    let other = MetalContext::open().unwrap();
    let foreign = other.buffer(64, true).unwrap();
    let err = ctx.f32_scale(&foreign, &a, 1.0, 4).unwrap_err();
    assert!(format!("{err}").contains("different context"), "{err}");
    assert!(ctx.q8_0_gemv(&foreign, &foreign, &a, 1, 32).is_err());

    // Private buffers have no host copy.
    let mut out = [0u8; 8];
    assert!(b.download(0, &mut out).is_err());

    drop(a);
    drop(b);
    assert_eq!(ctx.allocated_bytes(), 0);
    assert_eq!(other.allocated_bytes(), 64);
    drop(foreign);
    assert_eq!(other.allocated_bytes(), 0);
}

#[test]
fn dispatch_completion_accounting_is_exact() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let x = ctx.buffer(64, true).unwrap();
    let y = ctx.buffer(64, true).unwrap();
    x.upload(0, &f32_bytes(&[1.0; 16])).unwrap();
    for _ in 0..5 {
        ctx.f32_scale(&x, &y, 2.0, 16).unwrap();
    }
    assert_eq!(ctx.committed_dispatches(), 5);
    ctx.fence().unwrap();
    assert_eq!(ctx.completed_dispatches(), 5);
    assert!(ctx.last_gpu_ms() >= 0.0);
    let mut out = vec![0u8; 64];
    y.download(0, &mut out).unwrap();
    for chunk in out.chunks_exact(4) {
        assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), 2.0);
    }
}

#[test]
fn upload_download_respect_bounds_and_offsets() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let buf = ctx.buffer(16, true).unwrap();
    buf.upload(8, &[9u8; 8]).unwrap();
    let mut out = [0u8; 16];
    buf.download(0, &mut out).unwrap();
    assert_eq!(&out[8..], &[9u8; 8]);
    assert_eq!(&out[..8], &[0u8; 8]);
    assert!(buf.upload(12, &[0u8; 8]).is_err(), "range past the end");
    assert!(buf.download(16, &mut [0u8; 1]).is_err());
}

#[test]
fn copy_contract_synchronizes_with_in_flight_compute() {
    // A download immediately after asynchronous compute must observe the
    // finished result (the copy contract fences internally); an upload must
    // not race in-flight work either.
    let Some(ctx) = metal_context() else {
        return;
    };
    let x = ctx.buffer(64, true).unwrap();
    let y = ctx.buffer(64, true).unwrap();
    x.upload(0, &f32_bytes(&[3.0; 16])).unwrap();
    ctx.f32_scale(&x, &y, 4.0, 16).unwrap();
    // No explicit fence here: the copy contract orders it.
    let mut out = vec![0u8; 64];
    y.download(0, &mut out).unwrap();
    for chunk in out.chunks_exact(4) {
        assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), 12.0);
    }
    // Upload-while-pending: overwrite the input, run again, read back.
    x.upload(0, &f32_bytes(&[5.0; 16])).unwrap();
    ctx.f32_scale(&x, &y, 2.0, 16).unwrap();
    y.download(0, &mut out).unwrap();
    for chunk in out.chunks_exact(4) {
        assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), 10.0);
    }
}

#[test]
fn dropping_buffers_before_fence_is_safe() {
    // In-flight leases must keep device resources alive: the handles are
    // freed here before any fence, and the GPU work must still complete
    // correctly with the allocation accounting returning to zero.
    let Some(ctx) = metal_context() else {
        return;
    };
    let x = ctx.buffer(64, true).unwrap();
    let y = ctx.buffer(64, true).unwrap();
    let out_keep = ctx.buffer(64, true).unwrap();
    x.upload(0, &f32_bytes(&[7.0; 16])).unwrap();
    ctx.f32_scale(&x, &out_keep, 3.0, 16).unwrap();
    drop(x); // GPU work on `x` is still pending
    ctx.fence().unwrap();
    let mut out = vec![0u8; 64];
    out_keep.download(0, &mut out).unwrap();
    for chunk in out.chunks_exact(4) {
        assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), 21.0);
    }
    drop(y);
    drop(out_keep);
    assert_eq!(ctx.allocated_bytes(), 0);
}

#[test]
fn grouped_encoding_matches_individual_dispatch_and_aborts_cleanly() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let a = ctx.buffer(64, true).unwrap();
    let b = ctx.buffer(64, true).unwrap();
    a.upload(0, &f32_bytes(&[2.0; 16])).unwrap();

    // Grouped: two operations, one commit.
    let committed_before = ctx.committed_dispatches();
    ctx.begin_group().unwrap();
    ctx.f32_scale(&a, &b, 3.0, 16).unwrap();
    ctx.f32_axpy(&a, &b, 0.5, 16).unwrap();
    ctx.end_group().unwrap();
    assert_eq!(ctx.committed_dispatches(), committed_before + 1);
    ctx.fence().unwrap();
    let mut grouped = vec![0u8; 64];
    b.download(0, &mut grouped).unwrap();
    for chunk in grouped.chunks_exact(4) {
        assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), 7.0);
    }

    // Individual: same math, two commits.
    ctx.f32_scale(&a, &b, 3.0, 16).unwrap();
    ctx.f32_axpy(&a, &b, 0.5, 16).unwrap();
    ctx.fence().unwrap();
    let mut individual = vec![0u8; 64];
    b.download(0, &mut individual).unwrap();
    assert_eq!(grouped, individual);

    // Abort: the discarded group never commits and never writes.
    b.upload(0, &f32_bytes(&[0.0; 16])).unwrap();
    let committed_before = ctx.committed_dispatches();
    ctx.begin_group().unwrap();
    ctx.f32_scale(&a, &b, 99.0, 16).unwrap();
    ctx.abort_group().unwrap();
    assert_eq!(ctx.committed_dispatches(), committed_before);
    let mut aborted = vec![0u8; 64];
    b.download(0, &mut aborted).unwrap();
    assert!(aborted.iter().all(|byte| *byte == 0));

    // A fence with an open group is refused rather than guessed.
    ctx.begin_group().unwrap();
    assert!(ctx.fence().is_err());
    ctx.abort_group().unwrap();
}

/// CPU references for the transformer ops (mirroring tuckcore's scalar
/// executor semantics: f32, sequential accumulation).
fn cpu_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0f32 / (ss / x.len() as f32 + eps).sqrt();
    x.iter().zip(w).map(|(v, wv)| v * inv * wv).collect()
}

fn cpu_rope(row: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let half = row.len() / 2;
    for i in 0..half {
        let theta = pos as f32 * inv_freq[i];
        let (s, c) = theta.sin_cos();
        let (x0, x1) = (row[i], row[i + half]);
        row[i] = x0 * c - x1 * s;
        row[i + half] = x0 * s + x1 * c;
    }
}

fn rope_freqs(head_dim: usize, base: f64) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| (1.0 / base.powf(2.0 * i as f64 / head_dim as f64)) as f32)
        .collect()
}

#[test]
fn rmsnorm_matches_the_scalar_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let mut rng = Rng(11);
    for (rows, cols) in [(1usize, 1024usize), (16, 128), (8, 128), (1, 7)] {
        let x: Vec<f32> = (0..rows * cols)
            .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
            .collect();
        let w: Vec<f32> = (0..cols)
            .map(|_| 0.5 + (rng.next() % 1000) as f32 / 1000.0)
            .collect();
        let eps = 1e-6f32;
        let xb = ctx.buffer(rows * cols * 4, true).unwrap();
        let wb = ctx.buffer(cols * 4, true).unwrap();
        let yb = ctx.buffer(rows * cols * 4, true).unwrap();
        xb.upload(0, &f32_bytes(&x)).unwrap();
        wb.upload(0, &f32_bytes(&w)).unwrap();
        ctx.rmsnorm(&xb, &wb, &yb, rows as u32, cols as u32, eps)
            .unwrap();
        ctx.fence().unwrap();
        let mut out = vec![0u8; rows * cols * 4];
        yb.download(0, &mut out).unwrap();
        let got: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for r in 0..rows {
            let want = cpu_rmsnorm(&x[r * cols..(r + 1) * cols], &w, eps);
            for c in 0..cols {
                let (g, t) = (got[r * cols + c], want[c]);
                assert!(
                    (g - t).abs() <= 1e-5 * t.abs().max(1.0),
                    "({rows},{cols}) [{r},{c}]: got {g}, want {t}"
                );
            }
        }
    }
}

#[test]
fn rope_matches_the_scalar_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let (heads, kv_heads, head_dim) = (16usize, 8usize, 128usize);
    let inv = rope_freqs(head_dim, 1e6);
    let mut rng = Rng(13);
    let q: Vec<f32> = (0..heads * head_dim)
        .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
        .collect();
    let k: Vec<f32> = (0..kv_heads * head_dim)
        .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
        .collect();
    for pos in [0usize, 1, 37] {
        let qb = ctx.buffer(heads * head_dim * 4, true).unwrap();
        let kb = ctx.buffer(kv_heads * head_dim * 4, true).unwrap();
        let fb = ctx.buffer(inv.len() * 4, true).unwrap();
        qb.upload(0, &f32_bytes(&q)).unwrap();
        kb.upload(0, &f32_bytes(&k)).unwrap();
        fb.upload(0, &f32_bytes(&inv)).unwrap();
        ctx.rope(
            &qb,
            &kb,
            &fb,
            pos as u32,
            heads as u32,
            kv_heads as u32,
            head_dim as u32,
        )
        .unwrap();
        ctx.fence().unwrap();
        let mut q_out = vec![0u8; heads * head_dim * 4];
        let mut k_out = vec![0u8; kv_heads * head_dim * 4];
        qb.download(0, &mut q_out).unwrap();
        kb.download(0, &mut k_out).unwrap();
        let gq: Vec<f32> = q_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let gk: Vec<f32> = k_out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // Registered rope policy: sin/cos implementations differ by a few
        // ulps at large angles (pos 37 rad), which shows as ~3e-5 relative
        // deviation (CI measured 2.9e-5); the bound carries ~7x margin. The
        // end-to-end model comparison pins the aggregate effect (2.8e-6
        // relative on logits), so this is a per-op registration, not slack
        // for a hidden mismatch.
        let rope_tol = |t: f32| 2e-4 * t.abs().max(1.0);
        for h in 0..heads {
            let mut want = q[h * head_dim..(h + 1) * head_dim].to_vec();
            cpu_rope(&mut want, pos, &inv);
            for d in 0..head_dim {
                let (g, t) = (gq[h * head_dim + d], want[d]);
                assert!(
                    (g - t).abs() <= rope_tol(t),
                    "q head {h} pos {pos}: {g} vs {t}"
                );
            }
        }
        for h in 0..kv_heads {
            let mut want = k[h * head_dim..(h + 1) * head_dim].to_vec();
            cpu_rope(&mut want, pos, &inv);
            for d in 0..head_dim {
                let (g, t) = (gk[h * head_dim + d], want[d]);
                assert!(
                    (g - t).abs() <= rope_tol(t),
                    "k head {h} pos {pos}: {g} vs {t}"
                );
            }
        }
    }
}

#[test]
fn kv_store_and_attention_match_the_scalar_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let (heads, kv_heads, head_dim, group) = (16usize, 8usize, 128usize, 2usize);
    let cap = 64usize;
    let mut rng = Rng(17);
    let pos = 20usize;
    let seq = pos + 1;

    let mut k_steps: Vec<Vec<f32>> = Vec::new();
    let mut v_steps: Vec<Vec<f32>> = Vec::new();
    for _ in 0..seq {
        k_steps.push(
            (0..kv_heads * head_dim)
                .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
                .collect(),
        );
        v_steps.push(
            (0..kv_heads * head_dim)
                .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
                .collect(),
        );
    }
    let q: Vec<f32> = (0..heads * head_dim)
        .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
        .collect();

    let k_cache = ctx.buffer(kv_heads * cap * head_dim * 4, true).unwrap();
    let v_cache = ctx.buffer(kv_heads * cap * head_dim * 4, true).unwrap();
    let scratch = ctx.buffer(heads * cap * 4, true).unwrap();
    let qb = ctx.buffer(heads * head_dim * 4, true).unwrap();
    let out = ctx.buffer(heads * head_dim * 4, true).unwrap();
    qb.upload(0, &f32_bytes(&q)).unwrap();

    // Stage every step's k/v rows first (copies require no open group),
    // then store all positions with grouped encoding: one command buffer,
    // one completion for the whole sequence of stores.
    let mut k_srcs = Vec::new();
    let mut v_srcs = Vec::new();
    for (ks, vs) in k_steps.iter().zip(v_steps.iter()) {
        let k_src = ctx.buffer(kv_heads * head_dim * 4, true).unwrap();
        let v_src = ctx.buffer(kv_heads * head_dim * 4, true).unwrap();
        k_src.upload(0, &f32_bytes(ks)).unwrap();
        v_src.upload(0, &f32_bytes(vs)).unwrap();
        k_srcs.push(k_src);
        v_srcs.push(v_src);
    }
    ctx.begin_group().unwrap();
    for (j, (k_src, v_src)) in k_srcs.iter().zip(v_srcs.iter()).enumerate() {
        ctx.kv_store(
            k_src,
            v_src,
            &k_cache,
            &v_cache,
            kv_heads as u32,
            j as u32,
            cap as u32,
            head_dim as u32,
            false,
        )
        .unwrap();
    }
    ctx.end_group().unwrap();

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    ctx.attention(
        &qb,
        &k_cache,
        &v_cache,
        &out,
        &scratch,
        heads as u32,
        pos as u32,
        cap as u32,
        head_dim as u32,
        group as u32,
        scale,
    )
    .unwrap();
    ctx.fence().unwrap();

    let mut out_bytes = vec![0u8; heads * head_dim * 4];
    out.download(0, &mut out_bytes).unwrap();
    let got: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // CPU reference: softmax(q·k * scale) over 0..=pos, weighted v, GQA map.
    let mut max_diff = 0f32;
    let mut max_rel = 0f32;
    for h in 0..heads {
        let kvh = h / group;
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = Vec::with_capacity(seq);
        for ks in k_steps.iter().take(seq) {
            let kj = &ks[kvh * head_dim..(kvh + 1) * head_dim];
            let dot: f32 = qh.iter().zip(kj).map(|(a, b)| a * b).sum();
            scores.push(dot * scale);
        }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let denom: f32 = exps.iter().sum();
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for j in 0..seq {
                acc += (exps[j] / denom) * v_steps[j][kvh * head_dim + d];
            }
            let got_v = got[h * head_dim + d];
            let diff = (got_v - acc).abs();
            max_diff = max_diff.max(diff);
            max_rel = max_rel.max(diff / acc.abs().max(1e-3));
        }
    }
    // Registered policy (C2): the 128-term score dot reduces in tree order
    // on the GPU and sequentially in the reference, so scores differ by
    // ~1e-4 after the 1/sqrt(d) scale; softmax turns that into weight
    // deviations of similar relative size, which the weighted-value sum
    // carries into the output. Measured max |diff| is ~1.2e-4 for inputs
    // with |v| up to 10; the registered bound is 1e-3 (≈8x margin).
    eprintln!("attention max |diff| = {max_diff:.3e}, max rel = {max_rel:.3e}");
    assert!(
        max_diff <= 1e-3,
        "attention deviation {max_diff:.3e} exceeds the registered 1e-3 policy"
    );
}

#[test]
fn silu_mul_and_row_dequant_match_the_scalar_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let mut rng = Rng(19);
    for n in [3072usize, 3071, 1] {
        let gate: Vec<f32> = (0..n)
            .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
            .collect();
        let up: Vec<f32> = (0..n)
            .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
            .collect();
        let gb = ctx.buffer(n * 4, true).unwrap();
        let ub = ctx.buffer(n * 4, true).unwrap();
        let ob = ctx.buffer(n * 4, true).unwrap();
        gb.upload(0, &f32_bytes(&gate)).unwrap();
        ub.upload(0, &f32_bytes(&up)).unwrap();
        ctx.silu_mul(&gb, &ub, &ob, n as u32).unwrap();
        ctx.fence().unwrap();
        let mut out = vec![0u8; n * 4];
        ob.download(0, &mut out).unwrap();
        for (i, chunk) in out.chunks_exact(4).enumerate() {
            let g = f32::from_le_bytes(chunk.try_into().unwrap());
            let want = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
            assert!(
                (g - want).abs() <= 1e-5 * want.abs().max(1.0),
                "n={n} i={i}"
            );
        }
    }

    // Q8_0 row dequantization on boundary weights.
    let (rows, cols) = (8usize, 64usize);
    let w = boundary_weights(rows, cols, 23);
    let wb = ctx.buffer(w.len(), true).unwrap();
    let ob = ctx.buffer(cols * 4, true).unwrap();
    wb.upload(0, &w).unwrap();
    for row in [0u32, 5, 7] {
        ctx.q8_0_row(&wb, &ob, row, cols as u32, 0).unwrap();
        ctx.fence().unwrap();
        let mut out = vec![0u8; cols * 4];
        ob.download(0, &mut out).unwrap();
        let blocks = cols / 32;
        for d in 0..cols {
            let b = d / 32;
            let lane = d % 32;
            let off = (row as usize * blocks + b) * 34;
            let scale = f16_bits_to_f32(u16::from_le_bytes([w[off], w[off + 1]]));
            let q = w[off + 2 + lane] as i8 as f32;
            let got = f32::from_le_bytes(out[d * 4..d * 4 + 4].try_into().unwrap());
            assert_eq!(got, scale * q, "row {row} dim {d}");
        }
    }
}

#[test]
fn f16_gemv_matches_the_scalar_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let mut rng = Rng(29);
    for (rows, cols, seed) in [(1usize, 32usize, 1u64), (17, 64, 2), (1024, 1024, 3)] {
        let wf: Vec<f32> = (0..rows * cols)
            .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
            .collect();
        let w: Vec<u8> = wf
            .iter()
            .flat_map(|v| f32_to_f16_bits(*v).to_le_bytes())
            .collect();
        let x = boundary_input(cols, seed);
        let wb = ctx.buffer(w.len(), true).unwrap();
        let xb = ctx.buffer(cols * 4, true).unwrap();
        let yb = ctx.buffer(rows * 4, true).unwrap();
        wb.upload(0, &w).unwrap();
        xb.upload(0, &f32_bytes(&x)).unwrap();
        ctx.f16_gemv(&wb, &xb, &yb, rows as u32, cols as u32)
            .unwrap();
        ctx.fence().unwrap();
        let mut out = vec![0u8; rows * 4];
        yb.download(0, &mut out).unwrap();
        let gpu: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut scale = 1.0f32;
        for r in [0usize, rows / 2, rows - 1] {
            let row = &w[r * cols * 2..(r + 1) * cols * 2];
            let want: f32 = row
                .chunks_exact(2)
                .zip(&x)
                .map(|(c, xv)| f16_bits_to_f32(u16::from_le_bytes(c.try_into().unwrap())) * xv)
                .sum();
            scale = scale.max(want.abs());
            assert!(
                (gpu[r] - want).abs() <= 1e-4 * scale,
                "({rows},{cols}) row {r}: {} vs {want}",
                gpu[r]
            );
        }
    }
}

/// Real artifact operator check (machine-local): the packed gemv must match
/// the scalar reference on real Q8_0 tensor bytes, not just synthetic data.
#[test]
fn q8_0_gemv_matches_on_real_artifact_bytes() {
    let Ok(gguf) = std::env::var("TUCKMILL_REAL_GGUF") else {
        eprintln!("skipping: TUCKMILL_REAL_GGUF not set (real-weight label)");
        return;
    };
    let Some(ctx) = metal_context() else {
        return;
    };
    let path = std::path::PathBuf::from(&gguf);
    let inspection =
        tuckmill_manifest::gguf::inspect(&path, &tuckmill_manifest::gguf::Limits::default())
            .expect("inspect real artifact");
    let tensor = inspection
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .expect("blk.0.attn_q.weight present");
    assert_eq!(tensor.type_id, 8, "expected Q8_0");
    let dims = &tensor.dims; // [in, out]
    let (cols, rows) = (dims[0] as usize, dims[1] as usize);
    assert_eq!(tensor.byte_size as usize, rows * (cols / 32) * 34);

    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(&path).unwrap();
    file.seek(SeekFrom::Start(inspection.data_start + tensor.offset))
        .unwrap();
    let mut w = vec![0u8; tensor.byte_size as usize];
    file.read_exact(&mut w).unwrap();

    let x = boundary_input(cols, 42);
    let wb = ctx.buffer(w.len(), true).unwrap();
    let xb = ctx.buffer(cols * 4, true).unwrap();
    let yb = ctx.buffer(rows * 4, true).unwrap();
    wb.upload(0, &w).unwrap();
    xb.upload(0, &f32_bytes(&x)).unwrap();
    ctx.q8_0_gemv(&wb, &xb, &yb, rows as u32, cols as u32)
        .unwrap();
    ctx.fence().unwrap();

    let mut out = vec![0u8; rows * 4];
    yb.download(0, &mut out).unwrap();
    let gpu: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let cpu = cpu_q8_0_gemv(&w, &x, rows, cols);
    let scale = cpu.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
    for r in [0usize, 1, rows / 2, rows - 2, rows - 1] {
        let diff = (gpu[r] - cpu[r]).abs();
        assert!(
            diff <= 1e-4 * scale,
            "row {r}: gpu {}, cpu {}, diff {diff} over scale {scale}",
            gpu[r],
            cpu[r]
        );
    }
}

#[test]
fn in_flight_bytes_stay_visible_until_completion() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let x = ctx.buffer(4096, true).unwrap();
    let y = ctx.buffer(4096, true).unwrap();
    x.upload(0, &f32_bytes(&[1.0; 1024])).unwrap();
    assert_eq!(ctx.in_flight_bytes(), 0);
    ctx.f32_scale(&x, &y, 2.0, 1024).unwrap();
    // Handles dropped, GPU work still pending: the lease keeps the storage
    // physically alive and visible to accounting.
    drop(x);
    assert_eq!(ctx.allocated_bytes(), 4096, "only y is still handle-live");
    drop(y);
    assert_eq!(
        ctx.allocated_bytes(),
        0,
        "handle bytes are released on drop"
    );
    let leased_before = ctx.in_flight_bytes();
    assert!(
        leased_before >= 8192,
        "dropped-but-leased buffers must remain visible: {leased_before}"
    );
    assert!(ctx.live_gpu_bytes() >= leased_before);
    ctx.fence().unwrap();
    assert_eq!(ctx.in_flight_bytes(), 0, "leases release at completion");

    // Correctness after the fully-lease-only run: same math, kept handles.
    let x2 = ctx.buffer(4096, true).unwrap();
    let y2 = ctx.buffer(4096, true).unwrap();
    x2.upload(0, &f32_bytes(&[3.0; 1024])).unwrap();
    ctx.f32_scale(&x2, &y2, 2.0, 1024).unwrap();
    ctx.fence().unwrap();
    let mut out = vec![0u8; 4096];
    y2.download(0, &mut out).unwrap();
    for chunk in out.chunks_exact(4).take(1024) {
        assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), 6.0);
    }
    drop(x2);
    drop(y2);
    assert_eq!(ctx.allocated_bytes(), 0);
}

#[test]
fn batched_gemm_matches_the_scalar_reference_for_all_rows() {
    let Some(ctx) = metal_context() else {
        return;
    };
    // (rows, cols, all-rows?) — realistic projection shapes; the 151936-row
    // vocabulary projection is spot-checked to keep the reference bounded.
    let shapes: [(usize, usize, bool); 5] = [
        (32, 32, true),
        (64, 1024, true),
        (2048, 1024, true),
        (3072, 1024, true),
        (151_936, 1024, false),
    ];
    for (rows, cols, all_rows) in shapes {
        let w = boundary_weights(rows, cols, 101 + rows as u64);
        for batch in [1usize, 4, 8, 16] {
            let xs: Vec<Vec<f32>> = (0..batch)
                .map(|b| boundary_input(cols, 200 + b as u64))
                .collect();
            let mut flat = Vec::with_capacity(batch * cols);
            for x in &xs {
                flat.extend_from_slice(x);
            }
            let wb = ctx.buffer(w.len(), true).unwrap();
            let xb = ctx.buffer(flat.len() * 4, true).unwrap();
            let yb = ctx.buffer(batch * rows * 4, true).unwrap();
            wb.upload(0, &w).unwrap();
            xb.upload(0, &f32_bytes(&flat)).unwrap();
            ctx.q8_0_gemm(&wb, &xb, &yb, rows as u32, cols as u32, batch as u32, 0, 0)
                .unwrap();
            ctx.fence().unwrap();
            let mut out = vec![0u8; batch * rows * 4];
            yb.download(0, &mut out).unwrap();
            let gpu: Vec<f32> = out
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let check_rows: Vec<usize> = if all_rows {
                (0..rows).collect()
            } else {
                vec![0, 1, rows / 2, rows - 2, rows - 1]
            };
            for (b, x) in xs.iter().enumerate() {
                let cpu = cpu_q8_0_gemv(&w, x, rows, cols);
                let scale = cpu.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
                for &r in &check_rows {
                    let diff = (gpu[b * rows + r] - cpu[r]).abs();
                    assert!(
                        diff <= 1e-4 * scale,
                        "({rows},{cols}) batch {batch} b {b} row {r}: {} vs {}",
                        gpu[b * rows + r],
                        cpu[r]
                    );
                }
            }
        }
    }
}

#[test]
fn batched_gemm_batch_one_is_bit_identical_to_the_single_token_gemv() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let (rows, cols) = (512usize, 1024usize);
    let w = boundary_weights(rows, cols, 7);
    let x = boundary_input(cols, 9);
    let wb = ctx.buffer(w.len(), true).unwrap();
    let xb = ctx.buffer(cols * 4, true).unwrap();
    let y_single = ctx.buffer(rows * 4, true).unwrap();
    let y_batch = ctx.buffer(rows * 4, true).unwrap();
    wb.upload(0, &w).unwrap();
    xb.upload(0, &f32_bytes(&x)).unwrap();
    ctx.q8_0_gemv(&wb, &xb, &y_single, rows as u32, cols as u32)
        .unwrap();
    ctx.q8_0_gemm(&wb, &xb, &y_batch, rows as u32, cols as u32, 1, 0, 0)
        .unwrap();
    ctx.fence().unwrap();
    let mut a = vec![0u8; rows * 4];
    let mut b = vec![0u8; rows * 4];
    y_single.download(0, &mut a).unwrap();
    y_batch.download(0, &mut b).unwrap();
    assert_eq!(
        a, b,
        "batch=1 must be bit-identical to the single-token GEMV"
    );
}

#[test]
fn f16_gemm_matches_the_scalar_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let (rows, cols) = (128usize, 256usize);
    let mut rng = Rng(31);
    let wf: Vec<f32> = (0..rows * cols)
        .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
        .collect();
    let w: Vec<u8> = wf
        .iter()
        .flat_map(|v| f32_to_f16_bits(*v).to_le_bytes())
        .collect();
    let batch = 8usize;
    let xs: Vec<Vec<f32>> = (0..batch)
        .map(|b| boundary_input(cols, 41 + b as u64))
        .collect();
    let mut flat = Vec::new();
    for x in &xs {
        flat.extend_from_slice(x);
    }
    let wb = ctx.buffer(w.len(), true).unwrap();
    let xb = ctx.buffer(flat.len() * 4, true).unwrap();
    let yb = ctx.buffer(batch * rows * 4, true).unwrap();
    wb.upload(0, &w).unwrap();
    xb.upload(0, &f32_bytes(&flat)).unwrap();
    ctx.f16_gemm(&wb, &xb, &yb, rows as u32, cols as u32, batch as u32, 0, 0)
        .unwrap();
    ctx.fence().unwrap();
    let mut out = vec![0u8; batch * rows * 4];
    yb.download(0, &mut out).unwrap();
    let gpu: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (b, x) in xs.iter().enumerate() {
        for r in [0usize, rows / 2, rows - 1] {
            let row = &w[r * cols * 2..(r + 1) * cols * 2];
            let want: f32 = row
                .chunks_exact(2)
                .zip(x)
                .map(|(c, xv)| f16_bits_to_f32(u16::from_le_bytes(c.try_into().unwrap())) * xv)
                .sum();
            let got = gpu[b * rows + r];
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "batch {b} row {r}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn batched_rope_matches_single_token_rope_per_position() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let (heads, kv_heads, head_dim) = (16usize, 8usize, 128usize);
    let inv = rope_freqs(head_dim, 1e6);
    let n_tokens = 5usize;
    let start = 7usize;
    let mut rng = Rng(43);
    let q: Vec<f32> = (0..n_tokens * heads * head_dim)
        .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
        .collect();
    let k: Vec<f32> = (0..n_tokens * kv_heads * head_dim)
        .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
        .collect();
    let qb = ctx.buffer(q.len() * 4, true).unwrap();
    let kb = ctx.buffer(k.len() * 4, true).unwrap();
    let fb = ctx.buffer(inv.len() * 4, true).unwrap();
    qb.upload(0, &f32_bytes(&q)).unwrap();
    kb.upload(0, &f32_bytes(&k)).unwrap();
    fb.upload(0, &f32_bytes(&inv)).unwrap();
    ctx.rope_batch(
        &qb,
        &kb,
        &fb,
        start as u32,
        n_tokens as u32,
        heads as u32,
        kv_heads as u32,
        head_dim as u32,
    )
    .unwrap();
    ctx.fence().unwrap();
    let mut q_out = vec![0u8; q.len() * 4];
    let mut k_out = vec![0u8; k.len() * 4];
    qb.download(0, &mut q_out).unwrap();
    kb.download(0, &mut k_out).unwrap();
    let gq: Vec<f32> = q_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let gk: Vec<f32> = k_out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for t in 0..n_tokens {
        let base = t * heads * head_dim;
        for h in 0..heads {
            let mut want = q[base + h * head_dim..base + (h + 1) * head_dim].to_vec();
            cpu_rope(&mut want, start + t, &inv);
            for d in 0..head_dim {
                let (g, w) = (gq[base + h * head_dim + d], want[d]);
                assert!(
                    (g - w).abs() <= 2e-4 * w.abs().max(1.0),
                    "t {t} head {h} dim {d}: {g} vs {w}"
                );
            }
        }
        let kbase = t * kv_heads * head_dim;
        for h in 0..kv_heads {
            let mut want = k[kbase + h * head_dim..kbase + (h + 1) * head_dim].to_vec();
            cpu_rope(&mut want, start + t, &inv);
            for d in 0..head_dim {
                let (g, w) = (gk[kbase + h * head_dim + d], want[d]);
                assert!(
                    (g - w).abs() <= 2e-4 * w.abs().max(1.0),
                    "k t {t} head {h} dim {d}: {g} vs {w}"
                );
            }
        }
    }
}

#[test]
fn batched_kv_store_and_attention_match_the_scalar_reference() {
    let Some(ctx) = metal_context() else {
        return;
    };
    let (heads, kv_heads, head_dim, group) = (16usize, 8usize, 128usize, 2usize);
    let cap = 64usize;
    let n_tokens = 6usize;
    let start = 5usize;
    let mut rng = Rng(47);

    let mut k_steps: Vec<Vec<f32>> = Vec::new();
    let mut v_steps: Vec<Vec<f32>> = Vec::new();
    let mut q_steps: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n_tokens {
        k_steps.push(
            (0..kv_heads * head_dim)
                .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
                .collect(),
        );
        v_steps.push(
            (0..kv_heads * head_dim)
                .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
                .collect(),
        );
        q_steps.push(
            (0..heads * head_dim)
                .map(|_| ((rng.next() % 20_001) as f32 - 10_000.0) / 1000.0)
                .collect(),
        );
    }
    let flat = |steps: &[Vec<f32>]| -> Vec<f32> {
        let mut out = Vec::new();
        for s in steps {
            out.extend_from_slice(s);
        }
        out
    };
    let k_flat = flat(&k_steps);
    let v_flat = flat(&v_steps);
    let q_flat = flat(&q_steps);

    let k_cache = ctx.buffer(kv_heads * cap * head_dim * 4, true).unwrap();
    let v_cache = ctx.buffer(kv_heads * cap * head_dim * 4, true).unwrap();
    let scratch = ctx.buffer(n_tokens * heads * cap * 4, true).unwrap();
    let qb = ctx.buffer(q_flat.len() * 4, true).unwrap();
    let out = ctx.buffer(q_flat.len() * 4, true).unwrap();
    let kb = ctx.buffer(k_flat.len() * 4, true).unwrap();
    let vb = ctx.buffer(v_flat.len() * 4, true).unwrap();
    qb.upload(0, &f32_bytes(&q_flat)).unwrap();
    kb.upload(0, &f32_bytes(&k_flat)).unwrap();
    vb.upload(0, &f32_bytes(&v_flat)).unwrap();

    ctx.begin_group().unwrap();
    ctx.kv_store_batch(
        &kb,
        &vb,
        &k_cache,
        &v_cache,
        n_tokens as u32,
        start as u32,
        cap as u32,
        head_dim as u32,
        kv_heads as u32,
        false,
    )
    .unwrap();
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    ctx.attention_batch(
        &qb,
        &k_cache,
        &v_cache,
        &out,
        &scratch,
        heads as u32,
        n_tokens as u32,
        start as u32,
        cap as u32,
        head_dim as u32,
        group as u32,
        scale,
    )
    .unwrap();
    ctx.end_group().unwrap();
    ctx.fence().unwrap();

    let mut out_bytes = vec![0u8; q_flat.len() * 4];
    out.download(0, &mut out_bytes).unwrap();
    let got: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // CPU reference: per token, causal softmax over ALL positions 0..=pos.
    // Positions before `start` were never written and stay at their zero
    // initialization, so they contribute score 0 and no value — mirroring
    // the kernel exactly (the earlier version of this reference forgot the
    // zero-initialized prefix and disagreed with a correct kernel).
    for t in 0..n_tokens {
        let pos = start + t;
        for h in 0..heads {
            let kvh = h / group;
            let qh = &q_steps[t][h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::with_capacity(pos + 1);
            for j in 0..=pos {
                if j < start {
                    scores.push(0.0);
                    continue;
                }
                let ks = &k_steps[j - start];
                let kj = &ks[kvh * head_dim..(kvh + 1) * head_dim];
                let dot: f32 = qh.iter().zip(kj).map(|(a, b)| a * b).sum();
                scores.push(dot * scale);
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (j, e) in exps.iter().enumerate() {
                    if j < start {
                        continue;
                    }
                    acc += (e / denom) * v_steps[j - start][kvh * head_dim + d];
                }
                let got_v = got[(t * heads + h) * head_dim + d];
                assert!(
                    (got_v - acc).abs() <= 1e-3,
                    "t {t} head {h} dim {d}: {got_v} vs {acc}"
                );
            }
        }
    }
}
