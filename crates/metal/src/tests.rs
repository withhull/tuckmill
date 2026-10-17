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
