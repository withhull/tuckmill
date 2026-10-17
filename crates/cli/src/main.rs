//! Tuckmill command line interface.
//!
//! Exit codes: 0 success, 1 typed failure (printed to stderr with a safe next
//! action), 2 usage error (clap), 130 interrupted.
//! Data goes to stdout; diagnostics go to stderr.

mod commands;

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use clap::{Parser, Subcommand};

use tuckmill_manifest::error::TuckmillError;

/// Set by the SIGINT handler.
pub static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sigint(_sig: libc::c_int) {
    INTERRUPTED.store(true, Ordering::SeqCst);
    if let Some(flag) = commands::serve_shutdown_flag() {
        flag.store(true, Ordering::SeqCst);
    }
}

pub fn install_sigint_handler() {
    // SAFETY: installing a signal handler for SIGINT with a function that
    // only performs atomic stores (async-signal-safe).
    unsafe {
        libc::signal(libc::SIGINT, handle_sigint as *const () as usize);
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "tuckmill",
    version,
    about = "Big models. Small footprint. High-performance local inference for Apple Silicon Macs (uncertified engineering build).",
    propagate_version = true
)]
struct Cli {
    /// Machine-readable JSON output on stdout.
    #[arg(long, global = true)]
    json: bool,

    /// Data directory (defaults to ~/Library/Application Support/Tuckmill).
    #[arg(long, global = true, env = "TUCKMILL_DATA_DIR", hide_env_values = true)]
    data_dir: Option<PathBuf>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Report chip, memory, OS, backends, storage, and missing requirements.
    Doctor {
        /// Run the explicit, bounded storage probe (writes a temporary file
        /// under the data directory; never reads personal files).
        #[arg(long)]
        probe_storage: bool,
    },

    /// Inspect a model artifact without loading weights.
    Inspect {
        /// Path to a .gguf or .safetensors file.
        path: PathBuf,
    },

    /// Import a local artifact into the managed store (validated, atomic).
    Import {
        path: PathBuf,
        /// Register an alias for the imported artifact.
        #[arg(long)]
        alias: Option<String>,
        /// License note recorded with the artifact (not a license review).
        #[arg(long)]
        license: Option<String>,
        /// Source revision note recorded with the artifact.
        #[arg(long)]
        revision: Option<String>,
    },

    /// Pack an imported artifact into the bounded, layout-preserving format.
    Pack {
        /// Alias or artifact id.
        model: String,
    },

    /// Produce an execution plan or a typed infeasibility explanation.
    Plan {
        /// Alias or artifact id.
        model: String,
        /// Requested process budget (binary units, e.g. 2GiB). Defaults to
        /// a hardware-adaptive quarter of unified memory (2–24GiB).
        #[arg(long)]
        memory_budget: Option<String>,
        /// Total sequence allowance in tokens, including output.
        #[arg(long, default_value_t = 2048)]
        context: u64,
        /// Maximum output tokens (part of --context).
        #[arg(long, default_value_t = 256)]
        max_output: u64,
    },

    /// Run one generation request.
    Run {
        /// Alias or artifact id.
        model: String,
        /// Prompt text (reads stdin when omitted).
        #[arg(long)]
        prompt: Option<String>,
        /// Requested process budget (binary units, e.g. 2GiB). Defaults to
        /// a hardware-adaptive quarter of unified memory (2–24GiB).
        #[arg(long)]
        memory_budget: Option<String>,
        #[arg(long, default_value_t = 2048)]
        context: u64,
        #[arg(long, default_value_t = 256)]
        max_output: u64,
        /// Accept the reference test engine (synthetic output; executes no
        /// model weights). Without this flag, non-certified profiles are
        /// refused.
        #[arg(long)]
        allow_test_engine: bool,
        /// Backend: reference-test (default; also 'isolated'), tuckcore, or
        /// llama (pinned llama.cpp reference). 'in-process' is a debugging
        /// placement for the reference test engine.
        #[arg(long, default_value = "isolated")]
        engine: String,
        /// Suppress state messages on stderr.
        #[arg(long)]
        quiet: bool,
        /// llama.cpp GPU offload layers (Metal performance baseline ONLY;
        /// never for arithmetic diagnostics). Requires --engine llama.
        #[arg(long)]
        gpu_layers: Option<i32>,
        /// Kernel implementation for tuckcore: auto (default; NEON where
        /// available), scalar (force the reference kernels), neon.
        #[arg(long)]
        kernels: Option<String>,
        /// Weight-block storage for tuckcore: auto (default; map when
        /// possible), mapped (require zero-copy views), owned (copied
        /// bounded buffers).
        #[arg(long)]
        storage: Option<String>,
        /// Metal prefill micro-batch for tuckcore (1..=32; default 8). The
        /// effective size is the largest candidate whose exact workspace
        /// geometry fits the admitted plan.
        #[arg(long)]
        prefill_batch: Option<u32>,
    },

    /// Serve the local API on loopback.
    Serve {
        #[arg(long, default_value_t = tuckmill_supervisor::api::DEFAULT_PORT)]
        port: u16,
        /// Accept the reference test engine for incoming requests.
        #[arg(long)]
        allow_test_engine: bool,
        #[arg(long, default_value = "isolated")]
        engine: String,
        /// Start and exit immediately (used by tests to validate binding).
        #[arg(long, hide = true)]
        probe_only: bool,
        /// llama.cpp GPU offload layers (Metal performance baseline ONLY;
        /// never for arithmetic diagnostics). Requires --engine llama.
        #[arg(long)]
        gpu_layers: Option<i32>,
        /// Kernel implementation for tuckcore: auto (default; NEON where
        /// available), scalar (force the reference kernels), neon, metal.
        #[arg(long)]
        kernels: Option<String>,
        /// Weight-block storage for tuckcore: auto (default; map when
        /// possible), mapped (require zero-copy views), owned (copied
        /// bounded buffers).
        #[arg(long)]
        storage: Option<String>,
        /// Metal prefill micro-batch for tuckcore (1..=32; default 8).
        #[arg(long)]
        prefill_batch: Option<u32>,
    },

    /// Verify equivalence: identical token IDs through resident and
    /// forced-streaming tuckcore configurations, comparing teacher-forced
    /// logits and generated text (ADR-005).
    Verify {
        /// Alias or artifact id.
        model: String,
        /// Requested process budget (binary units, e.g. 2GiB). Defaults to
        /// a hardware-adaptive quarter of unified memory (2–24GiB).
        #[arg(long)]
        memory_budget: Option<String>,
        #[arg(long, default_value_t = 2048)]
        context: u64,
        /// Number of deterministic token ids scored per side.
        #[arg(long, default_value_t = 6)]
        positions: u64,
        /// Tokens generated per side for text equality.
        #[arg(long, default_value_t = 8)]
        generate: u64,
        /// Compare against the pinned llama.cpp reference backend instead of
        /// the resident/streamed scheduling comparison (requires
        /// TUCKMILL_LLAMA_LIB; see scripts/fetch-llama.sh).
        #[arg(long)]
        reference: bool,
        /// Override the registered logit tolerance for --reference
        /// comparisons (default: registered per dominant weight dtype —
        /// 5e-2 for F16/F32, 5e-1 for Q8_0).
        #[arg(long, default_value_t = 0.0)]
        tolerance: f64,
        /// TuckCore numerical profile for --reference comparisons:
        /// "llama-cpu" (mirrors pinned llama.cpp CPU kernels; default) or
        /// "oracle-f32" (scalar FP32 reference, no activation quantization).
        #[arg(long, default_value = "llama-cpu")]
        tuckcore_profile: String,
        /// KV cache precision for both sides in --reference comparisons:
        /// "f16" (matches llama.cpp's default) or "f32".
        #[arg(long, default_value = "f16")]
        kv_precision: String,
        /// Held-out qualification run: frozen inputs disjoint from the
        /// calibration workload (required before a configuration counts as
        /// qualified, not just calibrated).
        #[arg(long)]
        held_out: bool,
        /// Kernel implementation to exercise: auto (default), scalar, neon.
        #[arg(long)]
        kernels: Option<String>,
        /// Weight-block storage to exercise: auto (default), mapped, owned.
        #[arg(long)]
        storage: Option<String>,
        /// Metal prefill micro-batch for tuckcore (1..=32; default 8).
        #[arg(long)]
        prefill_batch: Option<u32>,
    },

    /// Run the standard benchmark suite and write a machine-readable record.
    Bench {
        /// Alias or artifact id.
        model: String,
        #[arg(long, default_value = "standard-v1")]
        suite: String,
        #[arg(long, default_value_t = 512)]
        input: u32,
        #[arg(long, default_value_t = 256)]
        output: u32,
        #[arg(long, default_value_t = 3)]
        repetitions: u32,
        /// Requested process budget (binary units, e.g. 2GiB). Defaults to
        /// a hardware-adaptive quarter of unified memory (2–24GiB).
        #[arg(long)]
        memory_budget: Option<String>,
        #[arg(long, default_value_t = 2048)]
        context: u64,
        #[arg(long)]
        allow_test_engine: bool,
        /// Backend selection: reference-test (default) or tuckcore.
        #[arg(long, default_value = "isolated")]
        engine: String,
        /// llama.cpp GPU offload layers (Metal performance baseline ONLY;
        /// never for arithmetic diagnostics). Requires --engine llama.
        #[arg(long)]
        gpu_layers: Option<i32>,
        /// Kernel implementation for tuckcore: auto (default; NEON where
        /// available), scalar (force the reference kernels), neon.
        #[arg(long)]
        kernels: Option<String>,
        /// Weight-block storage for tuckcore: auto (default; map when
        /// possible), mapped (require zero-copy views), owned (copied
        /// bounded buffers).
        #[arg(long)]
        storage: Option<String>,
        /// Metal prefill micro-batch for tuckcore (1..=32; default 8). The
        /// effective size is the largest candidate whose exact workspace
        /// geometry fits the admitted plan.
        #[arg(long)]
        prefill_batch: Option<u32>,
    },

    /// Show backend statistics and profile counters after a short probe run.
    Stats {
        /// Alias or artifact id.
        model: String,
        /// Backend: tuckcore (default) or llama.
        #[arg(long, default_value = "tuckcore")]
        engine: String,
        /// Probe tokens to process before reporting.
        #[arg(long, default_value_t = 3)]
        tokens: u64,
        /// Kernel implementation to exercise while probing.
        #[arg(long)]
        kernels: Option<String>,
        /// Weight-block storage to exercise while probing.
        #[arg(long)]
        storage: Option<String>,
    },

    /// Cache inventory and garbage collection.
    Cache {
        #[command(subcommand)]
        cmd: CacheCmd,
    },

    /// Internal: serve an inference backend over the framed protocol on a
    /// private socket.
    #[command(name = "__worker-reference", hide = true)]
    WorkerReference {
        #[arg(long)]
        socket: PathBuf,
        /// Backend: reference-test or tuckcore.
        #[arg(long, default_value = "reference-test")]
        engine: String,
    },
}

#[derive(Subcommand, Debug)]
enum CacheCmd {
    /// Show installed artifacts and reclaimable space.
    List,
    /// Remove derived, inactive data (never artifacts or source files).
    Prune {
        #[arg(long)]
        dry_run: bool,
    },
}

fn main() -> ExitCode {
    install_sigint_handler();
    let cli = Cli::parse();
    match commands::dispatch(&cli.cmd, cli.json, cli.data_dir.as_deref()) {
        Ok(code) => code,
        Err(e) => {
            print_error(&e, cli.json);
            ExitCode::from(1)
        }
    }
}

fn print_error(e: &TuckmillError, json: bool) {
    if json {
        eprintln!(
            "{}",
            serde_json::to_string_pretty(&e.to_json()).unwrap_or_else(|_| e.to_string())
        );
    } else {
        eprintln!("error[{}]: {}", e.code(), e.message());
        eprintln!("  next: {}", e.action());
    }
}
