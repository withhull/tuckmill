//! Generate a small, valid synthetic GGUF fixture for local development.
//!
//! Usage:
//!   cargo run -p tuckmill-manifest --example gen_fixture -- out.gguf [--moe] [--layers N] [--hidden N]
//!
//! The fixture proves code paths (parsing, planning, packing, pipeline). It
//! is not evidence about any real model's behavior.

use std::process::ExitCode;

use tuckmill_manifest::gguf::{inspect, Limits};
use tuckmill_manifest::testing::{write_synthetic, SyntheticConfig};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: gen_fixture <out.gguf> [--moe] [--layers N] [--hidden N]");
        return ExitCode::from(2);
    };
    let mut cfg = SyntheticConfig::tiny_qwen3();
    let mut rest: Vec<String> = args.collect();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--moe" => {
                cfg = SyntheticConfig::tiny_qwen3moe();
                i += 1;
            }
            "--layers" => {
                if i + 1 >= rest.len() {
                    eprintln!("--layers needs a value");
                    return ExitCode::from(2);
                }
                match rest[i + 1].parse() {
                    Ok(v) => cfg.layers = v,
                    Err(_) => {
                        eprintln!("--layers must be an integer");
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
            "--hidden" => {
                if i + 1 >= rest.len() {
                    eprintln!("--hidden needs a value");
                    return ExitCode::from(2);
                }
                match rest[i + 1].parse() {
                    Ok(v) => cfg.hidden = v,
                    Err(_) => {
                        eprintln!("--hidden must be an integer");
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
            other => {
                eprintln!("unknown argument {other:?}");
                return ExitCode::from(2);
            }
        }
    }
    rest.clear();

    let path = std::path::PathBuf::from(&path);
    match write_synthetic(&path, &cfg) {
        Ok(len) => {
            let insp = inspect(&path, &Limits::default()).expect("freshly written fixture parses");
            let stats = insp.tensor_stats();
            println!("wrote {} ({} bytes)", path.display(), len);
            println!("architecture: {}", insp.arch_id().unwrap_or_default());
            println!("tensors: {}", stats.total_tensors);
            println!("tensor bytes: {}", stats.total_bytes);
            println!("identity: {}", insp.identity_hash().unwrap());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
