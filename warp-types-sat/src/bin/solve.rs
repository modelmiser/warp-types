//! CLI entry point: solve a DIMACS CNF file.
//!
//! Usage: cargo run -p warp-types-sat --bin solve -- [--drat <proof.drat> [--drat-binary] | --lrat <proof.lrat> [--lrat-binary]] [--certify] [--hybrid] [--reduce-interval <n>] <file.cnf>
//!
//! `--drat FILE` writes a DRAT proof of an UNSATISFIABLE answer to FILE
//! (checkable with `drat-trim FILE.cnf FILE.drat`). On SATISFIABLE the file is
//! still written but proves nothing. See `warp_types_sat::proof`.
//!
//! `--certify` with `--drat` runs an external DRAT checker on the proof
//! before printing an UNSATISFIABLE verdict: `drat-trim` from `$DRAT_TRIM` or
//! `PATH`. If the checker is missing, cannot run, or does not print
//! `s VERIFIED`, the answer is printed as `s UNKNOWN` with the reason on
//! stderr and the exit status is 2. Fail-closed: an absent checker is not a
//! pass. With `--lrat` the checker is `lrat-check` (`$LRAT_CHECK` or `PATH`).
//! Without either, an UNSATISFIABLE answer cannot be certified and is printed
//! as `s UNKNOWN` (exit 2); `--certify` alone is for checking models.
//!
//! Under `--certify` a SATISFIABLE answer is checked too: the model is
//! evaluated against the clauses read again from the file by
//! [`model_check`], a reader that shares no code with the crate's DIMACS
//! parser — so a parser that dropped a clause cannot certify its own model.
//! A model that falsifies a clause, or does not cover the declared
//! variables, is printed as `s UNKNOWN` with exit status 2.
//!
//! `--hybrid` solves with the gradient→CDCL hybrid
//! (`gradient::hybrid_solve_{drat,lrat}`, default `GradientConfig`) instead
//! of plain CDCL. Proofs and `--certify` work the same way; when the gradient
//! phase finds the model, CDCL never runs and the proof file is empty.
//!
//! `--reduce-interval N` sets the conflicts between learned-clause reductions
//! (default 2000). A diagnostic: a small N makes short solves exercise clause
//! deletion, which is what proof-logging coverage needs. Honoured with and
//! without `--drat`, so timings of the two are like-for-like.
//!
//! Note: loads the entire file into memory via `fs::read_to_string`.
//! Not suitable for adversarial input without external size limits.

use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::process;

fn usage(prog: &str) -> ! {
    eprintln!("Usage: {prog} [--drat <proof.drat> [--drat-binary] | --lrat <proof.lrat> [--lrat-binary]] [--certify] [--hybrid] [--reduce-interval <n>] <file.cnf>");
    process::exit(1);
}

/// True when `proof` names the same file as `cnf`. The proof file may not
/// exist yet, so compare the canonical parent directory plus file name.
fn same_file(cnf: &str, proof: &str) -> bool {
    use std::path::Path;
    let canon = |p: &str| -> Option<std::path::PathBuf> {
        let p = Path::new(p);
        let parent = p
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        Some(fs::canonicalize(parent).ok()?.join(p.file_name()?))
    };
    match (canon(cnf), canon(proof)) {
        (Some(a), Some(b)) => a == b,
        _ => cnf == proof,
    }
}

/// Check `model` (index i = variable i+1) against the clauses in the DIMACS
/// text, read here with a deliberately separate reader: comment (`c`) and
/// header (`p`) lines are skipped, `%` ends the input (SATLIB), every other
/// token is an integer and `0` closes a clause; a trailing clause without its
/// `0` is checked too. Ok carries the number of clauses checked.
fn model_check(text: &str, model: &[bool]) -> Result<usize, String> {
    let mut declared: Option<usize> = None;
    let mut clause: Vec<i64> = Vec::new();
    let mut checked = 0usize;
    let check = |c: &[i64]| -> Result<(), String> {
        let sat = c.iter().any(|&l| {
            let v = l.unsigned_abs() as usize;
            v >= 1 && v <= model.len() && model[v - 1] == (l > 0)
        });
        if sat {
            Ok(())
        } else {
            Err(format!("the model falsifies clause {c:?}"))
        }
    };
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('%') {
            break;
        }
        if t.is_empty() || t.starts_with('c') {
            continue;
        }
        if t.starts_with('p') {
            let n = t.split_whitespace().nth(2).and_then(|n| n.parse().ok());
            declared = Some(n.ok_or_else(|| format!("unreadable header: {t}"))?);
            continue;
        }
        for tok in t.split_whitespace() {
            let l: i64 = tok.parse().map_err(|_| format!("bad token {tok:?}"))?;
            if l == 0 {
                check(&clause)?;
                checked += 1;
                clause.clear();
            } else {
                clause.push(l);
            }
        }
    }
    if !clause.is_empty() {
        check(&clause)?;
        checked += 1;
    }
    match declared {
        Some(n) if n == model.len() => Ok(checked),
        Some(n) => Err(format!(
            "the model has {} values for {n} declared variables",
            model.len()
        )),
        None => Err("no header".into()),
    }
}

/// Locate lrat-check (`$LRAT_CHECK`, else `PATH`) and run it on `cnf proof`.
/// Fail-closed like [`run_drat_trim`]: only `c VERIFIED` with exit 0 passes
/// (lrat-check's exit code alone is not trusted — see the research log).
fn run_lrat_check(cnf: &str, proof: &str) -> Result<(), String> {
    let bin = env::var("LRAT_CHECK")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            env::var_os("PATH").and_then(|p| {
                env::split_paths(&p)
                    .map(|d| d.join("lrat-check"))
                    .find(|c| c.is_file())
            })
        })
        .ok_or_else(|| "no LRAT checker: set LRAT_CHECK or put lrat-check on PATH".to_string())?;
    let out = process::Command::new(&bin)
        .arg(cnf)
        .arg(proof)
        .output()
        .map_err(|e| format!("could not run {}: {e}", bin.display()))?;
    let text = String::from_utf8_lossy(&out.stdout);
    if out.status.success() && text.lines().any(|l| l.trim() == "c VERIFIED") {
        Ok(())
    } else {
        Err(format!(
            "{} did not verify the proof (exit {:?}): {}",
            bin.display(),
            out.status.code(),
            text.lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("")
        ))
    }
}

/// Locate drat-trim (`$DRAT_TRIM`, else `PATH`) and run it on `cnf proof`.
/// Ok only when its output contains `s VERIFIED`.
fn run_drat_trim(cnf: &str, proof: &str, binary: bool) -> Result<(), String> {
    let bin = env::var("DRAT_TRIM").ok().or_else(|| {
        env::var_os("PATH").and_then(|p| {
            env::split_paths(&p)
                .map(|d| d.join("drat-trim"))
                .find(|f| f.is_file())
                .map(|f| f.to_string_lossy().into_owned())
        })
    });
    let Some(bin) = bin else {
        return Err("no DRAT checker: set DRAT_TRIM or put drat-trim on PATH".into());
    };
    let mut cmd = process::Command::new(&bin);
    cmd.arg(cnf).arg(proof);
    if binary {
        cmd.arg("-i"); // drat-trim detects the layout itself; say it anyway
    }
    let out = cmd
        .output()
        .map_err(|e| format!("could not run {bin}: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout).replace('\r', "\n");
    if text.contains("s VERIFIED") && !text.contains("s NOT VERIFIED") {
        Ok(())
    } else {
        Err(format!(
            "{bin} did not verify the proof (exit {:?})",
            out.status.code()
        ))
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let prog = args.first().map(String::as_str).unwrap_or("solve");
    let mut drat_path: Option<String> = None;
    let mut lrat_path: Option<String> = None;
    let mut lrat_binary = false;
    let mut drat_binary = false;
    let mut reduce_interval: u64 = warp_types_sat::solver::DEFAULT_REDUCE_INTERVAL;
    let mut certify = false;
    let mut hybrid = false;
    let mut cnf_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--drat" => {
                i += 1;
                if i >= args.len() || drat_path.is_some() {
                    usage(prog);
                }
                drat_path = Some(args[i].clone());
            }
            "--lrat" => {
                i += 1;
                if i >= args.len() || lrat_path.is_some() {
                    usage(prog);
                }
                lrat_path = Some(args[i].clone());
            }
            "--lrat-binary" => lrat_binary = true,
            "--drat-binary" => drat_binary = true,
            "--certify" => certify = true,
            "--hybrid" => hybrid = true,
            "--reduce-interval" => {
                i += 1;
                reduce_interval = args
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .filter(|&n: &u64| n > 0)
                    .unwrap_or_else(|| usage(prog));
            }
            a if a.starts_with("--") => usage(prog),
            a => {
                if cnf_path.is_some() {
                    usage(prog);
                }
                cnf_path = Some(a.to_string());
            }
        }
        i += 1;
    }
    let Some(cnf_path) = cnf_path else {
        usage(prog)
    };
    if drat_path.is_some() && lrat_path.is_some() {
        eprintln!("Error: --drat and --lrat are exclusive (one proof per run)");
        process::exit(1);
    }
    if lrat_binary && lrat_path.is_none() {
        eprintln!("Error: --lrat-binary requires --lrat <proof.lrat>");
        process::exit(1);
    }
    if drat_binary && drat_path.is_none() {
        eprintln!("Error: --drat-binary requires --drat <proof.drat>");
        process::exit(1);
    }
    if certify && lrat_binary {
        // lrat-check reads text LRAT only; a binary proof would always be
        // reported NOT certified after the whole solve. Say so up front.
        eprintln!("Error: --certify checks LRAT with lrat-check, which reads text only; drop --lrat-binary");
        process::exit(1);
    }
    for d in drat_path.iter().chain(lrat_path.iter()) {
        if same_file(&cnf_path, d) {
            eprintln!("Error: proof path {d} is the input CNF; refusing to overwrite it");
            process::exit(1);
        }
    }

    let contents = fs::read_to_string(&cnf_path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", cnf_path, e);
        process::exit(1);
    });

    let instance = warp_types_sat::dimacs::parse_dimacs_str(&contents).unwrap_or_else(|e| {
        eprintln!("Parse error: {}", e);
        process::exit(1);
    });

    eprintln!(
        "c {} vars, {} clauses",
        instance.num_vars, instance.num_clauses,
    );

    use warp_types_sat::gradient::{
        hybrid_solve_drat, hybrid_solve_lrat, hybrid_solve_with, GradientConfig,
    };
    let config = GradientConfig::default();
    if hybrid {
        eprintln!("c solver: gradient -> CDCL hybrid (default GradientConfig)");
    }
    let result = match (&drat_path, &lrat_path) {
        (None, None) if hybrid => {
            hybrid_solve_with(instance.db, instance.num_vars, &config, reduce_interval)
        }
        (None, None) => warp_types_sat::solver::solve_watched_with(
            instance.db,
            instance.num_vars,
            reduce_interval,
        ),
        (None, Some(path)) => {
            let file = fs::File::create(path).unwrap_or_else(|e| {
                eprintln!("Error creating {}: {}", path, e);
                process::exit(1);
            });
            let mut out = BufWriter::new(file);
            let (r, st) = if hybrid {
                hybrid_solve_lrat(
                    instance.db,
                    instance.num_vars,
                    &config,
                    &mut out,
                    reduce_interval,
                    lrat_binary,
                )
            } else {
                warp_types_sat::solver::solve_watched_lrat_with(
                    instance.db,
                    instance.num_vars,
                    &mut out,
                    reduce_interval,
                    lrat_binary,
                )
            }
            .and_then(|r| out.flush().map(|_| r))
            .unwrap_or_else(|e| {
                eprintln!("Error writing proof to {}: {}", path, e);
                process::exit(1);
            });
            eprintln!(
                "c LRAT proof written to {} ({}; {} additions, {} deletions, {} hints)",
                path,
                if lrat_binary { "binary" } else { "text" },
                st.additions,
                st.deletions,
                st.hints
            );
            r
        }
        (Some(path), _) => {
            let file = fs::File::create(path).unwrap_or_else(|e| {
                eprintln!("Error creating {}: {}", path, e);
                process::exit(1);
            });
            let mut out = BufWriter::new(file);
            let r = if hybrid {
                hybrid_solve_drat(
                    instance.db,
                    instance.num_vars,
                    &config,
                    &mut out,
                    reduce_interval,
                    drat_binary,
                )
            } else {
                warp_types_sat::solver::solve_watched_drat_format(
                    instance.db,
                    instance.num_vars,
                    &mut out,
                    reduce_interval,
                    drat_binary,
                )
            }
            .and_then(|r| out.flush().map(|_| r))
            .unwrap_or_else(|e| {
                eprintln!("Error writing proof to {}: {}", path, e);
                process::exit(1);
            });
            eprintln!(
                "c DRAT proof written to {} ({})",
                path,
                if drat_binary { "binary" } else { "text" }
            );
            r
        }
    };

    if certify {
        if let warp_types_sat::solver::SolveResult::Sat(model) = &result {
            match model_check(&contents, model) {
                Ok(n) => eprintln!("c certified: the model satisfies all {n} clauses"),
                Err(why) => {
                    eprintln!("c NOT certified: {why}");
                    println!("s UNKNOWN");
                    process::exit(2);
                }
            }
        }
        if let warp_types_sat::solver::SolveResult::Unsat = result {
            let outcome = match (&drat_path, &lrat_path) {
                (Some(proof), _) => run_drat_trim(&cnf_path, proof, drat_binary)
                    .map(|()| format!("drat-trim verified {proof}")),
                (None, Some(proof)) => run_lrat_check(&cnf_path, proof)
                    .map(|()| format!("lrat-check verified {proof}")),
                (None, None) => {
                    Err("no proof was written: UNSAT needs --drat or --lrat to be certified".into())
                }
            };
            match outcome {
                Ok(msg) => eprintln!("c certified: {msg}"),
                Err(why) => {
                    eprintln!("c NOT certified: {why}");
                    println!("s UNKNOWN");
                    process::exit(2);
                }
            }
        }
    }

    match result {
        warp_types_sat::solver::SolveResult::Sat(assign) => {
            println!("s SATISFIABLE");
            print!("v");
            for (i, &val) in assign.iter().enumerate() {
                let lit = if val {
                    (i + 1) as i32
                } else {
                    -((i + 1) as i32)
                };
                print!(" {}", lit);
            }
            println!(" 0");
        }
        warp_types_sat::solver::SolveResult::Unsat => {
            println!("s UNSATISFIABLE");
        }
        warp_types_sat::solver::SolveResult::Unknown => {
            println!("s UNKNOWN");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::model_check;

    #[test]
    fn model_check_accepts_and_rejects() {
        let cnf = "c x\np cnf 3 2\n1 -2 0\n2 3 0\n";
        assert_eq!(model_check(cnf, &[true, false, true]), Ok(2));
        // (1 ∨ ¬2) falsified by 1=F, 2=T.
        assert!(model_check(cnf, &[false, true, true])
            .unwrap_err()
            .contains("falsifies"));
        // Too few / too many values for the declared variables.
        assert!(model_check(cnf, &[true, false]).is_err());
        assert!(model_check(cnf, &[true, false, true, true]).is_err());
        assert!(model_check("1 0\n", &[true])
            .unwrap_err()
            .contains("header"));
    }

    #[test]
    fn model_check_reads_clauses_the_way_dimacs_files_write_them() {
        // Clause split over lines; trailing clause without its 0 is checked.
        let cnf = "p cnf 2 2\n1\n2 0\n-1";
        assert!(model_check(cnf, &[true, true]).is_err(), "the -1 clause");
        assert_eq!(model_check(cnf, &[false, true]), Ok(2));
        // SATLIB: `%` ends the input; the lone 0 after it is not an empty clause.
        assert_eq!(model_check("p cnf 1 1\n1 0\n%\n0\n", &[true]), Ok(1));
        // An empty clause before `%` is falsified by every model.
        assert!(model_check("p cnf 1 1\n0\n", &[true]).is_err());
    }
}
