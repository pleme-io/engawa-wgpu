//! MATRIX FORCING TEST — one row per catalog effect.
//!
//! Per row this asserts:
//!   (a) the effect lowers to the expected node count (>= 1)
//!       AND its canonical graph compiles (wiring proof),
//!   (b) the Params struct is Pod (`default_params_bytes` only
//!       compiles for Pod) and its byte length == `params_size`,
//!   (c) `size_of::<Params>()` == the `(params-size N)` declared
//!       in `effects/<name>.tlisp` — the file is read from disk
//!       here, so the authored form IS the contract,
//!   (d) the priority sits in engawa's post range (200..=799)
//!       and matches the `(priority N)` in the .tlisp form.
//!
//! The registry-coverage test pins `MATRIX.len() ==
//! CatalogEffect::ALL.len()` — `ALL` is emitted by
//! `pleme-allvariants-derive`, so a new enum variant grows the
//! registry mechanically and a missing matrix row fails the
//! build. Failures aggregate before asserting: one run reports
//! every broken effect, not just the first.

use std::path::PathBuf;

use engawa_wgpu::catalog::CatalogEffect;

struct MatrixRow {
    effect: CatalogEffect,
    expected_nodes: usize,
}

const MATRIX: &[MatrixRow] = &[
    MatrixRow { effect: CatalogEffect::Colorblind, expected_nodes: 1 },
    MatrixRow { effect: CatalogEffect::Crt, expected_nodes: 1 },
    MatrixRow { effect: CatalogEffect::Scanlines, expected_nodes: 1 },
    MatrixRow { effect: CatalogEffect::Bloom, expected_nodes: 4 },
    MatrixRow { effect: CatalogEffect::GlowOnBell, expected_nodes: 1 },
    MatrixRow { effect: CatalogEffect::Snow, expected_nodes: 1 },
];

/// Typed failure rows — Debug-rendered in the final aggregate
/// assert so one run reports every broken effect.
#[derive(Debug)]
#[allow(dead_code)] // fields exist to be Debug-rendered in the failure report
enum Failure {
    LoweredNodeCount { effect: &'static str, expected: usize, actual: usize },
    GraphCompile { effect: &'static str, error: String },
    PodBytesLen { effect: &'static str, params_size: usize, actual: usize },
    TlispUnreadable { effect: &'static str, path: PathBuf },
    TlispParamsSizeUnparsable { effect: &'static str, path: PathBuf },
    TlispParamsSizeMismatch { effect: &'static str, tlisp: usize, rust: usize },
    TlispPriorityUnparsable { effect: &'static str, path: PathBuf },
    TlispPriorityMismatch { effect: &'static str, tlisp: u64, rust: u16 },
    PriorityOutsidePostRange { effect: &'static str, priority: u16 },
}

/// Parse `(<key> <integer>)` out of a tlisp source. Text-level
/// on purpose: the test depends on the authored bytes, not on a
/// lisp runtime.
fn tlisp_uint(src: &str, key: &str) -> Option<u64> {
    let start = src.find(key)? + key.len();
    let rest = &src[start..];
    let end = rest.find(')')?;
    rest[..end].trim().parse().ok()
}

#[test]
fn every_catalog_effect_satisfies_the_contract() {
    let mut failures: Vec<Failure> = Vec::new();

    for row in MATRIX {
        let e = row.effect;
        let name = e.name();

        // (a) lowering node count + graph wiring.
        let nodes = e.lower(&"scene".into(), &"out".into());
        if nodes.is_empty() || nodes.len() != row.expected_nodes {
            failures.push(Failure::LoweredNodeCount {
                effect: name,
                expected: row.expected_nodes,
                actual: nodes.len(),
            });
        }
        if let Err(err) = e.graph().compile() {
            failures.push(Failure::GraphCompile { effect: name, error: err.to_string() });
        }

        // (b) Pod proof + byte-length equality.
        let bytes = e.default_params_bytes();
        if bytes.len() != e.params_size() {
            failures.push(Failure::PodBytesLen {
                effect: name,
                params_size: e.params_size(),
                actual: bytes.len(),
            });
        }

        // (c) + (d): the authored .tlisp form is the contract.
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(e.tlisp_path());
        match std::fs::read_to_string(&path) {
            Err(_) => failures.push(Failure::TlispUnreadable { effect: name, path }),
            Ok(src) => {
                match tlisp_uint(&src, "(params-size") {
                    None => failures.push(Failure::TlispParamsSizeUnparsable {
                        effect: name,
                        path: path.clone(),
                    }),
                    Some(declared) => {
                        let declared = usize::try_from(declared).expect("declared size fits");
                        if declared != e.params_size() {
                            failures.push(Failure::TlispParamsSizeMismatch {
                                effect: name,
                                tlisp: declared,
                                rust: e.params_size(),
                            });
                        }
                    }
                }
                match tlisp_uint(&src, "(priority") {
                    None => failures.push(Failure::TlispPriorityUnparsable {
                        effect: name,
                        path: path.clone(),
                    }),
                    Some(declared) => {
                        if declared != u64::from(e.priority()) {
                            failures.push(Failure::TlispPriorityMismatch {
                                effect: name,
                                tlisp: declared,
                                rust: e.priority(),
                            });
                        }
                    }
                }
            }
        }

        if !(200..=799).contains(&e.priority()) {
            failures.push(Failure::PriorityOutsidePostRange {
                effect: name,
                priority: e.priority(),
            });
        }
    }

    assert!(
        failures.is_empty(),
        "{} catalog matrix checks failed:\n{:#?}",
        failures.len(),
        failures
    );
}

#[test]
fn matrix_covers_every_registered_effect() {
    assert_eq!(
        MATRIX.len(),
        CatalogEffect::ALL.len(),
        "every CatalogEffect variant MUST have a matrix row — \
         ALL is derived from the enum, so a new effect without a \
         row fails here"
    );
    for e in CatalogEffect::ALL {
        let occurrences = MATRIX.iter().filter(|r| r.effect == *e).count();
        assert_eq!(occurrences, 1, "effect {e:?} must appear exactly once in MATRIX");
    }
}

#[test]
fn names_priorities_and_params_resources_are_unique() {
    use std::collections::BTreeSet;
    let names: BTreeSet<&str> = CatalogEffect::ALL.iter().map(|e| e.name()).collect();
    let params: BTreeSet<&str> =
        CatalogEffect::ALL.iter().map(|e| e.params_resource()).collect();
    let priorities: BTreeSet<u16> =
        CatalogEffect::ALL.iter().map(|e| e.priority()).collect();
    assert_eq!(names.len(), CatalogEffect::ALL.len(), "duplicate effect name");
    assert_eq!(params.len(), CatalogEffect::ALL.len(), "duplicate params resource");
    assert_eq!(
        priorities.len(),
        CatalogEffect::ALL.len(),
        "duplicate priority — render order would be ambiguous"
    );
}
