//! How an energy evaluation spreads over cores. The per-point work of the
//! searches and the smooth distances (`seen::Seen`, `neighbours::StaticList`)
//! runs on rayon's pool in chunks of about CHUNK points, cut where the work
//! allows, never by the thread count, and each member's own terms
//! (`members::Members::energy`) run one task per member. Each point's and
//! each member's answer depends on its own inputs alone, and the answers
//! are joined and summed in their order, so a result is bitwise the same
//! whatever the thread count, one included. The problems, the bursts and
//! their callbacks stay on the calling thread, between evaluations.

use std::ops::Range;

/// How many points one task takes, about. Measured with `./run bench
/// --only bus`, 64 and 1024 ran slower on the Z-shaped buses.
pub(crate) const CHUNK: usize = 256;

/// `0..len` in ranges of CHUNK.
pub(crate) fn chunks(len: usize) -> Vec<Range<usize>> {
    (0..len)
        .step_by(CHUNK)
        .map(|start| start..len.min(start + CHUNK))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Arc;

    use rayon::ThreadPoolBuilder;

    use crate::energy::EnergyParams;
    use crate::field::Field;
    use crate::geometry::V2;
    use crate::members::{Member, Members, Trace};
    use crate::optimize::problem::{Problem, ProblemVia};
    use crate::testing::{Disc, FreeSpace};
    use crate::vias::sight::ViaSight;

    const CHAINS: usize = 12;
    const VERTICES: usize = 40;
    const PITCH: f64 = 0.4;
    const MANY: usize = 8;

    fn params() -> EnergyParams {
        EnergyParams {
            repulsion: 0.5,
            falloff_range: 1.0,
            clearance_weight: 10.0,
            standoff: 0.05,
            points_per_edge: 3,
            attraction: 0.0,
            attraction_range: 3.0,
        }
    }

    fn chain(net: String, field: Arc<dyn Field>) -> Member {
        let trace = Trace {
            net: net.clone(),
            width: 0.2,
            clearance: 0.2,
        };
        Member {
            layer: "F.Cu".into(),
            viewer: trace.clone(),
            ignored: vec![net],
            shown: vec![trace],
            field,
            width: 0.2,
            contacts: vec![],
            section: None,
        }
    }

    /// A bus of CHAINS chains of as many nets in `field`, each bowed so that
    /// its neighbours press it, around a moving via that every chain sees;
    /// its layer's points span several chunks.
    fn bus(field: Arc<dyn Field>) -> (Problem, Vec<Vec<V2>>, Vec<V2>) {
        let members = (0..CHAINS)
            .map(|i| chain(format!("S{i}"), field.clone()))
            .collect();
        let polylines = (0..CHAINS)
            .map(|i| {
                (0..VERTICES)
                    .map(|k| {
                        let x = 0.25 * k as f64;
                        let bow = 0.1 * libm::sin(x + i as f64);
                        V2::new(x, PITCH * i as f64 + bow)
                    })
                    .collect()
            })
            .collect();
        let rings = vec![(0..CHAINS).map(|i| (i, 0.4)).collect()];
        let problem = Problem {
            members: Members::new(members),
            rings: rings.clone(),
            vias: vec![ProblemVia {
                diameter: 0.6,
                field: Arc::new(Disc {
                    cx: 5.0,
                    cy: 1.5,
                    radius: 0.3,
                }),
                stops: None,
            }],
            repel: ViaSight::new(&rings, &[vec![]], CHAINS),
            stop: ViaSight::new(&[vec![]], &[vec![]], CHAINS),
            pairs: None,
            params: params(),
            short_edge: 0.05,
        };
        (problem, polylines, vec![V2::new(5.0, 2.0)])
    }

    fn bits(points: &[V2]) -> Vec<u64> {
        points
            .iter()
            .flat_map(|p| [p.x.to_bits(), p.y.to_bits()])
            .collect()
    }

    /// The bus's energy and its gradients, searched and listed, and its
    /// stiffness, as bits, on a pool of `threads` threads.
    fn evaluated(threads: usize) -> Vec<Vec<u64>> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool");
        pool.install(|| {
            let (problem, polylines, positions) = bus(Arc::new(FreeSpace));
            let refs: Vec<&[V2]> = polylines.iter().map(Vec::as_slice).collect();
            let ends: Vec<Vec<bool>> = polylines
                .iter()
                .map(|p| (0..p.len()).map(|k| k == 0 || k + 1 == p.len()).collect())
                .collect();
            let masks: Vec<&[bool]> = ends.iter().map(Vec::as_slice).collect();
            let free: Vec<Vec<Option<usize>>> =
                polylines.iter().map(|p| vec![None; p.len()]).collect();
            let contact_of: Vec<&[Option<usize>]> = free.iter().map(Vec::as_slice).collect();
            let seen = problem.seen(&refs);
            let mut lookups = problem.lookups(&seen);
            let mut found = Vec::new();
            for listed in [false, true] {
                let energy = problem
                    .total(
                        &refs,
                        &positions,
                        &masks,
                        &contact_of,
                        &problem.short_edges(1.0),
                        &seen,
                        listed.then_some(&mut lookups),
                    )
                    .expect("finite");
                found.push(vec![energy.value.to_bits()]);
                found.extend(energy.gradients.iter().map(|g| bits(g)));
                found.push(bits(&energy.via_gradients));
            }
            let (members, vias) = problem.stiffness(&refs, &positions, &seen, Some(&mut lookups));
            found.extend(
                members
                    .iter()
                    .map(|s| s.iter().map(|v| v.to_bits()).collect()),
            );
            found.push(vias.iter().map(|v| v.to_bits()).collect());
            found
        })
    }

    #[test]
    fn an_evaluation_gives_the_same_bits_on_any_thread_count() {
        let one = evaluated(1);
        assert_ne!(f64::from_bits(one[0][0]), 0.0, "the bus has energy");
        assert_eq!(one, evaluated(MANY));
    }

    /// A field that panics when asked.
    struct Panicking;

    impl Field for Panicking {
        fn signed_distance(&self, _points: &[V2], _reach: f64) -> (Vec<f64>, Vec<V2>) {
            panic!("asked");
        }

        fn smooth_distance(&self, _points: &[V2], _reach: f64) -> (Vec<f64>, Vec<V2>) {
            panic!("asked");
        }

        fn segments_clear(&self, _starts: &[V2], _ends: &[V2], _tolerance: f64) -> Vec<bool> {
            panic!("asked");
        }
    }

    #[test]
    fn a_panic_in_a_worker_reaches_the_caller() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(MANY)
            .build()
            .expect("a pool");
        let (problem, polylines, positions) = bus(Arc::new(Panicking));
        let refs: Vec<&[V2]> = polylines.iter().map(Vec::as_slice).collect();
        let seen = problem.seen(&refs);
        let caught = catch_unwind(AssertUnwindSafe(|| {
            pool.install(|| problem.stiffness(&refs, &positions, &seen, None))
        }));
        let message = caught.expect_err("the panic reaches the caller");
        assert_eq!(message.downcast_ref::<&str>(), Some(&"asked"));
    }
}
