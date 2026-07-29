use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstructionTelemetry {
    pub name: String,
    pub opcode: String,
    pub resource: String,
    pub submitted_s: f64,
    pub start_s: f64,
    pub end_s: f64,
    pub nbytes: u64,
    pub simulated: bool,
    pub notes: String,
}

/// Peak concurrency via sweep over `[start_s, end_s)` intervals.
///
/// Ends at time `t` are processed before starts at `t`, so back-to-back
/// intervals do not count as overlapping.
#[must_use]
pub fn max_concurrency_from_intervals(intervals: &[(f64, f64)]) -> usize {
    if intervals.is_empty() {
        return 0;
    }
    let mut points: Vec<(f64, i32)> = Vec::with_capacity(intervals.len() * 2);
    for &(start, end) in intervals {
        if end <= start {
            continue;
        }
        points.push((start, 1));
        points.push((end, -1));
    }
    points.sort_by(|a, b| {
        match a.0.partial_cmp(&b.0) {
            Some(ord) => ord.then_with(|| a.1.cmp(&b.1)),
            None => a.1.cmp(&b.1),
        }
    });
    let mut cur = 0i32;
    let mut peak = 0i32;
    for (_, delta) in points {
        cur += delta;
        if cur > peak {
            peak = cur;
        }
    }
    peak.max(0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_known_overlaps() {
        // [0,2) and [1,3) overlap → 2; [3,4) alone → peak 2
        assert_eq!(
            max_concurrency_from_intervals(&[(0.0, 2.0), (1.0, 3.0), (3.0, 4.0)]),
            2
        );
        // abutting: no overlap
        assert_eq!(
            max_concurrency_from_intervals(&[(0.0, 1.0), (1.0, 2.0)]),
            1
        );
        assert_eq!(max_concurrency_from_intervals(&[]), 0);
        assert_eq!(
            max_concurrency_from_intervals(&[(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)]),
            3
        );
    }
}
