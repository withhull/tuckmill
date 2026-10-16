//! Score-frame reassembly with strict verification-integrity rules.
//!
//! The supervisor must never trust a scoring stream: a chunked logits row is
//! accepted only when its offsets are contiguous from zero, its total length
//! stays within hard bounds, and — at finish — exactly the requested
//! positions arrived, in order, once each. Any violation fails the request;
//! nothing is compared afterwards. Row content itself (exact vocabulary
//! length, finiteness) is validated by the caller that knows the expected
//! vocabulary.

use tuckmill_manifest::error::{ErrorCode, Result, TuckmillError};

/// Bound on logits buffered for one scored position; exceeds any legal
/// vocabulary (1,048,576) with headroom, and rejects oversize rows before
/// they are trusted.
pub const MAX_SCORE_VALUES_PER_POSITION: usize = 1_200_000;
/// Bound on logits buffered across a whole scoring request.
pub const MAX_SCORE_VALUES_TOTAL: usize = 12_000_000;

#[derive(Debug, Default)]
pub(crate) struct ScoreAccumulator {
    rows: Vec<(u64, Vec<f32>)>,
    total_values: usize,
}

impl ScoreAccumulator {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Accept one chunk. Offsets must continue exactly where the previous
    /// chunk for the same position ended; a new position must start at 0.
    pub(crate) fn push_chunk(&mut self, position: u64, offset: u32, logits: &[f32]) -> Result<()> {
        match self.rows.last_mut() {
            Some((last_position, buffer)) if *last_position == position => {
                if offset as usize != buffer.len() {
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        format!(
                            "score chunk integrity failure for position {position}: offset {offset} does not continue after {} values (duplicate, reordered, or lost chunk)",
                            buffer.len()
                        ),
                    ));
                }
                if buffer.len() + logits.len() > MAX_SCORE_VALUES_PER_POSITION {
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        format!(
                            "score row for position {position} exceeds {MAX_SCORE_VALUES_PER_POSITION} values"
                        ),
                    ));
                }
                buffer.extend_from_slice(logits);
            }
            _ => {
                if offset != 0 {
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        format!(
                            "score chunks for position {position} start at offset {offset}, not 0"
                        ),
                    ));
                }
                if logits.len() > MAX_SCORE_VALUES_PER_POSITION {
                    return Err(TuckmillError::new(
                        ErrorCode::WorkerFailed,
                        format!(
                            "score row for position {position} is {} values, over the {MAX_SCORE_VALUES_PER_POSITION} bound",
                            logits.len()
                        ),
                    ));
                }
                self.rows.push((position, logits.to_vec()));
            }
        }
        self.total_values = self.total_values.saturating_add(logits.len());
        if self.total_values > MAX_SCORE_VALUES_TOTAL {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                format!(
                    "scoring request accumulated {} values, over the {MAX_SCORE_VALUES_TOTAL} bound",
                    self.total_values
                ),
            ));
        }
        Ok(())
    }

    /// Terminate the request: the worker's `scored` count and the delivered
    /// rows must match the requested positions exactly, in order.
    pub(crate) fn finish(self, scored: u64, positions: &[u64]) -> Result<Vec<(u64, Vec<f32>)>> {
        let matches = self.rows.len() == scored as usize
            && self.rows.len() == positions.len()
            && self
                .rows
                .iter()
                .map(|(p, _)| *p)
                .eq(positions.iter().copied());
        if !matches {
            return Err(TuckmillError::new(
                ErrorCode::WorkerFailed,
                format!(
                    "worker scored {scored} positions but delivered {} rows {:?} (expected {:?})",
                    self.rows.len(),
                    self.rows.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
                    positions
                ),
            ));
        }
        Ok(self.rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acc() -> ScoreAccumulator {
        ScoreAccumulator::new()
    }

    #[test]
    fn contiguous_chunks_accumulate_in_order() {
        let mut a = acc();
        a.push_chunk(1, 0, &[1.0, 2.0]).unwrap();
        a.push_chunk(1, 2, &[3.0]).unwrap();
        a.push_chunk(3, 0, &[4.0, 5.0]).unwrap();
        let rows = a.finish(2, &[1, 3]).unwrap();
        assert_eq!(rows[0].1, vec![1.0, 2.0, 3.0]);
        assert_eq!(rows[1].1, vec![4.0, 5.0]);
    }

    #[test]
    fn duplicate_chunk_is_rejected() {
        let mut a = acc();
        a.push_chunk(1, 0, &[1.0, 2.0]).unwrap();
        let err = a.push_chunk(1, 0, &[1.0, 2.0]).unwrap_err();
        assert!(err.message().contains("does not continue"));
    }

    #[test]
    fn reordered_chunk_is_rejected() {
        let mut a = acc();
        a.push_chunk(1, 0, &[1.0, 2.0]).unwrap();
        let err = a.push_chunk(1, 4, &[3.0]).unwrap_err();
        assert!(err.message().contains("does not continue"));
    }

    #[test]
    fn new_position_must_start_at_offset_zero() {
        let mut a = acc();
        a.push_chunk(1, 0, &[1.0]).unwrap();
        let err = a.push_chunk(3, 5, &[2.0]).unwrap_err();
        assert!(err.message().contains("start at offset"));
    }

    #[test]
    fn oversized_single_chunk_is_rejected() {
        let mut a = acc();
        let big = vec![0.0f32; MAX_SCORE_VALUES_PER_POSITION + 1];
        let err = a.push_chunk(1, 0, &big).unwrap_err();
        assert!(err.message().contains("over the"));
    }

    #[test]
    fn oversized_row_built_from_chunks_is_rejected() {
        let mut a = acc();
        let chunk = vec![0.0f32; 1_000_000];
        a.push_chunk(1, 0, &chunk).unwrap();
        let err = a.push_chunk(1, 1_000_000, &chunk).unwrap_err();
        assert!(err.message().contains("exceeds"));
    }

    #[test]
    fn total_budget_is_enforced_across_positions() {
        let mut a = acc();
        let chunk = vec![0.0f32; 1_000_000];
        // Twelve full rows hit the 12M total budget.
        for pos in 0..13u64 {
            let offset = 0;
            let result = a.push_chunk(pos, offset, &chunk);
            if pos == 12 {
                assert!(
                    result.is_err(),
                    "13th million-value row must exceed the total bound"
                );
            } else {
                result.unwrap();
            }
        }
    }

    #[test]
    fn missing_position_fails_finish() {
        let mut a = acc();
        a.push_chunk(1, 0, &[1.0]).unwrap();
        let err = a.finish(1, &[1, 3]).unwrap_err();
        assert!(err.message().contains("expected"));
    }

    #[test]
    fn duplicate_or_out_of_order_final_row_fails_finish() {
        let mut a = acc();
        a.push_chunk(3, 0, &[1.0]).unwrap();
        a.push_chunk(1, 0, &[2.0]).unwrap();
        assert!(
            a.finish(2, &[1, 3]).is_err(),
            "order must match the request"
        );
    }

    #[test]
    fn truncated_row_fails_finish_when_positions_do_not_match_count() {
        // A missing FINAL chunk keeps the position present but short; the
        // count check here cannot see it (rows == positions), which is why
        // the caller validates exact row length against the vocabulary.
        let mut a = acc();
        a.push_chunk(1, 0, &[1.0]).unwrap();
        let rows = a.finish(1, &[1]).unwrap();
        assert_eq!(
            rows[0].1.len(),
            1,
            "short row survives finish; length is checked by the caller"
        );
    }

    #[test]
    fn empty_chunks_are_tolerated_but_do_not_complete_positions() {
        let mut a = acc();
        a.push_chunk(1, 0, &[]).unwrap();
        a.push_chunk(1, 0, &[1.0]).unwrap();
        let rows = a.finish(1, &[1]).unwrap();
        assert_eq!(rows[0].1, vec![1.0]);
    }

    #[test]
    fn scored_count_mismatch_fails_finish() {
        let mut a = acc();
        a.push_chunk(1, 0, &[1.0]).unwrap();
        assert!(a.finish(7, &[1]).is_err());
    }
}
