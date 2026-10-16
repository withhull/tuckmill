//! Bounded pending-output buffering with exact stop-sequence semantics.
//!
//! Contract (PRD §7, backend interface): a stop sequence or model-defined
//! EOS marker must never appear in user-visible output. A stop sequence can
//! span several generated tokens, so text is held back until it can no
//! longer become part of a stop sequence. The hold-back is bounded by the
//! longest stop sequence: memory never grows with generation length.
//!
//! Release is at byte granularity inside the held text: a token whose text
//! straddles a stop boundary is partially visible, never wholly dropped, so
//! the visible stream is identical no matter how a tokenizer splits the
//! stop string across tokens. Tokens whose bytes are still held when the
//! stream ends are flushed (EOS, length, or cancellation); tokens fully
//! inside a completed stop sequence are dropped and never counted visible.

use std::collections::VecDeque;

/// The result of feeding text to the buffer.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Release {
    /// Text that is now safe to publish (possibly empty).
    pub text: String,
    /// True when a stop sequence completed; `text` excludes it and the
    /// generator must finish with reason `stop`.
    pub stop: bool,
}

impl Release {
    fn empty() -> Self {
        Self::default()
    }
}

/// Bounded buffer between generated token text and user-visible output.
#[derive(Debug)]
pub struct VisibleOutputBuffer {
    /// Held text. Invariant: never ends with a proper prefix of a stop
    /// sequence (so a completing stop always starts inside this buffer).
    pending: String,
    /// Non-empty stop sequences in request order.
    stops: Vec<String>,
    /// (byte offset in `pending`, token ordinal) per token that contributed
    /// text now held. Bounded by the longest stop plus one token's text.
    marks: VecDeque<(usize, u64)>,
    /// Tokens with at least one visible byte. This is the honest
    /// completion-token count: tokens dropped inside a stop sequence or
    /// contributing nothing do not count.
    visible_tokens: u64,
}

impl VisibleOutputBuffer {
    pub fn new(stop: &[String]) -> Self {
        let stops: Vec<String> = stop.iter().filter(|s| !s.is_empty()).cloned().collect();
        Self {
            pending: String::new(),
            stops,
            marks: VecDeque::new(),
            visible_tokens: 0,
        }
    }

    /// Feed the decoded text of one generated token (already assembled as
    /// complete UTF-8 by the caller's stream decoder). `token` is the
    /// caller's monotonic token ordinal.
    pub fn push(&mut self, token: u64, text: &str) -> Release {
        if text.is_empty() {
            return Release::empty();
        }
        let start = self.pending.len();
        self.pending.push_str(text);
        self.marks.push_back((start, token));

        // A completed stop must end inside `pending` (see invariant).
        let mut matched: Option<(usize, usize)> = None; // (start, length)
        for s in &self.stops {
            if self.pending.ends_with(s.as_str()) {
                let st = self.pending.len() - s.len();
                let better = match matched {
                    None => true,
                    // Earliest occurrence wins; ties keep the longest.
                    Some((ms, ml)) => st < ms || (st == ms && s.len() > ml),
                };
                if better {
                    matched = Some((st, s.len()));
                }
            }
        }
        if let Some((start, _)) = matched {
            let visible: String = self.pending[..start].to_string();
            self.count_visible_before(start);
            self.pending.clear();
            // Remaining marks lie inside the dropped stop region.
            self.marks.clear();
            return Release {
                text: visible,
                stop: true,
            };
        }

        // Hold back the longest suffix that is a proper prefix of a stop.
        let keep = self.holdback_len();
        let visible_len = self.pending.len() - keep;
        if visible_len == 0 {
            return Release::empty();
        }
        let visible: String = self.pending[..visible_len].to_string();
        self.count_visible_before(visible_len);
        self.pending.drain(..visible_len);
        for (m, _) in self.marks.iter_mut() {
            *m -= visible_len;
        }
        Release {
            text: visible,
            stop: false,
        }
    }

    /// Flush everything still held (EOS, length, or cancellation). Held
    /// text was generated and is visible; no stop can complete any more.
    pub fn flush(&mut self) -> Release {
        let end = self.pending.len();
        let text = std::mem::take(&mut self.pending);
        self.count_visible_before(end);
        self.marks.clear();
        Release { text, stop: false }
    }

    /// Tokens with at least one visible byte so far.
    pub fn visible_tokens(&self) -> u64 {
        self.visible_tokens
    }

    /// Bytes currently held back (diagnostics/tests).
    pub fn held_bytes(&self) -> usize {
        self.pending.len()
    }

    /// A stop string can only complete if it begins at least one held byte;
    /// so a completing stop always starts inside P (see invariant).
    fn holdback_len(&self) -> usize {
        let mut keep = 0usize;
        let pending = self.pending.as_bytes();
        for s in &self.stops {
            let bytes = s.as_bytes();
            // Proper prefixes only: a full match would have been handled.
            for k in (1..bytes.len()).rev() {
                if k <= keep {
                    break;
                }
                if pending.ends_with(&bytes[..k]) {
                    keep = k;
                    break;
                }
            }
        }
        keep
    }

    /// Count tokens whose mark starts before `visible_end` as visible and
    /// drop their marks. Callers must have set `visible_end` to the length
    /// of the released region (before draining/mutating `pending`).
    fn count_visible_before(&mut self, visible_end: usize) {
        let mut count = 0u64;
        while let Some((m, _)) = self.marks.front() {
            if *m < visible_end {
                self.marks.pop_front();
                count += 1;
            } else {
                break;
            }
        }
        self.visible_tokens += count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn no_stops_releases_immediately() {
        let mut b = VisibleOutputBuffer::new(&[]);
        let r = b.push(0, "hello");
        assert_eq!(r.text, "hello");
        assert!(!r.stop);
        assert_eq!(b.held_bytes(), 0);
        assert_eq!(b.visible_tokens(), 1);
    }

    #[test]
    fn stop_spanning_tokens_never_leaks() {
        let mut b = VisibleOutputBuffer::new(&stops(&["<|im_end|>"]));
        let mut visible = String::new();
        let mut finished = false;
        for (i, t) in ["hello<|", "im_", "end", "|>", "more"].iter().enumerate() {
            let r = b.push(i as u64, t);
            visible.push_str(&r.text);
            if r.stop {
                finished = true;
                break;
            }
        }
        assert!(finished, "stop must fire once complete");
        assert_eq!(visible, "hello");
        assert!(
            !visible.contains("<|im_end|>"),
            "stop marker must not leak: {visible:?}"
        );
    }

    #[test]
    fn partial_token_keeps_its_visible_prefix() {
        // Token text "ab" then "cd" with stop "bcd": "a" is visible, the
        // rest of the stop region is dropped.
        let mut b = VisibleOutputBuffer::new(&stops(&["bcd"]));
        let r1 = b.push(0, "ab");
        assert_eq!(r1.text, "a");
        assert!(!r1.stop);
        let r2 = b.push(1, "cd");
        assert_eq!(r2.text, "");
        assert!(r2.stop);
        assert_eq!(
            b.visible_tokens(),
            1,
            "only the token with visible bytes counts"
        );
    }

    #[test]
    fn overlapping_stops_earliest_occurrence_wins() {
        let mut b = VisibleOutputBuffer::new(&stops(&["ab", "aab"]));
        let r = b.push(0, "aab");
        assert_eq!(r.text, "");
        assert!(r.stop, "earliest occurrence (aab at 0) must win");
        // And with the same text but stops that do not overlap at 0:
        let mut b = VisibleOutputBuffer::new(&stops(&["ab", "ba"]));
        let r = b.push(0, "xab");
        assert_eq!(r.text, "x");
        assert!(r.stop);
    }

    #[test]
    fn flush_releases_held_text() {
        let mut b = VisibleOutputBuffer::new(&stops(&["STOP"]));
        let r1 = b.push(0, "abcST");
        assert_eq!(r1.text, "abc", "only the potential-stop suffix is held");
        assert_eq!(b.held_bytes(), 2);
        let r2 = b.flush();
        assert_eq!(r2.text, "ST");
        assert_eq!(b.visible_tokens(), 1);
        assert_eq!(b.held_bytes(), 0);
    }

    #[test]
    fn holdback_is_bounded_by_the_longest_stop() {
        let mut b = VisibleOutputBuffer::new(&stops(&["abcdefgh"]));
        let mut rng = crate::sampling::Rng::new(1);
        for i in 0..1000u64 {
            // Single random letters never complete the stop.
            let c = (b'a' + (rng.next_u64() % 26) as u8) as char;
            let r = b.push(i, &c.to_string());
            assert!(!r.stop);
            assert!(
                b.held_bytes() <= 7,
                "held {} bytes after {i} pushes",
                b.held_bytes()
            );
        }
    }

    #[test]
    fn unicode_stop_boundaries_are_byte_exact() {
        let mut b = VisibleOutputBuffer::new(&stops(&["日本語"]));
        let r1 = b.push(0, "です日");
        assert_eq!(r1.text, "です");
        let r2 = b.push(1, "本語");
        assert_eq!(r2.text, "");
        assert!(r2.stop);
    }

    #[test]
    fn empty_stop_strings_are_ignored() {
        let mut b = VisibleOutputBuffer::new(&stops(&["", ""]));
        let r = b.push(0, "text");
        assert_eq!(r.text, "text");
    }
}
