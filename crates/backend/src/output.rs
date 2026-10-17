//! Bounded pending-output buffering with exact stop-sequence semantics.
//!
//! Contract (PRD §7, backend interface): a stop sequence or model-defined
//! EOS marker must never appear in user-visible output. A stop sequence can
//! span several generated tokens, and a token's text can contain trailing
//! characters after a completed stop — the buffer handles both: it searches
//! the held region plus each new piece for the earliest completed stop,
//! releases only the text before it, and discards the stop and any bytes
//! after it in that piece. The hold-back is bounded by the longest stop
//! sequence: memory never grows with generation length.
//!
//! Release is at byte granularity inside the held text: a token whose text
//! straddles a stop boundary is partially visible, never wholly dropped, so
//! the visible stream is identical no matter how a tokenizer splits the
//! stop string across tokens. Every pushed run carries the token owners of
//! its bytes, so a token that contributed at least one byte to visible
//! output is counted even when the bytes were split across tokens by
//! UTF-8 assembly. Each contributing token ordinal is counted **once for
//! the whole stream**, not once per decoded run: a token whose bytes span
//! several runs (the tail of a split character plus its own following
//! characters, or invalid-byte replacement runs) gets a single mark.
//! Tokens fully inside a completed stop sequence are dropped and never
//! counted visible.
//!
//! The deduplication is bounded by construction. Runs arrive in token
//! order, and a run can only carry owners within the last few tokens: a
//! decoded character spans at most four bytes, so it can be composed from
//! at most four token pieces and completes within three nonempty tokens of
//! its last byte's token. First-seen owners are therefore remembered in a
//! small window that evicts anything older than the newest owner minus
//! `OWNER_WINDOW`, and no owner can reappear after eviction.

use std::collections::VecDeque;

/// First-seen owner window: any character spans at most 4 tokens and
/// completes within 3 more, so an owner can never reappear in a run pushed
/// more than `OWNER_WINDOW` ordinals after it was first seen. The margin is
/// deliberately generous (the bound is 4).
const OWNER_WINDOW: u64 = 8;

/// The result of feeding text to the buffer.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Release {
    /// Text that is now safe to publish (possibly empty).
    pub text: String,
    /// True when a stop sequence completed; `text` excludes it (and any
    /// bytes that followed it in the same piece) and the generator must
    /// finish with reason `stop`.
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
    /// Held text. Invariant: never contains a complete stop sequence, and
    /// never ends with a proper prefix of a stop sequence (so a completing
    /// stop that starts before the newest piece cannot exist).
    pending: String,
    /// Non-empty stop sequences in request order.
    stops: Vec<String>,
    /// (byte offset in `pending`, token ordinal) per token-owner that
    /// contributed bytes now held. A byte run owned by several tokens
    /// (a multibyte character split across tokens) produces one mark per
    /// owner at the same offset; a token that appears in several runs gets
    /// exactly one mark for the whole stream. Bounded by the longest stop
    /// plus one push's text.
    marks: VecDeque<(usize, u64)>,
    /// Tokens with at least one visible byte. This is the honest
    /// completion-token count: tokens dropped inside a stop sequence or
    /// contributing nothing do not count.
    visible_tokens: u64,
    /// First-seen owner ordinals with a mark in `pending` or already
    /// counted; insertion order is nondecreasing. Evicted once
    /// `max_owner_seen - OWNER_WINDOW` passes them (they can no longer
    /// reappear).
    seen: VecDeque<u64>,
    /// Highest owner ordinal observed (None before any run).
    max_owner_seen: Option<u64>,
}

impl VisibleOutputBuffer {
    pub fn new(stop: &[String]) -> Self {
        let stops: Vec<String> = stop.iter().filter(|s| !s.is_empty()).cloned().collect();
        Self {
            pending: String::new(),
            stops,
            marks: VecDeque::new(),
            visible_tokens: 0,
            seen: VecDeque::new(),
            max_owner_seen: None,
        }
    }

    /// Feed one decoded run: `text` composed of bytes owned by `owners`
    /// (deduplicated token ordinals; more than one owner means a multibyte
    /// character was split across tokens). An owner that already
    /// contributed an earlier run keeps its single mark — distinct owners
    /// are counted once across the whole stream.
    pub fn push(&mut self, owners: &[u64], text: &str) -> Release {
        if text.is_empty() {
            return Release::empty();
        }
        if let Some(newest) = owners.iter().copied().max() {
            self.max_owner_seen = Some(self.max_owner_seen.map_or(newest, |m| m.max(newest)));
        }
        // Evict owners that can no longer reappear (see module docs).
        if let Some(max) = self.max_owner_seen {
            let floor = max.saturating_sub(OWNER_WINDOW);
            while self.seen.front().is_some_and(|o| *o < floor) {
                self.seen.pop_front();
            }
        }
        let start = self.pending.len();
        self.pending.push_str(text);
        for owner in owners {
            if !self.seen.contains(owner) {
                self.seen.push_back(*owner);
                self.marks.push_back((start, *owner));
            }
        }

        // Earliest completed stop inside the held region. An occurrence
        // cannot start in already-released text: the invariant keeps
        // visible output from ending in a proper stop prefix.
        let mut matched: Option<(usize, usize)> = None; // (start, stop length)
        for s in &self.stops {
            if let Some(at) = self.pending.find(s.as_str()) {
                let better = match matched {
                    None => true,
                    Some((ms, ml)) => at < ms || (at == ms && s.len() > ml),
                };
                if better {
                    matched = Some((at, s.len()));
                }
            }
        }
        if let Some((start, _)) = matched {
            // Release everything before the stop; discard the stop and
            // every byte that followed it in this piece.
            let visible: String = self.pending[..start].to_string();
            self.count_visible_before(start);
            self.pending.clear();
            // Remaining marks lie inside the discarded region.
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

    /// Longest suffix of the held text that is a proper prefix of a stop
    /// sequence — the only bytes that may still begin a completing stop.
    fn holdback_len(&self) -> usize {
        let mut keep = 0usize;
        let pending = self.pending.as_bytes();
        for s in &self.stops {
            let bytes = s.as_bytes();
            // Proper prefixes only: a full match was handled above.
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

    /// Count marks starting before `visible_end` as visible and drop them.
    /// Callers must pass the length of the released region before mutating
    /// `pending`.
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
        let r = b.push(&[0], "hello");
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
            let r = b.push(&[i as u64], t);
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
    fn stop_completed_inside_a_piece_discards_the_trailing_text() {
        // The counterexample from the Gate B review: the stop completes in
        // the middle of a single token's text, with extra characters after
        // it in the same piece.
        let mut b = VisibleOutputBuffer::new(&stops(&["STOP"]));
        let r = b.push(&[0], "helloSTOPworld");
        assert_eq!(r.text, "hello");
        assert!(r.stop);
        assert_eq!(b.visible_tokens(), 1);
    }

    #[test]
    fn stop_completed_across_pieces_with_trailing_text() {
        let mut b = VisibleOutputBuffer::new(&stops(&["STOP"]));
        let r1 = b.push(&[0], "helloST");
        assert_eq!(r1.text, "hello");
        assert!(!r1.stop);
        let r2 = b.push(&[1], "OPworld");
        assert_eq!(r2.text, "");
        assert!(r2.stop);
        assert_eq!(b.visible_tokens(), 1, "the piece after the stop is dropped");
        assert_eq!(b.held_bytes(), 0);
    }

    #[test]
    fn all_partitions_of_a_stop_inside_a_piece_are_equivalent() {
        // Exhaustive: every composition of "helloSTOPworld" into pieces must
        // produce the identical visible output "hello" with the stop fired
        // exactly once — independent of tokenizer boundaries.
        let text = b"helloSTOPworld";
        for mask in 0u32..(1 << (text.len() - 1)) {
            let mut pieces: Vec<&[u8]> = Vec::new();
            let mut start = 0usize;
            for i in 1..text.len() {
                if mask & (1 << (i - 1)) != 0 {
                    pieces.push(&text[start..i]);
                    start = i;
                }
            }
            pieces.push(&text[start..]);

            let mut b = VisibleOutputBuffer::new(&stops(&["STOP"]));
            let mut visible = String::new();
            let mut fired = 0usize;
            for (i, piece) in pieces.iter().enumerate() {
                let r = b.push(&[i as u64], std::str::from_utf8(piece).unwrap());
                visible.push_str(&r.text);
                if r.stop {
                    fired += 1;
                    break;
                }
                assert!(
                    !visible.contains("STOP"),
                    "mask {mask:#b}: marker leaked mid-stream: {visible:?}"
                );
            }
            assert_eq!(fired, 1, "mask {mask:#b}: stop must fire once");
            assert_eq!(visible, "hello", "mask {mask:#b}");
        }
    }

    #[test]
    fn partial_token_keeps_its_visible_prefix() {
        // Token text "ab" then "cd" with stop "bcd": "a" is visible, the
        // rest of the stop region is dropped.
        let mut b = VisibleOutputBuffer::new(&stops(&["bcd"]));
        let r1 = b.push(&[0], "ab");
        assert_eq!(r1.text, "a");
        assert!(!r1.stop);
        let r2 = b.push(&[1], "cd");
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
        let r = b.push(&[0], "aab");
        assert_eq!(r.text, "");
        assert!(r.stop, "earliest occurrence (aab at 0) must win");
        // And with the same text but stops that first overlap later:
        let mut b = VisibleOutputBuffer::new(&stops(&["ab", "ba"]));
        let r = b.push(&[0], "xab");
        assert_eq!(r.text, "x");
        assert!(r.stop);
    }

    #[test]
    fn flush_releases_held_text() {
        let mut b = VisibleOutputBuffer::new(&stops(&["STOP"]));
        let r1 = b.push(&[0], "abcST");
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
            let r = b.push(&[i], &c.to_string());
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
        let r1 = b.push(&[0], "です日");
        assert_eq!(r1.text, "です");
        let r2 = b.push(&[1], "本語");
        assert_eq!(r2.text, "");
        assert!(r2.stop);
    }

    #[test]
    fn empty_stop_strings_are_ignored() {
        let mut b = VisibleOutputBuffer::new(&stops(&["", ""]));
        let r = b.push(&[0], "text");
        assert_eq!(r.text, "text");
    }

    /// Compose the real decoder with the buffer: every contributing token
    /// ordinal counts exactly once across all decoded runs.
    mod composition {
        use super::*;
        use tuckmill_manifest::utf8::Utf8StreamDecoder;

        fn feed(
            decoder: &mut Utf8StreamDecoder,
            b: &mut VisibleOutputBuffer,
            bytes: &[u8],
            token: u64,
        ) {
            for run in decoder.push_token(bytes, token) {
                let r = b.push(&run.owners, &run.text);
                assert!(!r.stop);
            }
        }

        #[test]
        fn split_character_and_ascii_run_count_two_not_three() {
            // Review counterexample: token 0 starts €, token 1 completes it
            // and adds x. Runs: [0,1] -> "€", [1] -> "x"; distinct
            // contributors are two, not three.
            let mut d = Utf8StreamDecoder::new();
            let mut b = VisibleOutputBuffer::new(&[]);
            feed(&mut d, &mut b, &[0xE2], 0);
            feed(&mut d, &mut b, &[0x82, 0xAC, b'x'], 1);
            assert_eq!(b.visible_tokens(), 2);
            assert_eq!(b.held_bytes(), 0);
        }

        #[test]
        fn ascii_then_split_character_counts_two() {
            // The reverse direction: token 0 contributes "x" and starts €,
            // token 1 completes €.
            let mut d = Utf8StreamDecoder::new();
            let mut b = VisibleOutputBuffer::new(&[]);
            feed(&mut d, &mut b, b"x\xE2", 0);
            feed(&mut d, &mut b, &[0x82, 0xAC], 1);
            assert_eq!(b.visible_tokens(), 2);
        }

        #[test]
        fn one_token_split_into_several_runs_counts_once() {
            // Invalid-byte replacement: token 0 alone produces U+FFFD and
            // then "A" (two runs, same owner).
            let mut d = Utf8StreamDecoder::new();
            let mut b = VisibleOutputBuffer::new(&[]);
            feed(&mut d, &mut b, &[0xF0, 0x9F, 0x41], 0);
            assert_eq!(b.visible_tokens(), 1);
        }

        #[test]
        fn replacement_and_continuation_tokens_count_once_each() {
            let mut d = Utf8StreamDecoder::new();
            let mut b = VisibleOutputBuffer::new(&[]);
            feed(&mut d, &mut b, &[0xF0, 0x9F], 0);
            feed(&mut d, &mut b, &[0x41], 1);
            assert_eq!(b.visible_tokens(), 2);
        }

        #[test]
        fn dropped_stop_region_never_counts_its_contributors() {
            // A multi-owner run that is dropped by a stop must not count
            // any of its owners, while earlier visible text still does.
            let mut b = VisibleOutputBuffer::new(&stops(&["€"]));
            let r1 = b.push(&[0], "ok");
            assert_eq!(r1.text, "ok");
            let r2 = b.push(&[1, 2], "€");
            assert_eq!(r2.text, "");
            assert!(r2.stop);
            assert_eq!(
                b.visible_tokens(),
                1,
                "the dropped € contributors (tokens 1 and 2) never count"
            );
        }
    }

    #[test]
    fn multi_owner_runs_count_every_contributing_token() {
        // A multibyte character split across three tokens: the run is owned
        // by all three; when visible, all three count; when dropped inside a
        // stop, none do.
        let mut b = VisibleOutputBuffer::new(&[]);
        let r = b.push(&[4, 5, 6], "€");
        assert_eq!(r.text, "€");
        assert_eq!(b.visible_tokens(), 3);

        let mut b = VisibleOutputBuffer::new(&stops(&["€"]));
        let r = b.push(&[7, 8, 9], "€");
        assert!(r.stop);
        assert_eq!(b.visible_tokens(), 0, "a dropped run is not visible");
    }
}
