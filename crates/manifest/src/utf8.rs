//! Stateful incremental UTF-8 decoding for streaming text, with token byte
//! provenance.
//!
//! Multibyte characters split across committed token pieces are held until
//! complete instead of becoming replacement characters; only complete
//! characters are ever emitted. Invalid sequences become U+FFFD exactly once;
//! an unfinished tail at stream end is dropped rather than fabricated.
//!
//! Every decoded run carries the byte-level token owners so accounting that
//! follows ("a token with at least one visible byte counts") stays exact
//! even when a character was assembled from several tokens: the owners of
//! all bytes of a completed character are attributed to that character, not
//! just the token that happened to complete it.

/// A decoded text run with the token owners of every byte that composes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedRun {
    /// Deduplicated token ordinals in first-seen byte order.
    pub owners: Vec<u64>,
    pub text: String,
}

/// Incremental UTF-8 assembler.
#[derive(Debug, Default)]
pub struct Utf8StreamDecoder {
    /// Buffered bytes with their owning token, awaiting completion.
    pending: Vec<(u8, u64)>,
}

impl Utf8StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one token's piece bytes; returns the complete runs produced.
    /// Invalid sequences become U+FFFD (owned by the bytes that formed
    /// them); an incomplete trailing sequence is retained for later.
    pub fn push_token(&mut self, bytes: &[u8], token: u64) -> Vec<DecodedRun> {
        self.pending.extend(bytes.iter().map(|b| (*b, token)));
        let mut runs: Vec<DecodedRun> = Vec::new();
        loop {
            let plain: Vec<u8> = self.pending.iter().map(|(b, _)| *b).collect();
            if plain.is_empty() {
                return runs;
            }
            match std::str::from_utf8(&plain) {
                Ok(s) => {
                    append_run(&mut runs, self.pending.iter().map(|(_, t)| *t).collect(), s);
                    self.pending.clear();
                    return runs;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        let s = std::str::from_utf8(&plain[..valid])
                            .expect("valid_up_to guarantees a valid prefix");
                        append_run(
                            &mut runs,
                            self.pending[..valid].iter().map(|(_, t)| *t).collect(),
                            s,
                        );
                    }
                    match e.error_len() {
                        Some(bad) => {
                            // Definitive invalid sequence: one replacement
                            // char owned by the bytes that formed it, then
                            // continue after it. (The replacement is not a
                            // byte-for-byte image of the invalid bytes, so
                            // ownership is attached directly.)
                            let mut set: Vec<u64> = Vec::new();
                            for (_, token) in &self.pending[valid..valid + bad] {
                                if !set.contains(token) {
                                    set.push(*token);
                                }
                            }
                            push_group(&mut runs, set, "\u{FFFD}");
                            self.pending.drain(..valid + bad);
                        }
                        None => {
                            // Incomplete tail: keep for the next push.
                            self.pending.drain(..valid);
                            return runs;
                        }
                    }
                }
            }
        }
    }

    /// Convenience wrapper without provenance (all output owned by token 0).
    /// Used by tests and call sites that do not track token owners.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.push_token(bytes, 0)
            .into_iter()
            .map(|r| r.text)
            .collect()
    }

    /// True when bytes are buffered awaiting a continuation piece.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
}

/// Group a decoded text into runs of characters sharing the same owner set.
/// `owners` covers exactly the bytes of `text`, one entry per byte.
fn append_run(runs: &mut Vec<DecodedRun>, owners: Vec<u64>, text: &str) {
    let mut offset = 0usize;
    for ch in text.chars() {
        let len = ch.len_utf8();
        let mut set: Vec<u64> = Vec::new();
        for owner in &owners[offset..offset + len] {
            if !set.contains(owner) {
                set.push(*owner);
            }
        }
        push_group(runs, set, &ch.to_string());
        offset += len;
    }
}

/// Append text with a fixed owner set, merging with the previous run when
/// the owner set matches.
fn push_group(runs: &mut Vec<DecodedRun>, set: Vec<u64>, text: &str) {
    match runs.last_mut() {
        Some(last) if last.owners == set => last.text.push_str(text),
        _ => runs.push(DecodedRun {
            owners: set,
            text: text.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owners(runs: &[DecodedRun]) -> Vec<(Vec<u64>, &str)> {
        runs.iter()
            .map(|r| (r.owners.clone(), r.text.as_str()))
            .collect()
    }

    #[test]
    fn reassembles_split_multibyte_characters() {
        let (a, b) = ([0xF0u8, 0x9F], [0x98u8, 0x80]);
        let mut d = Utf8StreamDecoder::new();
        assert_eq!(d.push(&a), "");
        assert!(d.has_pending());
        assert_eq!(d.push(&b), "😀");
        assert!(!d.has_pending());
        assert_eq!(d.push("tail".as_bytes()), "tail");
    }

    #[test]
    fn handles_invalid_and_complete_input() {
        let mut d = Utf8StreamDecoder::new();
        assert_eq!(d.push(b"ok"), "ok");
        assert_eq!(d.push(&[0xFF]), "\u{FFFD}");
        assert_eq!(d.push(b"rest"), "rest");
        assert_eq!(d.push("héllo".as_bytes()), "héllo");
    }

    #[test]
    fn bytes_after_invalid_sequence_still_decode() {
        let mut d = Utf8StreamDecoder::new();
        // Invalid 0x80 in the middle, then a split emoji across pushes.
        assert_eq!(d.push(&[0x80, 0xF0]), "\u{FFFD}");
        assert_eq!(d.push(&[0x9F, 0x98, 0x80]), "😀");
    }

    #[test]
    fn split_character_attributes_all_contributing_tokens() {
        // The review's counterexample: bytes E2 82 AC (€) across three
        // tokens. The completed run must be owned by all three tokens.
        let mut d = Utf8StreamDecoder::new();
        assert!(d.push_token(&[0xE2], 1).is_empty());
        assert!(d.push_token(&[0x82], 2).is_empty());
        let runs = d.push_token(&[0xAC], 3);
        assert_eq!(owners(&runs), vec![(vec![1, 2, 3], "€")]);
    }

    #[test]
    fn mixed_ascii_and_split_character_runs() {
        let mut d = Utf8StreamDecoder::new();
        let runs = d.push_token("aé".as_bytes(), 1); // complete in one token
        assert_eq!(owners(&runs), vec![(vec![1], "aé")]);
        assert!(d.push_token(&[0xE2], 2).is_empty());
        let runs = d.push_token(&[0x82, 0xAC, b'b'], 3);
        // "€" is owned by tokens 2 and 3; "b" only by token 3.
        assert_eq!(owners(&runs), vec![(vec![2, 3], "€"), (vec![3], "b")]);
    }

    #[test]
    fn replacement_character_is_owned_by_its_bytes_tokens() {
        let mut d = Utf8StreamDecoder::new();
        let runs = d.push_token(&[0xF0, 0x9F], 5);
        assert!(runs.is_empty());
        // The 4-byte lead + continuation are broken by 0x41: the invalid
        // pair (owned by token 5) becomes one U+FFFD, and 0x41 resynchronizes
        // as its own character (owned by token 6).
        let runs = d.push_token(&[0x41], 6);
        assert_eq!(owners(&runs), vec![(vec![5], "\u{FFFD}"), (vec![6], "A")]);
        assert_eq!(d.push(b"x"), "x".to_string());
    }
}
