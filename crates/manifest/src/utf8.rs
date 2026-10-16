//! Stateful incremental UTF-8 decoding for streaming text.
//!
//! Multibyte characters split across committed token pieces are held until
//! complete instead of becoming replacement characters; only complete
//! characters are ever emitted. Invalid sequences become U+FFFD exactly once;
//! an unfinished tail at stream end is dropped rather than fabricated.

/// Incremental UTF-8 assembler.
#[derive(Debug, Default)]
pub struct Utf8StreamDecoder {
    pending: Vec<u8>,
}

impl Utf8StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed piece bytes; returns the longest complete-UTF-8 prefix. Invalid
    /// sequences become U+FFFD; an incomplete trailing sequence is retained
    /// for the next call.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        out.push_str(
                            std::str::from_utf8(&self.pending[..valid])
                                .expect("valid_up_to guarantees a valid prefix"),
                        );
                    }
                    match e.error_len() {
                        Some(bad) => {
                            // Definitive invalid sequence: one replacement
                            // char, then continue after it.
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid + bad);
                            if self.pending.is_empty() {
                                return out;
                            }
                        }
                        None => {
                            // Incomplete tail: keep for the next push.
                            self.pending.drain(..valid);
                            return out;
                        }
                    }
                }
            }
        }
    }

    /// True when bytes are buffered awaiting a continuation piece.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
