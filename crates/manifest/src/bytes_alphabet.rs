//! GPT-2-style byte↔unicode alphabet, shared by fixture generation and the
//! byte-level BPE tokenizer so both sides agree by construction.
//!
//! The mapping (as in GPT-2 and llama.cpp): printable byte ranges map to
//! themselves; every other byte maps to `U+0100 + byte`. This makes token
//! strings valid UTF-8 while remaining a bijection with the 256 byte values.

/// The 256-entry alphabet: `BYTES_TO_UNICODE[b]` is the character used for
/// byte `b`.
pub fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n: u32 = 0;
    for b in 0u16..=255 {
        let b8 = b as u8;
        let printable = (0x21..=0x7e).contains(&b8)
            || (0xa1..=0xac).contains(&b8)
            || (0xae..=0xff).contains(&b8);
        let mapped = if printable {
            b8 as u32
        } else {
            let c = 256 + n;
            n += 1;
            c
        };
        table[b as usize] = char::from_u32(mapped).expect("mapped code point is valid");
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn mapping_is_a_bijection() {
        let table = bytes_to_unicode();
        let unique: HashSet<char> = table.iter().copied().collect();
        assert_eq!(unique.len(), 256, "every byte maps to a distinct char");
        // Known anchors from the canonical table.
        assert_eq!(table[0x21], '!');
        assert_eq!(table[0x7e], '~');
        assert_eq!(table[b' ' as usize], 'Ġ');
        assert_eq!(table[b'\n' as usize], 'Ċ');
    }
}
