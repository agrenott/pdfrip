use std::{fs, sync::Arc};

use super::Producer;

/// Case transforms applied to each word slot in a combination.
///
/// The variants are enumerated in a fixed, deterministic order so the resulting keyspace stays
/// stable across runs, which is required for exact progress accounting and checkpoint/resume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseVariant {
    /// The word exactly as it appears in the wordlist.
    Original,
    /// Every ASCII letter lowercased.
    Lower,
    /// Every ASCII letter uppercased.
    Upper,
    /// First ASCII letter uppercased, remaining ASCII letters lowercased.
    Capitalized,
}

impl CaseVariant {
    /// Applies the case transform to `word`, appending the result into `output`.
    fn apply_into(self, word: &[u8], output: &mut Vec<u8>) {
        match self {
            Self::Original => output.extend_from_slice(word),
            Self::Lower => output.extend(word.iter().map(|byte| byte.to_ascii_lowercase())),
            Self::Upper => output.extend(word.iter().map(|byte| byte.to_ascii_uppercase())),
            Self::Capitalized => {
                let mut seen_letter = false;
                for byte in word {
                    if byte.is_ascii_alphabetic() {
                        if seen_letter {
                            output.push(byte.to_ascii_lowercase());
                        } else {
                            output.push(byte.to_ascii_uppercase());
                            seen_letter = true;
                        }
                    } else {
                        output.push(*byte);
                    }
                }
            }
        }
    }
}

/// Which set of case transforms to apply to each word slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseMode {
    /// Only the word as-is: `[Original]`.
    None,
    /// Lower and upper: `[Lower, Upper]`.
    LowerUpper,
    /// Lower, upper, and capitalized: `[Lower, Upper, Capitalized]`.
    All,
}

impl CaseMode {
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "none" => Ok(Self::None),
            "lower-upper" | "lowerupper" => Ok(Self::LowerUpper),
            "all" => Ok(Self::All),
            _ => Err(format!(
                "unsupported case mode '{name}'; expected one of none, lower-upper, all"
            )),
        }
    }

    fn variants(self) -> &'static [CaseVariant] {
        match self {
            Self::None => &[CaseVariant::Original],
            Self::LowerUpper => &[CaseVariant::Lower, CaseVariant::Upper],
            Self::All => &[
                CaseVariant::Lower,
                CaseVariant::Upper,
                CaseVariant::Capitalized,
            ],
        }
    }
}

/// Generates candidates by concatenating between `min_words` and `max_words` wordlist entries,
/// applying a configurable set of case transforms to every word slot.
///
/// Each slot selects one `(word, case_variant)` pair, so a combination of exactly `k` slots has
/// `base^k` candidates where `base = words * case_variants`. The producer enumerates the whole span
/// `min_words..=max_words` in a stable mixed-radix order. The keyspace is deterministic, finite,
/// exactly countable, and suitable for checkpoint/resume.
#[derive(Clone)]
pub struct WordCombinatorProducer {
    words: Arc<Vec<Vec<u8>>>,
    variants: &'static [CaseVariant],
    min_words: usize,
    /// `base = words.len() * variants.len()`.
    base: usize,
    /// Precomputed candidate count for each combination length, indexed by `length - min_words`.
    counts_per_length: Arc<[usize]>,
    size: usize,
    position: usize,
}

impl WordCombinatorProducer {
    pub fn new(path: &str, min_words: usize, max_words: usize, case_mode: CaseMode) -> Self {
        Self::try_new(path, min_words, max_words, case_mode)
            .expect("word-combinator configuration should be valid")
    }

    pub fn try_new(
        path: &str,
        min_words: usize,
        max_words: usize,
        case_mode: CaseMode,
    ) -> Result<Self, String> {
        if min_words == 0 {
            return Err(String::from(
                "word-combinator requires min-words to be at least 1",
            ));
        }
        if min_words > max_words {
            return Err(format!(
                "minimum word count ({min_words}) must not exceed maximum word count ({max_words})"
            ));
        }

        let bytes = fs::read(path)
            .map_err(|err| format!("Unable to read wordlist file '{path}': {err}"))?;
        let words = Arc::new(parse_words(&bytes));
        if words.is_empty() {
            return Err(String::from(
                "word-combinator requires at least one non-empty word in the supplied file",
            ));
        }

        let variants = case_mode.variants();
        let base = words
            .len()
            .checked_mul(variants.len())
            .ok_or_else(|| String::from("word-combinator base is too large to count exactly"))?;

        let mut counts_per_length = Vec::with_capacity(max_words - min_words + 1);
        let mut size = 0usize;
        for length in min_words..=max_words {
            let count = base.checked_pow(length as u32).ok_or_else(|| {
                String::from("word-combinator search space is too large to count exactly")
            })?;
            counts_per_length.push(count);
            size = size.checked_add(count).ok_or_else(|| {
                String::from("word-combinator search space is too large to count exactly")
            })?;
        }

        Ok(Self {
            words,
            variants,
            min_words,
            base,
            counts_per_length: Arc::from(counts_per_length),
            size,
            position: 0,
        })
    }

    /// Renders the candidate at the current position into `output`.
    ///
    /// Decodes `position` into a combination length and a mixed-radix slot index. Slots are decoded
    /// most-significant first so incrementing `position` walks the last slot fastest, giving a
    /// stable odometer-style ordering.
    ///
    /// Returns an error if the current position falls outside the counted keyspace. Callers guard
    /// against exhaustion before invoking this method, so an out-of-range offset indicates an
    /// internal accounting bug rather than normal termination.
    fn render_position_into(&self, output: &mut Vec<u8>) -> Result<(), String> {
        let mut offset = self.position;
        output.clear();

        for (index, &count) in self.counts_per_length.iter().enumerate() {
            if offset >= count {
                offset -= count;
                continue;
            }

            let length = self.min_words + index;
            // Decode `offset` as a `length`-digit base-`self.base` number, MSB first.
            let mut divisor = count / self.base.max(1);
            for _ in 0..length {
                let slot = if divisor == 0 { offset } else { offset / divisor };
                let word_index = slot / self.variants.len();
                let variant = self.variants[slot % self.variants.len()];
                variant.apply_into(&self.words[word_index], output);

                if divisor == 0 {
                    offset = 0;
                } else {
                    offset %= divisor;
                    divisor /= self.base;
                }
            }
            return Ok(());
        }

        Err(String::from(
            "word-combinator offset exceeded the available search space",
        ))
    }
}

impl Producer for WordCombinatorProducer {
    fn next(&mut self) -> Result<Option<Vec<u8>>, String> {
        if self.position >= self.size {
            return Ok(None);
        }

        let mut candidate = Vec::new();
        let produced = self.next_into(&mut candidate)?;
        debug_assert!(produced, "position checked before calling next_into");
        Ok(Some(candidate))
    }

    fn next_into(&mut self, output: &mut Vec<u8>) -> Result<bool, String> {
        if self.position >= self.size {
            output.clear();
            return Ok(false);
        }

        self.render_position_into(output)?;
        self.position += 1;
        Ok(true)
    }

    fn size(&self) -> usize {
        self.size
    }

    fn skip(&mut self, count: usize) -> Result<usize, String> {
        let remaining = self.size.saturating_sub(self.position);
        let skipped = count.min(remaining);
        self.position += skipped;
        Ok(skipped)
    }

    fn boxed_clone(&self) -> Option<Box<dyn Producer>> {
        Some(Box::new(self.clone()))
    }
}

fn parse_words(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut words = Vec::new();
    let mut start = 0usize;

    for (index, byte) in bytes.iter().enumerate() {
        if *byte != b'\n' {
            continue;
        }

        let mut end = index;
        if end > start && bytes[end - 1] == b'\r' {
            end -= 1;
        }
        if start < end {
            words.push(bytes[start..end].to_vec());
        }
        start = index + 1;
    }

    if start < bytes.len() {
        let mut end = bytes.len();
        if end > start && bytes[end - 1] == b'\r' {
            end -= 1;
        }
        if start < end {
            words.push(bytes[start..end].to_vec());
        }
    }

    words
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::Producer;

    use super::{CaseMode, CaseVariant, WordCombinatorProducer};

    fn temp_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        std::env::temp_dir().join(format!("pdfrip-{name}-{}-{unique}.txt", std::process::id()))
    }

    fn drain(producer: &mut dyn Producer) -> Vec<Vec<u8>> {
        let mut values = Vec::new();
        while let Some(value) = producer.next().unwrap() {
            values.push(value);
        }
        values
    }

    fn drain_into(producer: &mut dyn Producer) -> Vec<Vec<u8>> {
        let mut values = Vec::new();
        let mut candidate = Vec::new();
        while producer.next_into(&mut candidate).unwrap() {
            values.push(candidate.clone());
        }
        values
    }

    #[test]
    fn case_variants_transform_ascii_letters() {
        let mut out = Vec::new();
        CaseVariant::Original.apply_into(b"aBc1", &mut out);
        assert_eq!(out, b"aBc1");

        out.clear();
        CaseVariant::Lower.apply_into(b"aBc1", &mut out);
        assert_eq!(out, b"abc1");

        out.clear();
        CaseVariant::Upper.apply_into(b"aBc1", &mut out);
        assert_eq!(out, b"ABC1");

        out.clear();
        CaseVariant::Capitalized.apply_into(b"aBc1d", &mut out);
        assert_eq!(out, b"Abc1d");

        // Leading non-letters do not consume the first-letter uppercase slot.
        out.clear();
        CaseVariant::Capitalized.apply_into(b"1ab", &mut out);
        assert_eq!(out, b"1Ab");
    }

    #[test]
    fn single_word_no_case_matches_wordlist_order() {
        let path = temp_path("wc-single");
        std::fs::write(&path, b"foo\nbar\n").expect("wordlist should be writable");

        let mut producer = WordCombinatorProducer::try_new(
            path.to_str().unwrap(),
            1,
            1,
            CaseMode::None,
        )
        .expect("producer should build");

        assert_eq!(producer.size(), 2);
        assert_eq!(drain(&mut producer), vec![b"foo".to_vec(), b"bar".to_vec()]);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn concatenates_two_words_with_lower_upper_variants() {
        let path = temp_path("wc-two");
        std::fs::write(&path, b"ab\n").expect("wordlist should be writable");

        // 1 word * 2 case variants = base 2. length-2 => 2^2 = 4 candidates.
        let mut producer = WordCombinatorProducer::try_new(
            path.to_str().unwrap(),
            2,
            2,
            CaseMode::LowerUpper,
        )
        .expect("producer should build");

        assert_eq!(producer.size(), 4);
        // Odometer order: last slot varies fastest. Variants order: [Lower, Upper].
        assert_eq!(
            drain(&mut producer),
            vec![
                b"abab".to_vec(),
                b"abAB".to_vec(),
                b"ABab".to_vec(),
                b"ABAB".to_vec(),
            ]
        );

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn spans_min_to_max_word_counts() {
        let path = temp_path("wc-span");
        std::fs::write(&path, b"x\ny\n").expect("wordlist should be writable");

        // 2 words, case none => base 2. lengths 1..=2 => 2 + 4 = 6.
        let mut producer = WordCombinatorProducer::try_new(
            path.to_str().unwrap(),
            1,
            2,
            CaseMode::None,
        )
        .expect("producer should build");

        assert_eq!(producer.size(), 6);
        assert_eq!(
            drain(&mut producer),
            vec![
                b"x".to_vec(),
                b"y".to_vec(),
                b"xx".to_vec(),
                b"xy".to_vec(),
                b"yx".to_vec(),
                b"yy".to_vec(),
            ]
        );

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn skip_is_consistent_with_iteration() {
        let path = temp_path("wc-skip");
        std::fs::write(&path, b"a\nb\n").expect("wordlist should be writable");

        let expected = {
            let mut producer =
                WordCombinatorProducer::try_new(path.to_str().unwrap(), 1, 3, CaseMode::All)
                    .expect("producer should build");
            drain(&mut producer)
        };

        let mut producer =
            WordCombinatorProducer::try_new(path.to_str().unwrap(), 1, 3, CaseMode::All)
                .expect("producer should build");
        assert_eq!(producer.skip(5).unwrap(), 5);
        assert_eq!(producer.next().unwrap().unwrap(), expected[5]);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn next_into_matches_next_ordering() {
        let path = temp_path("wc-next-into");
        std::fs::write(&path, b"cat\nDog\n").expect("wordlist should be writable");

        let expected = {
            let mut producer =
                WordCombinatorProducer::try_new(path.to_str().unwrap(), 1, 2, CaseMode::All)
                    .expect("producer should build");
            drain(&mut producer)
        };

        let mut producer =
            WordCombinatorProducer::try_new(path.to_str().unwrap(), 1, 2, CaseMode::All)
                .expect("producer should build");
        assert_eq!(drain_into(&mut producer), expected);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn rejects_invalid_configuration() {
        let path = temp_path("wc-invalid");
        std::fs::write(&path, b"word\n").expect("wordlist should be writable");
        let p = path.to_str().unwrap();

        assert!(WordCombinatorProducer::try_new(p, 0, 2, CaseMode::None).is_err());
        assert!(WordCombinatorProducer::try_new(p, 3, 2, CaseMode::None).is_err());

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn errors_when_position_exceeds_search_space() {
        let path = temp_path("wc-out-of-range");
        std::fs::write(&path, b"a\nb\n").expect("wordlist should be writable");

        let mut producer =
            WordCombinatorProducer::try_new(path.to_str().unwrap(), 1, 2, CaseMode::None)
                .expect("producer should build");

        // Force the internal cursor past the counted keyspace, bypassing the exhaustion guard, to
        // prove render_position_into fails loudly instead of emitting a bogus empty candidate.
        producer.position = producer.size + 1;
        let mut output = vec![0xAA];
        let err = producer
            .render_position_into(&mut output)
            .expect_err("out-of-range offset should error");
        assert!(err.contains("exceeded the available search space"));

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn rejects_empty_wordlists() {
        let path = temp_path("wc-empty");
        std::fs::write(&path, b"\n\r\n").expect("wordlist should be writable");

        assert!(
            WordCombinatorProducer::try_new(path.to_str().unwrap(), 1, 2, CaseMode::None).is_err()
        );

        std::fs::remove_file(&path).unwrap();
    }
}
