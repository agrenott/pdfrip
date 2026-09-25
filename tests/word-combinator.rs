use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use cli_interface::{arguments, entrypoint, Code};

fn temp_wordlist() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time should move forward")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "pdfrip-word-combinator-{}-{unique}.txt",
        std::process::id()
    ))
}

fn base_args(wordlist: String, min_words: usize, max_words: usize, case_mode: &str) -> arguments::Arguments {
    arguments::Arguments {
        number_of_threads: 2,
        batch_size: engine::default_batch_size(),
        filename: "crates/cracker/tests/fixtures/mask-upper-digit.pdf".to_string(),
        json: false,
        user_password_only: false,
        checkpoint: None,
        resume: None,
        subcommand: arguments::Method::WordCombinator(arguments::WordCombinatorArgs {
            wordlist,
            min_words,
            max_words,
            case_mode: case_mode.to_string(),
        }),
    }
}

#[test]
fn finds_password_from_uppercased_word_combination() {
    // The mask-upper-digit fixture is unlocked by the password "AB12". Providing the lowercase word
    // "ab12" and enabling case variants must let the combinator reach the uppercase form.
    let path = temp_wordlist();
    std::fs::write(&path, b"ab12\nbob\n").expect("wordlist should be writable");

    let args = base_args(path.display().to_string(), 1, 1, "lower-upper");
    let res = entrypoint(args).expect("An error occured when cracking file");

    std::fs::remove_file(&path).expect("temporary wordlist should be removable");

    assert!(matches!(res, Code::Success), "Failed cracking file.")
}
