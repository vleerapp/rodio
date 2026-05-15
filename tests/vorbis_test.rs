#![cfg(feature = "symphonia-vorbis")]

use rodio::{Decoder, Source};

#[test]
fn vorbis_decoder_not_exhausted_at_construction() {
    let file = std::fs::File::open("assets/music.ogg").unwrap();
    let decoder = Decoder::try_from(file).unwrap();

    assert!(
        !decoder.is_exhausted(),
        "decoder should not be exhausted immediately after construction; \
         current_span_len={:?}",
        decoder.current_span_len(),
    );
}
