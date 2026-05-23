#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("character not in vocabulary: {0:?}")]
    UnknownChar(char),

    #[error("token id out of vocabulary: {0}")]
    UnknownToken(u32),
}
