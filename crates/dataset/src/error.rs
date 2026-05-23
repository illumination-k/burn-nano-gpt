use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Http(Box<ureq::Error>),

    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

impl From<ureq::Error> for Error {
    fn from(value: ureq::Error) -> Self {
        Self::Http(Box::new(value))
    }
}
