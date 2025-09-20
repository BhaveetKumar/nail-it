use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};

pub struct BoundedQueue<T> {
    tx: Sender<T>,
    rx: Receiver<T>,
}

impl<T> BoundedQueue<T> {
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = bounded(capacity);
        Self { tx, rx }
    }

    pub fn try_produce(&self, item: T) -> Result<(), TrySendError<T>> {
        self.tx.try_send(item)
    }

    pub fn consume(&self) -> Option<T> {
        self.rx.try_recv().ok()
    }

    pub fn sender(&self) -> Sender<T> { self.tx.clone() }
    pub fn receiver(&self) -> Receiver<T> { self.rx.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backpressure_works() {
        let q = BoundedQueue::new(2);
        assert!(q.try_produce(1).is_ok());
        assert!(q.try_produce(2).is_ok());
        assert!(matches!(q.try_produce(3), Err(TrySendError::Full(3))));
        assert_eq!(q.consume(), Some(1));
        assert!(q.try_produce(3).is_ok());
    }
}
