pub struct RingBuffer<T> {
    buf: Vec<Option<T>>,
    head: usize,
    tail: usize,
    len: usize,
}

impl<T> RingBuffer<T> {
    pub fn with_capacity(cap: usize) -> Self {
        assert!(cap > 0);
        let buf: Vec<Option<T>> = (0..cap).map(|_| None).collect();
        Self { buf, head: 0, tail: 0, len: 0 }
    }

    pub fn capacity(&self) -> usize { self.buf.len() }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn is_full(&self) -> bool { self.len == self.capacity() }

    pub fn push(&mut self, item: T) -> Result<(), T> {
        if self.is_full() { return Err(item); }
        self.buf[self.tail] = Some(item);
        self.tail = (self.tail + 1) % self.capacity();
        self.len += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() { return None; }
        let v = self.buf[self.head].take();
        self.head = (self.head + 1) % self.capacity();
        self.len -= 1;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let mut rb = RingBuffer::with_capacity(2);
        assert!(rb.push(1).is_ok());
        assert!(rb.push(2).is_ok());
        assert!(rb.push(3).is_err());
        assert_eq!(rb.pop(), Some(1));
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), None);
    }
}
