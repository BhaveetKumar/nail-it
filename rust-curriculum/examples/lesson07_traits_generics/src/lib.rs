pub trait Storage {
    type Key: Ord;
    type Value;
    fn put(&mut self, k: Self::Key, v: Self::Value);
    fn get(&self, k: &Self::Key) -> Option<&Self::Value>;
}

pub struct MapStore<K: Ord, V> {
    pub(crate) inner: std::collections::BTreeMap<K, V>,
}
impl<K: Ord, V> Default for MapStore<K, V> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<K: Ord, V> Storage for MapStore<K, V> {
    type Key = K;
    type Value = V;
    fn put(&mut self, k: K, v: V) {
        self.inner.insert(k, v);
    }
    fn get(&self, k: &K) -> Option<&V> {
        self.inner.get(k)
    }
}

pub fn min_by_key<T, F, K>(slice: &[T], f: F) -> Option<&T>
where
    F: Fn(&T) -> K,
    K: Ord,
{
    let mut it = slice.iter();
    let first = it.next()?;
    let mut best = first;
    let mut best_k = f(best);
    for item in it {
        let k = f(item);
        if k < best_k {
            best = item;
            best_k = k;
        }
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn storage_roundtrip() {
        let mut s: MapStore<i32, &str> = MapStore::default();
        s.put(2, "b");
        s.put(1, "a");
        assert_eq!(s.get(&1), Some(&"a"));
    }
    #[test]
    fn min_key() {
        let a = [3, 1, 2];
        assert_eq!(min_by_key(&a, |x| *x), Some(&1));
    }
}
