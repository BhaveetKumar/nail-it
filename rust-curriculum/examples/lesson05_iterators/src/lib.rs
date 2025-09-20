use itertools::Itertools;
use std::collections::HashMap;

pub fn top_k(s: &str, k: usize) -> Vec<(String, usize)> {
    let mut map: HashMap<String, usize> = HashMap::new();
    for w in s.split_whitespace() {
        *map.entry(w.to_lowercase()).or_insert(0) += 1;
    }
    map.into_iter()
        .sorted_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)))
        .take(k)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic() {
        let v = top_k("a a b c c c", 2);
        assert_eq!(v, vec![("c".into(), 3), ("a".into(), 2)]);
    }
}
