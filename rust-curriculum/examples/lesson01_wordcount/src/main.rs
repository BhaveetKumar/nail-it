use std::collections::HashMap;
use std::io::{self, Read};

fn wordcount(s: &str) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for w in s.split_whitespace() {
        *map.entry(w.to_lowercase()).or_insert(0) += 1;
    }
    map
}

fn main() -> anyhow::Result<()> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let counts = wordcount(&input);
    for (w, c) in counts {
        println!("{w} {c}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counts_basic() {
        let m = wordcount("a a b");
        assert_eq!(m.get("a"), Some(&2));
        assert_eq!(m.get("b"), Some(&1));
    }
}
