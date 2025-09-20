pub fn is_palindrome(s: &str) -> bool {
    let cleaned: String = s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect();
    cleaned.chars().eq(cleaned.chars().rev())
}

pub fn add(a: i64, b: i64) -> i64 { a + b }

pub fn sorted(mut v: Vec<i32>) -> Vec<i32> { v.sort(); v }

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        // Addition is commutative and doesn't overflow when constrained
        #[test]
        fn add_commutative(a in -1_000_000i64..=1_000_000, b in -1_000_000i64..=1_000_000) {
            prop_assert_eq!(add(a,b), add(b,a));
        }

        // Sorting returns a non-decreasing sequence and is idempotent
        #[test]
        fn sort_non_decreasing(data in proptest::collection::vec(-1000i32..=1000, 0..100)) {
            let s1 = sorted(data.clone());
            // non-decreasing
            prop_assert!(s1.windows(2).all(|w| w[0] <= w[1]));
            // idempotent
            let s2 = sorted(s1.clone());
            prop_assert_eq!(&s1, &s2);
            // same multiset length
            prop_assert_eq!(s1.len(), data.len());
        }

        // Palindrome check agrees with manual reverse for sanitized strings
        #[test]
        fn palindrome_symmetry(s in ".{0,64}") {
            let cleaned: String = s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect();
            let is_pal = cleaned.chars().eq(cleaned.chars().rev());
            prop_assert_eq!(is_palindrome(&s), is_pal);
        }
    }
}
