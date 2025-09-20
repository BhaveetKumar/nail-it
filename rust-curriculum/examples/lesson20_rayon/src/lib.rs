use rayon::prelude::*;

pub fn parallel_sum(v: &[i64]) -> i64 {
    v.par_iter().cloned().sum()
}

pub fn parallel_sort(mut v: Vec<i32>) -> Vec<i32> {
    v.par_sort();
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn sums_matches_sequential() {
        let mut rng = rand::thread_rng();
        let data: Vec<i64> = (0..10_000).map(|_| rng.gen_range(-1000..=1000)).collect();
        let s1: i64 = data.iter().sum();
        let s2 = parallel_sum(&data);
        assert_eq!(s1, s2);
    }

    #[test]
    fn sort_is_ordered() {
        let mut rng = rand::thread_rng();
        let data: Vec<i32> = (0..10_000).map(|_| rng.gen_range(-1000..=1000)).collect();
        let sorted = parallel_sort(data);
        assert!(sorted.windows(2).all(|w| w[0] <= w[1]));
    }
}
