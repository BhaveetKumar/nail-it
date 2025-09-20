/// Adds two numbers.
///
/// # Examples
/// ```
/// assert_eq!(lesson03_tooling::add(2,2), 4);
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        assert_eq!(add(1, 2), 3);
    }
}
