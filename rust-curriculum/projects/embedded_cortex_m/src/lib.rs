#![cfg_attr(not(feature = "std"), no_std)]

pub struct FixedAccum {
    value: i32, // Q16.16 fixed point
}

impl FixedAccum {
    pub fn new() -> Self { Self { value: 0 } }
    pub fn from_i32(x: i32) -> Self { Self { value: x << 16 } }
    pub fn add_i32(&mut self, x: i32) { self.value = self.value.wrapping_add(x << 16); }
    pub fn mul_i32(&mut self, x: i32) { self.value = self.value.wrapping_mul(x); }
    pub fn to_i32(&self) -> i32 { self.value >> 16 }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basics() {
        let mut a = FixedAccum::from_i32(2);
        a.add_i32(3);
        assert_eq!(a.to_i32(), 5);
        a.mul_i32(4);
        assert_eq!(a.to_i32(), 20);
    }
}
