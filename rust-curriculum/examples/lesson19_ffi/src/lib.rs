use libc::c_int;

extern "C" {
    fn add_i64(a: i64, b: i64) -> i64;
    fn mul_i32(a: c_int, b: c_int) -> c_int;
}

#[no_mangle]
pub extern "C" fn rust_square(x: c_int) -> c_int {
    x * x
}

pub fn add_via_c(a: i64, b: i64) -> i64 {
    unsafe { add_i64(a as i64, b as i64) as i64 }
}

pub fn mul_via_c(a: i32, b: i32) -> i32 {
    unsafe { mul_i32(a as c_int, b as c_int) as i32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calls_c_and_rust_export() {
        assert_eq!(add_via_c(2, 40), 42);
        assert_eq!(mul_via_c(6, 7), 42);
        // call exported Rust symbol from Rust via extern block
        extern "C" { fn rust_square(x: c_int) -> c_int; }
        let r = unsafe { rust_square(7) };
        assert_eq!(r, 49);
    }
}
