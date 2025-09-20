pub unsafe fn copy_bytes(src: *const u8, dst: *mut u8, len: usize) {
    std::ptr::copy_nonoverlapping(src, dst, len);
}

pub unsafe fn split_at_mut_manual(slice: &mut [u8], mid: usize) -> (&mut [u8], &mut [u8]) {
    let len = slice.len();
    let ptr = slice.as_mut_ptr();
    assert!(mid <= len);
    (
        std::slice::from_raw_parts_mut(ptr, mid),
        std::slice::from_raw_parts_mut(ptr.add(mid), len - mid),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_bytes() {
        let src = b"hello";
        let mut dst = [0u8; 5];
        unsafe { copy_bytes(src.as_ptr(), dst.as_mut_ptr(), src.len()) };
        assert_eq!(&dst, src);
    }

    #[test]
    fn splits_mut_slice() {
        let mut buf = *b"abcdef";
        let (a,b) = unsafe { split_at_mut_manual(&mut buf, 2) };
        assert_eq!(a, b"ab");
        assert_eq!(b, b"cdef");
        b[0] = b'X';
        assert_eq!(&buf, b"abXdef");
    }
}
