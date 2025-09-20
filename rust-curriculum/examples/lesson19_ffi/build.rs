fn main() {
    cc::Build::new()
        .file("src/native.c")
        .compile("nativeffi");
    println!("cargo:rerun-if-changed=src/native.c");
}
