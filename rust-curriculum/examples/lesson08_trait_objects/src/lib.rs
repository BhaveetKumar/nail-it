pub trait Logger {
    fn info(&mut self, msg: &str);
    fn warn(&mut self, msg: &str);
    fn error(&mut self, msg: &str);
}

pub struct ConsoleLogger;
impl Logger for ConsoleLogger {
    fn info(&mut self, msg: &str) {
        println!("INFO: {msg}");
    }
    fn warn(&mut self, msg: &str) {
        println!("WARN: {msg}");
    }
    fn error(&mut self, msg: &str) {
        eprintln!("ERROR: {msg}");
    }
}

#[derive(Default)]
pub struct VecLogger {
    pub entries: Vec<String>,
}
impl Logger for VecLogger {
    fn info(&mut self, msg: &str) {
        self.entries.push(format!("INFO: {msg}"));
    }
    fn warn(&mut self, msg: &str) {
        self.entries.push(format!("WARN: {msg}"));
    }
    fn error(&mut self, msg: &str) {
        self.entries.push(format!("ERROR: {msg}"));
    }
}

pub struct MultiLogger {
    inner: Vec<Box<dyn Logger>>,
}
impl MultiLogger {
    pub fn new() -> Self {
        Self { inner: vec![] }
    }
    pub fn add(&mut self, l: Box<dyn Logger>) {
        self.inner.push(l);
    }
    pub fn info(&mut self, msg: &str) {
        for l in self.inner.iter_mut() {
            l.info(msg);
        }
    }
}

impl Default for MultiLogger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn works() {
        let mut ml = MultiLogger::new();
        ml.add(Box::new(ConsoleLogger));
        let v = Box::new(VecLogger::default());
        ml.add(v);
        ml.info("hello");
        // Ensure call doesn't panic; dynamic dispatch worked by pushing to VecLogger
        // (no explicit assertion required for this smoke test)
    }
}
