use sled::IVec;

pub struct Kv {
    db: sled::Db,
}

impl Kv {
    pub fn open(path: &std::path::Path) -> sled::Result<Self> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }
    pub fn put(&self, k: &str, v: &[u8]) -> sled::Result<Option<IVec>> {
        self.db.insert(k, v)
    }
    pub fn get(&self, k: &str) -> sled::Result<Option<IVec>> {
        self.db.get(k)
    }
    pub fn del(&self, k: &str) -> sled::Result<Option<IVec>> {
        self.db.remove(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn roundtrip() {
        let dir = TempDir::new().unwrap();
        let kv = Kv::open(dir.path()).unwrap();
        kv.put("a", b"1").unwrap();
        assert_eq!(kv.get("a").unwrap().unwrap().as_ref(), b"1");
        assert_eq!(kv.del("a").unwrap().unwrap().as_ref(), b"1");
        assert!(kv.get("a").unwrap().is_none());
    }
}
