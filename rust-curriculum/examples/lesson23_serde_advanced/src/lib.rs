use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "data")]
pub enum Message {
    Text(String),
    Binary(#[serde(with = "b64")] Vec<u8>),
}

pub mod b64 {
    use super::*;
    use base64::{engine::general_purpose, Engine};

    pub fn serialize<S: Serializer>(bytes: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        let encoded = general_purpose::STANDARD.encode(bytes);
        String::serialize(&encoded, s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        general_purpose::STANDARD
            .decode(s.as_bytes())
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SerdeAdvError {
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn to_json(msg: &Message) -> Result<String, SerdeAdvError> {
    Ok(serde_json::to_string(msg)?)
}

pub fn from_json(s: &str) -> Result<Message, SerdeAdvError> {
    Ok(serde_json::from_str::<Message>(s)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_text() {
        let m = Message::Text("hello".into());
        let j = to_json(&m).unwrap();
        assert!(j.contains("\"type\":\"Text\""));
        let back = from_json(&j).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn roundtrip_binary_base64() {
        let m = Message::Binary(vec![1, 2, 3, 4, 255]);
        let j = to_json(&m).unwrap();
        assert!(j.contains("\"type\":\"Binary\""));
        // data should be base64, not raw numbers
        assert!(j.contains("data"));
        let back = from_json(&j).unwrap();
        assert_eq!(m, back);
    }
}
