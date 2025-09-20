use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct User {
    pub id: u64,
    #[serde(rename = "user_name")]
    pub username: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub email: Option<String>,
}

impl User {
    pub fn new(id: u64, username: impl Into<String>, email: Option<String>) -> Self {
        Self { id, username: username.into(), email }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SerdeExampleError {
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn to_json(u: &User) -> Result<String, SerdeExampleError> {
    Ok(serde_json::to_string(u)?)
}

pub fn from_json(s: &str) -> Result<User, SerdeExampleError> {
    Ok(serde_json::from_str::<User>(s)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_with_email() {
        let user = User::new(1, "alice", Some("alice@example.com".to_string()));
        let json = to_json(&user).unwrap();
        // email present when Some
        assert!(json.contains("\"email\":"));
        let back = from_json(&json).unwrap();
        assert_eq!(user, back);
    }

    #[test]
    fn skip_email_when_none() {
        let user = User::new(2, "bob", None);
        let json = to_json(&user).unwrap();
        // email omitted when None due to skip_serializing_if
        assert!(!json.contains("email"));
        let back = from_json(&json).unwrap();
        assert_eq!(user, back);
    }

    #[test]
    fn accepts_renamed_field() {
        // username is mapped to user_name in JSON
        let json = r#"{"id":3,"user_name":"carol"}"#;
        let user = from_json(json).unwrap();
        assert_eq!(user.username, "carol");
        assert_eq!(user.email, None);
    }
}
