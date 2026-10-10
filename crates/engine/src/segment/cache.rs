//! On-disk and in-memory cache for precomputed AI masks (`Subject`, `Sky`, etc.).
//!
//! Stored in the library folder (`<library>/masks/<photo_id>/<kind>.json`), each mask is
//! a serialized [`SegMask`] (~15 KB). A small bounded in-memory cache avoids disk reads
//! for active editing sessions.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

use lightcraft_catalog::PhotoId;
use lightcraft_develop::SegMask;

const MEMORY_CAP: usize = 64;

/// In-memory and on-disk cache for AI masks.
#[derive(Default)]
pub struct MaskCache {
    memory: Mutex<HashMap<(PhotoId, String), SegMask>>,
}

impl MaskCache {
    pub fn new() -> Self {
        Self { memory: Mutex::new(HashMap::new()) }
    }

    /// Retrieve a cached mask, checking in-memory cache first, then on disk.
    pub fn get(&self, masks_dir: Option<&Path>, photo: PhotoId, kind: &str) -> Option<SegMask> {
        let key = (photo, kind.to_string());
        if let Ok(mem) = self.memory.lock()
            && let Some(seg) = mem.get(&key)
        {
            return Some(seg.clone());
        }

        let dir = masks_dir?;
        let file = dir.join(photo.0.to_string()).join(format!("{kind}.json"));
        if !file.is_file() {
            return None;
        }

        let content = fs::read_to_string(&file).ok()?;
        let seg: SegMask = serde_json::from_str(&content).ok()?;

        if let Ok(mut mem) = self.memory.lock() {
            if mem.len() >= MEMORY_CAP {
                mem.clear();
            }
            mem.insert(key, seg.clone());
        }

        Some(seg)
    }

    /// Store a precomputed mask in memory and atomically on disk.
    pub fn store(&self, masks_dir: Option<&Path>, photo: PhotoId, kind: &str, seg: &SegMask) {
        let key = (photo, kind.to_string());
        if let Ok(mut mem) = self.memory.lock() {
            if mem.len() >= MEMORY_CAP {
                mem.clear();
            }
            mem.insert(key, seg.clone());
        }

        let Some(dir) = masks_dir else { return };
        let photo_dir = dir.join(photo.0.to_string());
        if fs::create_dir_all(&photo_dir).is_err() {
            return;
        }

        let path = photo_dir.join(format!("{kind}.json"));
        let Ok(json) = serde_json::to_string(seg) else { return };
        let tmp = photo_dir.join(format!("{kind}.json.tmp.{}", std::process::id()));
        if fs::write(&tmp, json).is_ok() {
            let _ = fs::rename(tmp, path);
        }
    }

    /// Invalidate/remove all cached masks for a photo (e.g. when removed or modified).
    pub fn remove(&self, masks_dir: Option<&Path>, photo: PhotoId) {
        if let Ok(mut mem) = self.memory.lock() {
            mem.retain(|(p, _), _| *p != photo);
        }
        if let Some(dir) = masks_dir {
            let photo_dir = dir.join(photo.0.to_string());
            let _ = fs::remove_dir_all(photo_dir);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_round_trip_and_invalidation() {
        let tmp = std::env::temp_dir().join(format!("lc-mask-cache-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let cache = MaskCache::new();
        let photo = PhotoId(12345);
        let seg = SegMask::from_logits(4, &[-1.0; 16]);

        assert!(cache.get(Some(&tmp), photo, "subject").is_none());

        cache.store(Some(&tmp), photo, "subject", &seg);
        let retrieved = cache.get(Some(&tmp), photo, "subject");
        assert_eq!(retrieved, Some(seg.clone()));

        // Clear memory to test disk retrieval
        if let Ok(mut mem) = cache.memory.lock() {
            mem.clear();
        }
        let from_disk = cache.get(Some(&tmp), photo, "subject");
        assert_eq!(from_disk, Some(seg));

        cache.remove(Some(&tmp), photo);
        assert!(cache.get(Some(&tmp), photo, "subject").is_none());
        let _ = fs::remove_dir_all(&tmp);
    }
}
