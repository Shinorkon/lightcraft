//! Face and body part segmentation classes and utilities (BiSeNet).

/// Standard 19 CelebAMask-HQ classes produced by BiSeNet.
pub const CLASSES: &[&str] = &[
    "background",
    "skin",
    "left_eyebrow",
    "right_eyebrow",
    "left_eye",
    "right_eye",
    "glasses",
    "left_ear",
    "right_ear",
    "earrings",
    "nose",
    "mouth",
    "upper_lip",
    "lower_lip",
    "neck",
    "necklace",
    "clothes",
    "hair",
    "hat",
];

/// Resolve user-facing part names (e.g., "facial_skin", "eyes", "lips", "hair")
/// to BiSeNet class indices (0..19).
pub fn resolve_parts(parts: &[String]) -> Vec<usize> {
    let mut indices = Vec::new();
    for part in parts {
        let p = part.to_ascii_lowercase().replace(['-', ' '], "_");
        match p.as_str() {
            "all" | "person" => {
                indices.extend(1..=18);
            }
            "skin" | "facial_skin" | "face" => indices.push(1),
            "eyebrow" | "eyebrows" => {
                indices.push(2);
                indices.push(3);
            }
            "left_eyebrow" => indices.push(2),
            "right_eyebrow" => indices.push(3),
            "eye" | "eyes" => {
                indices.push(4);
                indices.push(5);
            }
            "left_eye" => indices.push(4),
            "right_eye" => indices.push(5),
            "glasses" => indices.push(6),
            "ear" | "ears" => {
                indices.push(7);
                indices.push(8);
            }
            "left_ear" => indices.push(7),
            "right_ear" => indices.push(8),
            "earrings" => indices.push(9),
            "nose" => indices.push(10),
            "mouth" => indices.push(11),
            "lip" | "lips" => {
                indices.push(12);
                indices.push(13);
            }
            "upper_lip" => indices.push(12),
            "lower_lip" => indices.push(13),
            "neck" => indices.push(14),
            "necklace" => indices.push(15),
            "clothes" | "clothing" => indices.push(16),
            "hair" => indices.push(17),
            "hat" => indices.push(18),
            _ => {
                if let Some(pos) = CLASSES.iter().position(|&c| c == p) {
                    indices.push(pos);
                }
            }
        }
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Combine class logits [19, side, side] for the selected class indices into a single logit map [side, side].
pub fn combine_logits(logits: &[f32], side: usize, indices: &[usize]) -> Vec<f32> {
    let num_classes = 19;
    let plane_size = side * side;
    if logits.len() < num_classes * plane_size || indices.is_empty() {
        return vec![-10.0; plane_size];
    }
    let mut out = vec![-10.0; plane_size];
    for (i, v) in out.iter_mut().enumerate().take(plane_size) {
        let mut max_selected = -100.0f32;
        let mut max_other = -100.0f32;
        for c in 0..num_classes {
            let l = logits.get(c * plane_size + i).copied().unwrap_or(-10.0);
            if indices.contains(&c) {
                if l > max_selected {
                    max_selected = l;
                }
            } else if l > max_other {
                max_other = l;
            }
        }
        *v = max_selected - max_other;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_composite_parts() {
        let parts = vec!["lips".into(), "hair".into(), "skin".into()];
        let indices = resolve_parts(&parts);
        assert!(indices.contains(&1));
        assert!(indices.contains(&12));
        assert!(indices.contains(&13));
        assert!(indices.contains(&17));
    }
}
