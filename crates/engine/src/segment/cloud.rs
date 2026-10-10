//! Cloud AI masking client (pure Rust): fal.ai / Replicate / Custom.
//!
//! Provides fast remote segmentation without local heavy models or GPU requirements.

use std::sync::atomic::AtomicBool;
use std::time::Duration;

use lightcraft_develop::SegMask;
use lightcraft_segment::MASK_SIDE;

use super::worker::{Job, Kind};
use super::Input;

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk.first().copied().unwrap_or(0),
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(char::from(B64[((n >> (18 - 6 * i)) & 63) as usize]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut buf = 0u32;
    let mut bits = 0;
    for &b in s.as_bytes() {
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\r' | b'\n' | b' ' => continue,
            _ => return None,
        } as u32;
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Some(out)
}

/// Execute a cloud segmentation job via fal.ai, Replicate, or a configured endpoint.
pub(crate) fn cloud_infer(job: &mut Job, input: &Input) -> Result<Option<SegMask>, String> {
    let key = if !job.config.cloud_key.trim().is_empty() {
        job.config.cloud_key.trim().to_string()
    } else if let Ok(k) = std::env::var("FAL_KEY") {
        k
    } else if let Ok(k) = std::env::var("LIGHTCRAFT_CLOUD_KEY") {
        k
    } else {
        return Err("Cloud AI masking requires an API key. Please enter your API key in Settings (⌘,) ▸ AI Masking, or choose Local Models.".into());
    };

    // 1. Encode preview input to JPEG
    let img = lightcraft_codecs::EncodeImage::new(
        input.w as u32,
        input.h as u32,
        3,
        lightcraft_codecs::Samples::U8(&input.rgb),
    );
    let jpeg_bytes = lightcraft_codecs::encode::encode_jpeg(
        &img,
        85,
        lightcraft_codecs::ChromaSubsampling::S420,
        &lightcraft_codecs::EncodeMeta::default(),
    )
    .map_err(|e| format!("Failed to encode image for cloud: {e}"))?;

    let data_url = format!("data:image/jpeg;base64,{}", base64_encode(&jpeg_bytes));

    // 2. Select endpoint and payload based on service and job kind
    let (endpoint, payload, auth_header) = match job.config.cloud_service.as_str() {
        "replicate" => {
            let ep = if job.config.cloud_endpoint.trim().is_empty() {
                "https://api.replicate.com/v1/predictions".to_string()
            } else {
                job.config.cloud_endpoint.clone()
            };
            let body = serde_json::json!({
                "input": {
                    "image": data_url
                }
            });
            (ep, body, format!("Bearer {key}"))
        }
        _ => {
            // Default to fal.ai
            let ep = match &job.kind {
                Kind::Subject | Kind::People { .. } | Kind::Refine { .. } => {
                    "https://fal.run/fal-ai/birefnet".to_string()
                }
                Kind::Text(_) => {
                    "https://fal.run/fal-ai/image-segmentation".to_string()
                }
                Kind::Clicks(_) => {
                    "https://fal.run/fal-ai/sam2".to_string()
                }
                _ => "https://fal.run/fal-ai/birefnet".to_string(),
            };
            let body = match &job.kind {
                Kind::Text(prompt) => serde_json::json!({
                    "image_url": data_url,
                    "prompt": prompt,
                }),
                Kind::Clicks(clicks) => {
                    let pts: Vec<_> = clicks.iter().map(|c| serde_json::json!({"x": c.x, "y": c.y, "label": if c.positive { 1 } else { 0 }})).collect();
                    serde_json::json!({
                        "image_url": data_url,
                        "points": pts,
                    })
                }
                _ => serde_json::json!({
                    "image_url": data_url,
                }),
            };
            (ep, body, format!("Key {key}"))
        }
    };

    let url = lightcraft_fetch::http::Url::parse(&endpoint).map_err(|e| e.to_string())?;
    let headers = [
        ("Authorization", auth_header),
        ("Content-Type", "application/json".to_string()),
        ("Accept", "application/json".to_string()),
    ];
    let body_bytes = serde_json::to_vec(&payload).map_err(|e| e.to_string())?;
    let cancel = AtomicBool::new(false);
    let limits = lightcraft_fetch::http::Limits {
        connect: Duration::from_secs(10),
        stall: Duration::from_secs(45),
        cancel: &cancel,
    };

    let mut resp = lightcraft_fetch::http::post(&url, &headers, &body_bytes, &limits)
        .map_err(|e| format!("Cloud request failed: {e}"))?;

    let mut resp_body = Vec::new();
    let mut chunk = [0u8; 16384];
    loop {
        let n = resp.read(&mut chunk, &limits).map_err(|e| format!("Failed to read cloud response: {e}"))?;
        if n == 0 {
            break;
        }
        resp_body.extend_from_slice(chunk.get(..n).unwrap_or(&[]));
    }

    if resp.status < 200 || resp.status >= 300 {
        let err_text = String::from_utf8_lossy(&resp_body);
        return Err(format!("Cloud API error (status {}): {err_text}", resp.status));
    }

    let val: serde_json::Value = serde_json::from_slice(&resp_body)
        .map_err(|e| format!("Failed to parse cloud response: {e}"))?;

    // 3. Extract mask image URL / data
    let mask_url = val.get("image").and_then(|v| v.get("url")).and_then(|v| v.as_str())
        .or_else(|| val.get("mask").and_then(|v| v.get("url")).and_then(|v| v.as_str()))
        .or_else(|| val.get("output").and_then(|v| v.get("mask_url")).and_then(|v| v.as_str()))
        .or_else(|| val.get("output").and_then(|v| v.as_str()))
        .ok_or_else(|| format!("Cloud response did not contain a mask URL: {}", serde_json::to_string(&val).unwrap_or_default()))?;

    // 4. Load the mask bytes
    let mask_bytes = if let Some(b64) = mask_url.strip_prefix("data:image/png;base64,")
        .or_else(|| mask_url.strip_prefix("data:image/jpeg;base64,"))
    {
        base64_decode(b64).ok_or("Failed to decode base64 mask from cloud")?
    } else if mask_url.starts_with("http://") || mask_url.starts_with("https://") {
        let mask_http_url = lightcraft_fetch::http::Url::parse(mask_url).map_err(|e| e.to_string())?;
        let mut get_resp = lightcraft_fetch::http::get(&mask_http_url, &[], &limits)
            .map_err(|e| format!("Failed to fetch cloud mask image: {e}"))?;
        let mut buf = Vec::new();
        loop {
            let n = get_resp.read(&mut chunk, &limits).map_err(|e| format!("Failed to download mask: {e}"))?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(chunk.get(..n).unwrap_or(&[]));
        }
        buf
    } else {
        return Err(format!("Unrecognized mask URL format: {mask_url}"));
    };

    // 5. Decode mask image to alpha logits
    let decoded = lightcraft_codecs::decode(&mask_bytes, lightcraft_codecs::DecodeOptions::default())
        .map_err(|e| format!("Failed to decode returned mask image: {e}"))?;
    let srgb = decoded.to_srgb8();

    let mw = srgb.width;
    let mh = srgb.height;
    if mw == 0 || mh == 0 {
        return Err("Empty mask returned from cloud".into());
    }

    // Determine if mask is in alpha or luminance
    let has_varying_alpha = srgb.data.iter().any(|px| px[3] != 255);
    let side = MASK_SIDE;
    let mut logits = Vec::with_capacity(side * side);

    for y in 0..side {
        let sy = (y * mh / side).min(mh.saturating_sub(1));
        for x in 0..side {
            let sx = (x * mw / side).min(mw.saturating_sub(1));
            let idx = sy * mw + sx;
            let px = srgb.data.get(idx).copied().unwrap_or([0, 0, 0, 0]);
            let alpha = if has_varying_alpha {
                px[3]
            } else {
                ((u32::from(px[0]) + u32::from(px[1]) + u32::from(px[2])) / 3) as u8
            };
            let p = ((f32::from(alpha) + 0.5) / 256.0).clamp(0.0001, 0.9999);
            let logit = (p / (1.0 - p)).ln();
            logits.push(logit);
        }
    }

    Ok(Some(SegMask::from_logits(side, &logits)))
}
