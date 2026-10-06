//! Probability sampling algorithms for autoregressive language model generation.
//!
//! Supports:
//! - Greedy Argmax Decoding (`temperature = 0.0`)
//! - Temperature Scaling
//! - Top-K Filtering
//! - Top-P (Nucleus) Filtering

use rand::Rng;

/// Samples the next token ID from an unnormalized logit distribution.
///
/// # Arguments
/// * `logits` - Raw vocabulary logits from model output.
/// * `temperature` - Sampling temperature. A value of `<= 0.0` triggers deterministic greedy decoding.
/// * `top_k` - Optional limit on the number of highest-probability tokens considered.
/// * `top_p` - Optional cumulative probability threshold for nucleus sampling (e.g. `0.9`).
/// * `rng` - Random number generator instance.
pub fn sample_token<R: Rng>(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rng: &mut R,
) -> usize {
    if logits.is_empty() {
        return 0;
    }

    // 1. Greedy decoding for temperature <= 0.0
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    // 2. Temperature scaling
    let inv_temp = 1.0 / temperature;
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &l)| (idx, l * inv_temp))
        .collect();

    // Sort descending by scaled logit value
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 3. Top-K filtering
    if let Some(k) = top_k {
        if k > 0 && k < indexed.len() {
            indexed.truncate(k);
        }
    }

    // 4. Compute Softmax probabilities over candidates (numerically stable with max subtraction)
    let max_val = indexed.first().map(|p| p.1).unwrap_or(0.0);
    let mut probs: Vec<(usize, f32)> = indexed
        .into_iter()
        .map(|(idx, val)| (idx, (val - max_val).exp()))
        .collect();

    let sum_prob: f32 = probs.iter().map(|p| p.1).sum();
    if sum_prob <= 0.0 || !sum_prob.is_finite() {
        return probs.first().map(|p| p.0).unwrap_or(0);
    }

    for p in &mut probs {
        p.1 /= sum_prob;
    }

    // 5. Top-P (Nucleus) filtering
    if let Some(p_thresh) = top_p {
        if (0.0..1.0).contains(&p_thresh) {
            let mut cumsum = 0.0;
            let mut cutoff_idx = probs.len();
            for (i, &(_, prob)) in probs.iter().enumerate() {
                cumsum += prob;
                if cumsum >= p_thresh {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff_idx);

            // Re-normalize probabilities
            let new_sum: f32 = probs.iter().map(|p| p.1).sum();
            if new_sum > 0.0 {
                for p in &mut probs {
                    p.1 /= new_sum;
                }
            }
        }
    }

    // 6. Categorical sampling using uniform random variable
    let r: f32 = rng.gen_range(0.0..1.0);
    let mut acc = 0.0;
    for &(token_id, prob) in &probs {
        acc += prob;
        if r <= acc {
            return token_id;
        }
    }

    // Fallback to highest probability candidate
    probs.first().map(|p| p.0).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_greedy_sampling() {
        let logits = vec![-1.0, 5.0, 2.0, 0.5];
        let mut rng = StdRng::seed_from_u64(42);

        let selected = sample_token(&logits, 0.0, None, None, &mut rng);
        assert_eq!(selected, 1);
    }

    #[test]
    fn test_top_k_filtering() {
        let logits = vec![10.0, 9.0, 1.0, 0.0];
        let mut rng = StdRng::seed_from_u64(42);

        // With top_k=2, only token 0 or 1 should ever be picked
        for _ in 0..50 {
            let selected = sample_token(&logits, 1.0, Some(2), None, &mut rng);
            assert!(selected == 0 || selected == 1);
        }
    }

    #[test]
    fn test_top_p_filtering() {
        let logits = vec![10.0, 1.0, 0.0, -5.0];
        let mut rng = StdRng::seed_from_u64(42);

        // Token 0 has > 99.9% probability mass, so top_p=0.9 will select token 0 every time
        for _ in 0..20 {
            let selected = sample_token(&logits, 1.0, None, Some(0.9), &mut rng);
            assert_eq!(selected, 0);
        }
    }
}
