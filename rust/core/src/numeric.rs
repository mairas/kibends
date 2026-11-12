//! Scalar operations as numpy performs them, so that results the optimizer
//! takes agree with the numpy code they replaced to the last bit: its
//! `minimum`, `maximum` and `clip`, which propagate NaN where Rust's `min`
//! and `max` drop it, the pairwise summation of `numpy.sum` and the fused
//! products of `numpy.einsum`; and a correctly rounded sum, for the sums
//! numpy left to its BLAS, whose order no platform shares.

/// numpy's `minimum`: NaN if either is NaN.
pub fn minimum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a <= b {
        a
    } else {
        b
    }
}

/// numpy's `maximum`: NaN if either is NaN.
pub fn maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

/// numpy's `clip`: `x` held between `low` and `high`, NaN if it is NaN.
pub fn clip(x: f64, low: f64, high: f64) -> f64 {
    minimum(maximum(x, low), high)
}

/// Below this many values numpy adds them one after another.
const UNROLLED: usize = 8;
/// numpy's `PW_BLOCKSIZE`: up to this many values it adds with eight
/// running sums, beyond it it splits the values in two.
const BLOCK: usize = 128;

/// The sum of `values` as `numpy.sum` takes it over a contiguous array:
/// pairwise, after its identity, 0.
pub fn numpy_sum(values: &[f64]) -> f64 {
    0.0 + pairwise_sum(values)
}

fn pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < UNROLLED {
        return values.iter().fold(-0.0, |sum, &v| sum + v);
    }
    if n <= BLOCK {
        let mut r = [0.0; UNROLLED];
        r.copy_from_slice(&values[..UNROLLED]);
        let whole = n - n % UNROLLED;
        for chunk in values[UNROLLED..whole].as_chunks::<UNROLLED>().0 {
            for (sum, &v) in r.iter_mut().zip(chunk) {
                *sum += v;
            }
        }
        let mut sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &values[whole..] {
            sum += v;
        }
        return sum;
    }
    let mut half = n / 2;
    half -= half % UNROLLED;
    pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
}

/// Σ a[k]·b[k] in order, each product fused into the running sum, as
/// numpy's `einsum` contracts an axis.
pub fn fused_dot(a: impl IntoIterator<Item = f64>, b: impl IntoIterator<Item = f64>) -> f64 {
    a.into_iter()
        .zip(b)
        .fold(0.0, |sum, (x, y)| x.mul_add(y, sum))
}

/// The sum of `values` correctly rounded, as Python's `math.fsum` gives
/// it: Shewchuk's exact partials, rounded half to even once at the end.
/// NaN if a value is NaN or infinities of both signs meet, an infinity if
/// one is among the values or the sum overflows.
pub fn exact_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    let mut special = 0.0;
    for value in values {
        if !value.is_finite() {
            special += value;
            continue;
        }
        let mut x = value;
        let mut kept = 0;
        for k in 0..partials.len() {
            let mut y = partials[k];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let high = x + y;
            let low = y - (high - x);
            if low != 0.0 {
                partials[kept] = low;
                kept += 1;
            }
            x = high;
        }
        partials.truncate(kept);
        if !x.is_finite() {
            special += x;
            partials.clear();
            continue;
        }
        partials.push(x);
    }
    if special != 0.0 || special.is_nan() {
        return special;
    }
    let Some(mut n) = partials.len().checked_sub(1) else {
        return 0.0;
    };
    let mut high = partials[n];
    let mut low = 0.0;
    while n > 0 {
        let x = high;
        n -= 1;
        let y = partials[n];
        high = x + y;
        low = y - (high - x);
        if low != 0.0 {
            break;
        }
    }
    // Rounding half to even: the partials below may push a tie either way.
    if n > 0 && ((low < 0.0 && partials[n - 1] < 0.0) || (low > 0.0 && partials[n - 1] > 0.0)) {
        let y = low * 2.0;
        let x = high + y;
        if y == x - high {
            high = x;
        }
    }
    high
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimum_and_maximum_propagate_nan() {
        assert!(minimum(f64::NAN, 1.0).is_nan());
        assert!(maximum(1.0, f64::NAN).is_nan());
        assert!(clip(f64::NAN, 0.0, 1.0).is_nan());
        assert_eq!(clip(2.0, 0.0, 1.0), 1.0);
    }

    #[test]
    fn numpy_sum_adds_every_value() {
        for n in [0usize, 1, 7, 8, 9, 127, 128, 129, 1000] {
            let values: Vec<f64> = (0..n).map(|k| k as f64).collect();
            assert_eq!(numpy_sum(&values), (n * n.saturating_sub(1) / 2) as f64);
        }
    }

    #[test]
    fn exact_sum_is_correctly_rounded() {
        assert_eq!(exact_sum([1.0, 1e100, 1.0, -1e100]), 2.0);
        assert_eq!(exact_sum([0.1; 10]), 1.0);
        let tie = 2f64.powi(-53);
        assert_eq!(exact_sum([1.0, tie, tie * tie]), 1.0 + 2.0 * tie);
        assert_eq!(exact_sum([]), 0.0);
        assert!(exact_sum([1.0, f64::NAN]).is_nan());
        assert!(exact_sum([f64::INFINITY, f64::NEG_INFINITY]).is_nan());
        assert_eq!(exact_sum([f64::INFINITY, 1.0]), f64::INFINITY);
    }

    #[test]
    fn numpy_sum_rounds_in_numpy_s_order() {
        // Sequentially 1 + 2⁻⁵³ + ... stays 1; numpy's eight running sums
        // lose only the one 2⁻⁵³ added to 1 itself.
        let mut values = vec![1.0];
        values.extend([2f64.powi(-53); 15]);
        assert_eq!(numpy_sum(&values), 1.0 + 14.0 * 2f64.powi(-53));
    }
}
