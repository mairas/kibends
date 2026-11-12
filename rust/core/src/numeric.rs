//! Scalar operations in numpy's order: its `minimum`, `maximum` and
//! `clip`, which propagate NaN where Rust's `min` and `max` drop it, the
//! pairwise summation of `numpy.sum` and the fused products of
//! `numpy.einsum`; `numpy.interp` as the macOS arm64 wheel computes it,
//! fused, which Linux wheels do not; and a correctly rounded sum where
//! numpy leaves the order to its BLAS, whose order no platform shares.
//!
//! Keeping these orders keeps the optimizer's results on the recorded
//! scipy baseline (`tests/test_scipy_baseline.py`) and the same on every
//! platform. Changing one moves moving-via cases into other local minima,
//! so re-run the baseline test after changing one.

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

/// `numpy.interp(x, xp, fp)` with its default ends, for knots `xp` in
/// ascending order: `fp`'s end values beyond the knots, `fp[j]` at a knot
/// (the last of repeated ones), otherwise the slope of the knots' interval
/// times the offset from its lower knot, fused into the sum with that
/// knot's value as the macOS arm64 wheel compiles it. Linux wheels round
/// the product first, which is the platform difference this removes. A NaN
/// result is retried from the upper knot, then given the knots' value if
/// they are equal; NaN `x` stays NaN, except with one knot.
///
/// Panics if `xp` is empty or longer than `fp`.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let last = xp.len() - 1;
    if last == 0 {
        return fp[0];
    }
    if x.is_nan() {
        return x;
    }
    if x < xp[0] {
        return fp[0];
    }
    let j = xp.partition_point(|&knot| knot <= x) - 1;
    if j == last || xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let y = slope.mul_add(x - xp[j], fp[j]);
    if !y.is_nan() {
        return y;
    }
    let y = slope.mul_add(x - xp[j + 1], fp[j + 1]);
    if y.is_nan() && fp[j] == fp[j + 1] {
        fp[j]
    } else {
        y
    }
}

/// Σ a[k]·b[k] in order, each product fused into the running sum, as
/// numpy's `einsum` contracts an axis.
pub fn fused_dot(a: impl IntoIterator<Item = f64>, b: impl IntoIterator<Item = f64>) -> f64 {
    a.into_iter()
        .zip(b)
        .fold(0.0, |sum, (x, y)| x.mul_add(y, sum))
}

/// The sum of `values` as Python's built-in `sum` takes floats from a start
/// of 0: Neumaier's compensated summation, the compensation added once at
/// the end if it is finite.
pub fn python_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for x in values {
        let t = sum + x;
        compensation += if f64::abs(sum) >= x.abs() {
            (sum - t) + x
        } else {
            (x - t) + sum
        };
        sum = t;
    }
    if compensation != 0.0 && compensation.is_finite() {
        sum += compensation;
    }
    sum
}

/// Python's `max(a, b)`: `a` unless `b` is greater.
pub fn python_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python's `min(a, b)`: `a` unless `b` is less.
pub fn python_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
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
    fn python_sum_compensates() {
        assert_eq!(python_sum([0.1; 10]), 1.0);
        assert_eq!(python_sum([1.0, 1e100, 1.0, -1e100]), 2.0);
        assert_eq!(python_sum([]), 0.0);
    }

    #[test]
    fn interp_fuses_the_product_into_the_sum() {
        let x = 1.200_000_000_000_000_2;
        let unfused = (1.0 / 3.0) * x + 1.0;
        assert_eq!(interp(x, &[0.0, 3.0], &[1.0, 2.0]), 1.400_000_000_000_000_1);
        assert_ne!(interp(x, &[0.0, 3.0], &[1.0, 2.0]), unfused);
    }

    #[test]
    fn interp_holds_the_end_values_beyond_the_knots() {
        let (xp, fp) = ([0.0, 1.0, 2.0], [5.0, 6.0, 8.0]);
        assert_eq!(interp(-1.0, &xp, &fp), 5.0);
        assert_eq!(interp(3.0, &xp, &fp), 8.0);
        assert_eq!(interp(2.0, &xp, &fp), 8.0);
        assert_eq!(interp(1.0, &xp, &fp), 6.0);
        assert_eq!(interp(1.5, &xp, &fp), 7.0);
    }

    #[test]
    fn interp_takes_the_last_of_repeated_knots() {
        let (xp, fp) = ([0.0, 1.0, 1.0, 2.0], [0.0, 1.0, 3.0, 4.0]);
        assert_eq!(interp(1.0, &xp, &fp), 3.0);
        assert_eq!(interp(0.5, &xp, &fp), 0.5);
        assert_eq!(interp(1.5, &xp, &fp), 3.5);
    }

    #[test]
    fn interp_passes_nan_through_unless_there_is_one_knot() {
        assert!(interp(f64::NAN, &[0.0, 1.0], &[0.0, 1.0]).is_nan());
        assert_eq!(interp(f64::NAN, &[0.0], &[4.0]), 4.0);
        assert_eq!(interp(-1.0, &[0.0], &[4.0]), 4.0);
    }

    #[test]
    fn interp_gives_equal_infinite_knots_their_value() {
        // Between ∞ and ∞ the slope is NaN from either end, so the knots'
        // common value stands; between -∞ and ∞ NaN does.
        let near_nan = interp(0.5, &[0.0, 1.0], &[f64::NEG_INFINITY, f64::INFINITY]);
        assert!(near_nan.is_nan());
        let flat = interp(0.5, &[0.0, 1.0], &[f64::INFINITY, f64::INFINITY]);
        assert_eq!(flat, f64::INFINITY);
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
