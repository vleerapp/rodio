use crate::Float;

/// Divide `a` by `b` unless `b` is NaN, infinite, or <= 0,
/// in which case `fallback` is returned.
#[inline(always)]
pub(super) fn div_or_fallback(a: Float, b: Float, fallback: Float) -> Float {
    if b.is_finite() && b > 0.0 {
        a / b
    } else {
        fallback
    }
}
