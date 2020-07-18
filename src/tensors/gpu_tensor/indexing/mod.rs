/// Copied shamelessly from NDArray
mod index_to_dim_stride;
use std::ops::{Deref, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use std::iter::StepBy;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SliceRangeInfo {
    pub start: usize,
    pub inclusive_end: Option<usize>,
    pub step: usize,
}

impl SliceRangeInfo {
    /// Create a new `Slice` with the given extents.
    ///
    /// See also the `From` impls, converting from ranges; for example
    /// `Slice::from(i..)` or `Slice::from(j..k)`.
    ///
    /// `step` must be nonzero.
    /// (This method checks with a debug assertion that `step` is not zero.)
    pub fn new(start: usize, inclusive_end: Option<usize>, step: usize) -> SliceRangeInfo {
        debug_assert_ne!(step, 0, "Slice::new: step must be nonzero");
        SliceRangeInfo { start, inclusive_end, step }
    }

    /// Create a new `Slice` with the given step size (multiplied with the
    /// previous step size).
    ///
    /// `step` must be nonzero.
    /// (This method checks with a debug assertion that `step` is not zero.)
    #[inline]
    pub fn step_by(self, step: usize) -> Self {
        debug_assert_ne!(step, 0, "Slice::step_by: step must be nonzero");
        SliceRangeInfo {
            step: self.step * step,
            ..self
        }
    }
}



macro_rules! impl_slice_variant_from_range {
    ($self:ty, $constructor:path, $index:ty) => {
        impl From<Range<$index>> for $self {
            #[inline]
            fn from(r: Range<$index>) -> $self {
                $constructor {
                    start: r.start as usize,
                    inclusive_end: Some(r.end - 1 as usize),
                    step: 1,
                }
            }
        }

        impl From<RangeInclusive<$index>> for $self {
            #[inline]
            fn from(r: RangeInclusive<$index>) -> $self {
                let end = *r.end() as usize;
                $constructor {
                    start: *r.start() as usize,
                    inclusive_end: Some(end),
                    step: 1,
                }
            }
        }

        impl From<RangeFrom<$index>> for $self {
            #[inline]
            fn from(r: RangeFrom<$index>) -> $self {
                $constructor {
                    start: r.start as usize,
                    inclusive_end: None,
                    step: 1,
                }
            }
        }

        impl From<RangeTo<$index>> for $self {
            #[inline]
            fn from(r: RangeTo<$index>) -> $self {
                $constructor {
                    start: 0,
                    inclusive_end: Some(r.end as usize - 1),
                    step: 1,
                }
            }
        }

        impl From<RangeToInclusive<$index>> for $self {
            #[inline]
            fn from(r: RangeToInclusive<$index>) -> $self {
                let end = r.end as usize;
                $constructor {
                    start: 0,
                    inclusive_end: Some(end),
                    step: 1,
                }
            }
        }
    };
}
// impl_slice_variant_from_range!(Slice, Slice, usize);
impl_slice_variant_from_range!(SliceRangeInfo, SliceRangeInfo, usize);
// impl_slice_variant_from_range!(Slice, Slice, i32);

impl From<usize> for SliceRangeInfo {
    fn from(number: usize) -> Self {
        SliceRangeInfo::new(number, Some(number), 1)
    }
}

impl From<(usize, usize, usize)> for SliceRangeInfo {
    fn from(tuple: (usize, usize, usize)) -> Self {
        SliceRangeInfo{
            start: tuple.0,
            inclusive_end: Some(tuple.2 - 1),
            step: tuple.1
        }
    }
}

// Calls Into::<SliceRangeInfo>::into on each of its inputs and puts all of them into a Vector<SliceRangeInfo>
#[macro_export]
macro_rules! s (
    // Creates a Vec and converts each expression into a SliceRangeInfo
    // Example: s![2] => let mut vector = Vector::new(); vector.push(Into::<SliceRangeInfo>::into(2))
    ($($other_dims:expr);+) => {
    {
        let mut vector = Vec::new();
        $(vector.push(Into::<SliceRangeInfo>::into($other_dims));)+
        vector
    }
    };
    ($($t:tt)*) => { compile_error!("Invalid syntax in s![] call.") };
);

#[test]
fn test(){
    let a = SliceRangeInfo::from((0..3));
    // let b: dyn Into<Slice> = (0..4);
    println!("{:?}", a);
}

