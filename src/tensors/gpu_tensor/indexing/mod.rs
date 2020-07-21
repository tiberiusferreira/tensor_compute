/// Copied shamelessly from NDArray
mod index_to_dim_stride;
pub use index_to_dim_stride::shape_strides_for_slice_range;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SliceRangeInfo {
    pub start: usize,
    pub inclusive_end: Option<usize>,
    pub step: usize,
}

impl SliceRangeInfo {
    pub fn new(start: usize, mut exclusive_end: Option<usize>, step: usize) -> Self {
        if let Some(end) = &mut exclusive_end {
            let isize_end = *end as isize;
            assert!(
                start as isize <= isize_end - 1,
                format!(
                    "Start ({:?}) needs to be smaller than end ({:?}).",
                    start,
                    isize_end - 1
                )
            );
            *end -= 1;
        }
        assert!(step > 0, "Step need to be greater than 0");
        SliceRangeInfo {
            start,
            inclusive_end: exclusive_end,
            step,
        }
    }
}

impl SliceRangeInfo {
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
                <$constructor>::new(r.start as usize, Some(r.end as usize), 1)
            }
        }

        impl From<RangeInclusive<$index>> for $self {
            #[inline]
            fn from(r: RangeInclusive<$index>) -> $self {
                let end = *r.end() as usize;
                <$constructor>::new(*r.start() as usize, Some(end + 1), 1)
            }
        }

        impl From<RangeFrom<$index>> for $self {
            #[inline]
            fn from(r: RangeFrom<$index>) -> $self {
                <$constructor>::new(r.start as usize, None, 1)
            }
        }

        impl From<RangeTo<$index>> for $self {
            #[inline]
            fn from(r: RangeTo<$index>) -> $self {
                <$constructor>::new(0, Some(r.end as usize), 1)
            }
        }

        impl From<RangeToInclusive<$index>> for $self {
            #[inline]
            fn from(r: RangeToInclusive<$index>) -> $self {
                let end = r.end as usize;
                <$constructor>::new(0, Some(end + 1), 1)
            }
        }
        impl From<RangeFull> for $self {
            #[inline]
            fn from(_r: RangeFull) -> $self {
                <$constructor>::new(0, None, 1)
            }
        }
    };
}
// impl_slice_variant_from_range!(Slice, Slice, usize);
impl_slice_variant_from_range!(SliceRangeInfo, SliceRangeInfo, usize);
// impl_slice_variant_from_range!(Slice, Slice, i32);

impl From<usize> for SliceRangeInfo {
    fn from(number: usize) -> Self {
        SliceRangeInfo::new(number, Some(number + 1), 1)
    }
}

impl From<(usize, usize, usize)> for SliceRangeInfo {
    fn from(tuple: (usize, usize, usize)) -> Self {
        SliceRangeInfo::new(tuple.0, Some(tuple.1), tuple.2)
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
        $(vector.push(Into::<$crate::SliceRangeInfo>::into($other_dims));)+
        vector
    }
    };
    ($($t:tt)*) => { compile_error!("Invalid syntax in s![] call. \
    Inputs can be:\
       1- Range\
       2- START ; STOP ; STEP\
       3- single usize\
    each separated by ';'") };
);

#[test]
fn test() {
    let a = SliceRangeInfo::from((0..3));
    // let b: dyn Into<Slice> = (0..4);
    println!("{:?}", a);
}
