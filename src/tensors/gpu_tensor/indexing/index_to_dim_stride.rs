use crate::tensors::gpu_tensor::indexing::SliceRangeInfo;
use crate::DimStride;

pub fn index<'a, T: Into<SliceRangeInfo>>(original: DimStride, bounds: Vec<T>) -> DimStride{
    let bounds: Vec<SliceRangeInfo> = bounds.into_iter().map(|e| e.into()).collect();

    for b in bounds{
        println!("{:?}", b.start);
        println!("{:?}", b.step);
        println!("{:?}", b.inclusive_end);
    }

    unimplemented!()
}

#[test]
fn can_index(){

}