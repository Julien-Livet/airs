use ndarray::Array2;

use crate::airs;

pub fn fliplr(x: &[Array2<i8>]) -> Vec<Array2<i8> > {
    x.iter().map(airs::fliplr).collect()
}

pub fn flipud(x: &[Array2<i8>]) -> Vec<Array2<i8> > {
    x.iter().map(airs::flipud).collect()
}
