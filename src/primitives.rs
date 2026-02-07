use ndarray::Array2;
use std::collections::HashMap;

use crate::airs;

pub fn fliplr(x: &[Array2<i8>]) -> Vec<Array2<i8> > {
    x.iter().map(airs::fliplr).collect()
}

pub fn flipud(x: &[Array2<i8>]) -> Vec<Array2<i8> > {
    x.iter().map(airs::flipud).collect()
}

pub fn map(x: &Vec<Array2<i8> >, mapping: &HashMap<i8, i8>) -> Vec<Array2<i8> > {
    x.iter()
        .map(|y| airs::map(y, mapping))
        .collect()
}

pub fn infer_color_mapping(pairs: &Vec<(Array2<i8>, Array2<i8>)>) -> HashMap<i8, i8> {
    let mut mapping: HashMap<i8, i8> = HashMap::new();

    for (input, output) in pairs {
        let (h, w) = input.dim();

        for i in 0..h {
            for j in 0..w {
                let a = input[(i, j)];
                let b = output[(i, j)];

                if let Some(&existing) = mapping.get(&a) {
                    if existing != b {
                        return mapping;
                    }
                } else {
                    mapping.insert(a, b);
                }
            }
        }
    }

    mapping
}

pub fn same_element(pairs: &Vec<Vec<((isize, isize), (isize, isize))> >, first: bool) -> Vec<Vec<((isize, isize), (isize, isize))>> {
    let mut result = Vec::new();

    for group in pairs {
        let mut filtered = Vec::new();

        for &p in group {
            if first {
                if p.0 .0 == p.1 .0 {
                    filtered.push(p);
                }
            } else {
                if p.0 .1 == p.1 .1 {
                    filtered.push(p);
                }
            }
        }

        result.push(filtered);
    }

    result
}

pub fn segments(dst: &Vec<Array2<i8>>, pairs: &Vec<Vec<((isize, isize), (isize, isize))> >, value: i8, start: bool, finish: bool) -> Vec<Array2<i8>> {
    if dst.len() != pairs.len() {
        return Vec::new();
    }

    let mut result = Vec::new();

    for i in 0..dst.len() {
        let mut m = dst[i].clone();

        for &(p0, p1) in &pairs[i] {
            if airs::valid_index(&m, p0) && airs::valid_index(&m, p1) {
                let s = m[(p0.0 as usize, p0.1 as usize)];
                let f = m[(p1.0 as usize, p1.1 as usize)];

                m = airs::dot_segment(&m, p0, p1, value, 1);

                if start {
                    m[(p0.0 as usize, p0.1 as usize)] = s;
                }

                if finish {
                    m[(p1.0 as usize, p1.1 as usize)] = f;
                }
            }
        }

        result.push(m);
    }

    result
}
