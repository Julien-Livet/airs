use ndarray::{Array2, Axis};
use std::cmp::min;

pub fn cartesian_product<T: Clone>(lists: Vec<Vec<T> >) -> Vec<Vec<T> > {
    let mut result: Vec<Vec<T> > = vec![vec![]];

    for list in lists {
        let mut tmp = Vec::new();

        for prefix in result.drain(..) {
            for element in &list {
                let mut n = prefix.clone();
                n.push(element.clone());
                tmp.push(n);
            }
        }

        result = tmp;
    }

    result
}

#[inline]
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    
    let n = a_bytes.len();
    let m = b_bytes.len();

    let mut dp: Vec<usize> = (0..=m).collect();

    for i in 1..=n {
        let mut prev = dp[0];
        dp[0] = i;

        for j in 1..=m {
            let cur = dp[j];

            if a_bytes[i - 1] == b_bytes[j - 1] {
                dp[j] = prev;
            } else {
                dp[j] = 1 + min(prev, min(dp[j], dp[j - 1]));
            }

            prev = cur;
        }
    }

    dp[m]
}

pub fn fliplr(x: &Array2<i8>) -> Array2<i8> {
    let mut y = x.clone();
    y.invert_axis(Axis(1));
    y
}

pub fn flipud(x: &Array2<i8>) -> Array2<i8> {
    let mut y = x.clone();
    y.invert_axis(Axis(0));
    y
}
