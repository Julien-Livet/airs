use ndarray::{Array2, Axis};
use std::collections::{HashMap, HashSet};
use std::cmp::min;

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

pub fn valid_index(a: &Array2<i8>, at: (isize, isize)) -> bool {
    let (r, c) = at;
 
    r >= 0
        && c >= 0
        && (r as usize) < a.nrows()
        && (c as usize) < a.ncols()
}

pub fn neighbors(loc: (isize, isize), size: (isize, isize), diagonals: bool) -> Vec<(isize, isize)> {
    (-1..=1)
        .flat_map(|di: i8| {
            (-1..=1).filter_map(move |dj: i8| {
                if di == 0 && dj == 0 {
                    return None;
                }

                if !diagonals && di.abs() == dj.abs() {
                    return None;
                }

                let i = loc.0 as i8 + di;
                let j = loc.1 as i8 + dj;

                if i < 0 || j < 0 || i >= size.0 as i8 || j >= size.1 as i8 {
                    return None;
                }

                Some((i as isize, j as isize))
            })
        })
        .collect()
}

pub fn dot_segment(a: &Array2<i8>,  begin: (isize, isize), end: (isize, isize), value: i8, dot_step: usize) -> Array2<i8> {
    if dot_step < 1 {
        return a.clone();
    }

    let mut m = a.clone();

    let bx = begin.0 as f64;
    let by = begin.1 as f64;
    let ex = end.0 as f64;
    let ey = end.1 as f64;

    let ux = ex - bx;
    let uy = ey - by;

    let u_norm = (ux * ux + uy * uy).sqrt();

    if u_norm == 0.0 {
        return m;
    }

    let mut vx = ux / u_norm;
    let mut vy = uy / u_norm;

    let min = vx.min(vy);

    if min > f64::EPSILON {
        vx *= min;
        vy *= min;
    }

    let step = ((vx * vx + vy * vy).sqrt()) / u_norm;

    let mut i = 0usize;
    let mut t = 0.0;

    while t < 1.0 + 0.1 * step {
        if i % dot_step == 0 {
            let x = (bx + t * ux).round() as isize;
            let y = (by + t * uy).round() as isize;

            if x >= 0
                && y >= 0
                && (x as usize) < m.nrows()
                && (y as usize) < m.ncols()
            {
                m[(x as usize, y as usize)] = value;
            }
        }

        i += 1;
        t += step;
    }

    m
}

pub fn map(dst: &Array2<i8>, mapping: &HashMap<i8, i8>) -> Array2<i8> {
    let mut m = dst.clone();

    for v in m.iter_mut() {
        if let Some(&new) = mapping.get(v) {
            *v = new;
        }
    }

    m
}

pub fn fill_region(a: &Array2<i8>, region: &Vec<(isize, isize)>, x: i8) -> Array2<i8> {
    let mut b = a.clone();

    for &(i, j) in region {
        if valid_index(&b, (i, j)) {
            b[(i as usize, j as usize)] = x;
        }
    }

    b
}

pub fn region(a: &Array2<i8>, at: (isize, isize), diagonals: bool) -> Vec<(isize, isize)> {
    if !valid_index(a, at) {
        return Vec::new();
    }

    let mut visited = HashSet::new();
    let mut stack = vec![at];
    let v = a[(at.0 as usize, at.1 as usize)];
    let mut indices = Vec::new();

    while let Some(loc) = stack.pop() {
        if visited.insert(loc) {
            let cell_value = a[(loc.0 as usize, loc.1 as usize)];

            if cell_value == v {
                indices.push(loc);

                for n in neighbors(loc, (a.nrows() as isize, a.ncols() as isize), diagonals) {
                    stack.push(n);
                }
            }
        }
    }

    indices
}

pub fn paired_regions(a: &Array2<i8>, regions: &Vec<Vec<(isize, isize)> >) -> Vec<(Vec<(isize, isize)>, Vec<(isize, isize)>)> {
    let mut result = Vec::new();

    for i in 0..regions.len() {
        if regions[i].is_empty() {
            continue;
        }

        for j in (i + 1)..regions.len() {
            if regions[j].is_empty() {
                continue;
            }

            let mut b = a.clone();

            let value = *a.iter().max().unwrap() + 1;

            b = fill_region(&b, &regions[i], value);
            b = fill_region(&b, &regions[j], value);

            let r = region(&b, regions[i][0], true);

            if r.len() == regions[i].len() + regions[j].len() {
                let mut pairs: Vec<(isize, isize)> = Vec::new();
                pairs.extend(regions[i].iter().copied());
                pairs.extend(regions[j].iter().copied());

                pairs.sort_by(|a, b| {
                    if a.0 == b.0 {
                        a.1.cmp(&b.1)
                    } else {
                        a.0.cmp(&b.0)
                    }
                });

                if regions[i].contains(&pairs[0]) {
                    result.push((regions[j].clone(), regions[i].clone()));
                } else {
                    result.push((regions[i].clone(), regions[j].clone()));
                }
            }
        }
    }

    result
}

pub fn region_set(m: &Array2<i8>, diagonals: bool,) -> HashSet<Vec<(isize, isize)> > {
    let mut seen = Array2::<bool>::from_elem(m.raw_dim(), false);
    let mut result = HashSet::new();

    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            if seen[(i, j)] {
                continue;
            }

            let mut r = region(m, (i as isize, j as isize), diagonals);

            for &(x, y) in &r {
                seen[(x as usize, y as usize)] = true;
            }

            r.sort();
            result.insert(r);
        }
    }

    result
}
