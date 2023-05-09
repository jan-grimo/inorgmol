use rand::prelude::SliceRandom;
use rand::Rng;
use thiserror::Error;

extern crate nalgebra as na;
type Matrix4N = na::Matrix4xX<f64>;

/// Number of spatial dimensions embedding and refinement operate in
const DIMENSIONS: usize = 4;

#[derive(Clone)]
pub struct DistanceBounds {
    /// Empty diagonal, lower bounds in lower triangle, upper above
    mat: na::DMatrix<f64>
}

impl DistanceBounds {
    pub fn new(n: usize) -> DistanceBounds {
        const DEFAULT_LOWER: f64 = 0.0;
        const DEFAULT_UPPER: f64 = 100.0;
        let mat = na::DMatrix::from_fn(n, n, |i, j| {
            match i.cmp(&j) {
                std::cmp::Ordering::Less => DEFAULT_UPPER,
                std::cmp::Ordering::Equal => 0.0,
                std::cmp::Ordering::Greater => DEFAULT_LOWER,
            }
        });
        DistanceBounds {mat}
    }

    pub fn n(&self) -> usize {
        self.mat.ncols()
    }

    pub fn order_indices(mut i: usize, mut j: usize) -> (usize, usize) {
        assert!(i != j);
        if i > j {
            std::mem::swap(&mut i, &mut j);
        }
        (i, j)
    }

    fn lower_tuple(i: usize, j: usize) -> (usize, usize) {
        debug_assert!(i < j);
        (j, i)
    }

    fn upper_tuple(i: usize, j: usize) -> (usize, usize) {
        debug_assert!(i < j);
        (i, j)
    }

    pub fn ordered_lower(&self, i: usize, j: usize) -> &f64 {
        debug_assert!(i < j);
        &self.mat[Self::lower_tuple(i, j)]
    }

    pub fn lower(&self, i: usize, j: usize) -> &f64 {
        let (i, j) = Self::order_indices(i, j);
        self.ordered_lower(i, j)
    }

    pub fn ordered_upper(&self, i: usize, j: usize) -> &f64 {
        debug_assert!(i < j);
        &self.mat[Self::upper_tuple(i, j)]
    }

    pub fn upper(&self, i: usize, j: usize) -> &f64 {
        let (i, j) = Self::order_indices(i, j);
        self.ordered_upper(i, j)
    }

    pub fn lower_upper(&self, i: usize, j: usize) -> (f64, f64) {
        let (i, j) = Self::order_indices(i, j);
        (*self.ordered_lower(i, j), *self.ordered_upper(i, j))
    }

    pub fn collapse(&mut self, i: usize, j: usize, value: f64) {
        assert!(i != j);
        self.mat[(i, j)] = value;
        self.mat[(j, i)] = value;
    }

    pub fn increase_lower_bound(&mut self, i: usize, j: usize, value: f64) -> bool {
        let (i, j) = Self::order_indices(i, j);
        if (self.mat[Self::lower_tuple(i, j)]..self.mat[Self::upper_tuple(i, j)]).contains(&value) {
            self.mat[Self::lower_tuple(i, j)] = value;
            return true;
        } 

        false
    }

    pub fn decrease_upper_bound(&mut self, i: usize, j: usize, value: f64) -> bool {
        let (i, j) = Self::order_indices(i, j);
        if self.mat[Self::upper_tuple(i, j)] >= value
            && value > self.mat[Self::lower_tuple(i, j)]
        {
            self.mat[Self::upper_tuple(i, j)] = value;
            return true;
        }

        false
    }

    /// Try to smooth the triangle inequalities with Floyd's algorithm: O(N^3)
    /// 
    /// If triangle bound inversion is encountered, returns None
    ///
    /// TODO more helpful error type
    pub fn floyd_triangle_smooth(mut self) -> Option<Self> {
        let n = self.mat.ncols();

        let idx_tuples = |a: usize, b: usize| {
            let (a, b) = Self::order_indices(a, b);
            (Self::lower_tuple(a, b), Self::upper_tuple(a, b))
        };

        for k in 0..n {
            for i in (0..(n-1)).filter(|&x| x != k) { // TODO maybe split range
                let (ik_lower, ik_upper) = idx_tuples(i, k);

                if self.mat[ik_lower] > self.mat[ik_upper] {
                    return None;
                }

                for j in ((i + 1)..n).filter(|&x| x != k) {
                    // i < j due to loop conditions
                    let ij_lower = Self::lower_tuple(i, j);
                    let ij_upper = Self::upper_tuple(i, j);

                    let (jk_lower, jk_upper) = idx_tuples(j, k);

                    // Actual algorithm begins here
                    if self.mat[ij_upper] > self.mat[ik_upper] + self.mat[jk_upper] {
                        self.mat[ij_upper] = self.mat[ik_upper] + self.mat[jk_upper];
                    }

                    if self.mat[ij_lower] < self.mat[ik_lower] - self.mat[jk_upper] {
                        self.mat[ij_lower] = self.mat[ik_lower] - self.mat[jk_upper];
                    } else if self.mat[ij_lower] < self.mat[jk_lower] - self.mat[ik_upper] {
                        self.mat[ij_lower] = self.mat[jk_lower] - self.mat[ik_upper];
                    }

                    if self.mat[ij_lower] > self.mat[ij_upper] {
                        return None;
                    }
                }
            }
        }

        Some(self)
    }

    pub fn take_matrix(self) -> na::DMatrix<f64> {
        self.mat
    }
}

pub struct DistanceMatrix {
    mat: na::DMatrix<f64>
}

/// For what subset of particles to repeat triangle inequality smoothing during distance generation
pub enum MetrizationPartiality {
    FourAtom,
    TenPercent,
    Complete
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum DistanceGeometryError {
    #[error("Impossible graph")]
    GraphImpossible
}

impl DistanceMatrix {
    pub fn try_from_distance_bounds(mut bounds: DistanceBounds, metrization: MetrizationPartiality) -> Result<DistanceMatrix, DistanceGeometryError> {
        let n = bounds.n(); 
        let mut index_order: Vec<usize> = (0..n).collect();
        let mut rng = rand::thread_rng();
        index_order.as_mut_slice().shuffle(&mut rng);

        let metrization_boundary = match metrization {
            MetrizationPartiality::FourAtom => n.min(4),
            MetrizationPartiality::TenPercent => n.min(4.max(n / 10)),
            MetrizationPartiality::Complete => n
        };

        for (count, i) in index_order.into_iter().enumerate() {
            for j in (0..n).filter(|&x| x != i) {
                let (lower, upper) = bounds.lower_upper(i, j);

                // Skip already-chosen combinations
                if lower == upper {
                    continue;
                }

                // Bound reversals during metrization are errors
                if lower > upper && count < metrization_boundary {
                    return Err(DistanceGeometryError::GraphImpossible);
                }

                bounds.collapse(i, j, rng.gen_range(lower..upper));

                if count < metrization_boundary {
                    bounds = bounds.floyd_triangle_smooth().ok_or(DistanceGeometryError::GraphImpossible)?;
                }
            }
        }
        
        Ok(DistanceMatrix {mat: bounds.take_matrix()})
    }

    pub fn n(&self) -> usize {
        self.mat.ncols()
    }

    pub fn take_matrix(self) -> na::DMatrix<f64> {
        self.mat
    }
}

pub struct MetricMatrix {
    mat: na::DMatrix<f64>
}

impl MetricMatrix {
    pub fn from_distance_matrix(distances: DistanceMatrix) -> MetricMatrix {
        let n = distances.n();
        let mut mat: na::DMatrix<f64> = na::DMatrix::zeros(n, n);

        /* We need to accomplish the following:
         *
         * Every matrix entry needs to be set according to the result of the
         * following equations:
         *
         *    D0[i]² =   (1/N) * sum_{j}(distances[i, j]²)
         *             - (1/(N²)) * sum_{j < k}(distances[j, k]²)
         *
         * (The second term is independent of i and can be precalculated!)
         *
         *    [i, j] = ( D0[i]² + D0[j]² - distances[i, j]² ) / 2
         *
         * On the diagonal, where i == j:
         *    [i, i] = ( D0[i]² + D0[i]² - distances[i, i]² ) / 2 = D0[i]²
         *                ^--------^      ^-------------^
         *                   equal               =0
         *
         * So, we can store all of D0 immediately on the diagonal and
         * perform the remaining transformation afterwards.
         */
        let square_distances = distances.take_matrix().map(|v| v.powi(2));
        let second_term = square_distances.sum() / (2 * n * n) as f64;

        // Write diagonal
        for i in 0..n {
            let first_term = square_distances.column(i).sum() / n as f64;
            mat[(i, i)] = first_term - second_term;
        }

        // Write off-diagonal elements
        for i in 0..n {
            for j in (i + 1)..n {
                let value = (mat[(i, i)] + mat[(j, j)] - square_distances[(i, j)]) / 2.0;
                mat[(i, j)] = value;
                mat[(j, i)] = value;
            }
        }

        MetricMatrix {mat}
    }

    pub fn embed(self) -> Matrix4N {
        let n = self.mat.ncols();
        let decomposition = self.mat.symmetric_eigen();

        let mut usable_eigenvalue_indices: Vec<(usize, &f64)> = decomposition.eigenvalues.iter()
            .enumerate()
            .filter(|(_, &eigenvalue)| eigenvalue > 0.0)
            .collect();

        usable_eigenvalue_indices.sort_by(|(_, &eigenvalue_a), (_, &eigenvalue_b)| eigenvalue_a.partial_cmp(&eigenvalue_b).expect("No NaNs"));

        // set up the coordinate matrix by multiplying up to four positive
        // eigenvalues by their respective eigenvectors
        let mut xyzw: Matrix4N = Matrix4N::zeros(n);
        for (i, (eigenvalue_idx, _)) in usable_eigenvalue_indices.iter().rev().enumerate().take(DIMENSIONS) {
            xyzw.set_row(i, &(decomposition.eigenvalues[*eigenvalue_idx].sqrt() * decomposition.eigenvectors.column(*eigenvalue_idx).transpose()));
        }

        xyzw
    }
}

pub mod refinement;
pub mod modeling;
