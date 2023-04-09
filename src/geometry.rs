extern crate nalgebra as na;
type Matrix3N = na::Matrix3xX<f64>;
type Vector3 = na::Vector3<f64>;

pub struct Plane {
    pub normal: Vector3,
    pub offset: Vector3
}

impl Plane {
    pub fn fit_matrix(mut cloud: Matrix3N) -> Plane {
        // Remove centroid
        let centroid: Vector3 = cloud.column_sum() / (cloud.ncols() as f64);
        for mut col in cloud.column_iter_mut() {
            col -= centroid;
        }

        // Normal is left singular vector of least singular value
        let u = cloud.svd(true, true).u.expect("SVD worked and computed u");
        // NOTE: guaranteed sorted descending, take last column of u
        let normal: Vector3 = u.column(u.ncols() - 1).into();
        assert!((normal.norm() - 1.0).abs() < 1e-6);
        Plane {normal, offset: centroid}
    }

    pub fn fit_matrix_points(cloud: &Matrix3N, vertices: &[usize]) -> Plane {
        let n = vertices.len();
        let mut plane_vertices = Matrix3N::zeros(n);
        for (i, &v) in vertices.iter().enumerate() {
            plane_vertices.set_column(i, &cloud.column(v));
        }

        Self::fit_matrix(plane_vertices)
    }

    pub fn signed_distance<S>(&self, point: &na::Matrix<f64, na::Const<3>, na::Const<1>, S>) -> f64 where S: na::Storage<f64, na::Const<3>, na::Const<1>> {
        self.normal.dot(&(point - self.offset))
    }

    pub fn rmsd(&self, cloud: &Matrix3N, vertices: &[usize]) -> f64 {
        let sum_of_squares: f64 = vertices.iter()
            .map(|&v| self.signed_distance(&cloud.column(v)).powi(2))
            .sum();
        (sum_of_squares / (vertices.len() as f64)).sqrt()
    }

    pub fn parallel(&self, other: &Plane) -> bool {
        self.normal.cross(&other.normal).norm() < 1e-6 
            || self.normal.cross(&-other.normal).norm() < 1e-6
    }

    pub fn signed_angle<S1, S2>(
        &self, 
        a: &na::Matrix<f64, na::Const<3>, na::Const<1>, S1>,
        b: &na::Matrix<f64, na::Const<3>, na::Const<1>, S2>) -> f64 where 
        S1: na::Storage<f64, na::Const<3>, na::Const<1>>, 
        S2: na::Storage<f64, na::Const<3>, na::Const<1>> {
        a.cross(b).dot(&self.normal).atan2(a.dot(b))
    }
}


pub fn axis_perpendicular_component<S1, S2>(
    axis: &na::Matrix<f64, na::Const<3>, na::Const<1>, S1>, 
    point: &na::Matrix<f64, na::Const<3>, na::Const<1>, S2>) -> Vector3
where S1: na::Storage<f64, na::Const<3>, na::Const<1>>, 
    S2: na::Storage<f64, na::Const<3>, na::Const<1>> {
    point - (point.dot(axis)) * axis
}

pub fn axis_distance<S1, S2>(
    axis: &na::Matrix<f64, na::Const<3>, na::Const<1>, S1>, 
    point: &na::Matrix<f64, na::Const<3>, na::Const<1>, S2>) -> f64 
where S1: na::Storage<f64, na::Const<3>, na::Const<1>>, 
    S2: na::Storage<f64, na::Const<3>, na::Const<1>> {
    axis_perpendicular_component(axis, point).norm()
}

#[cfg(test)]
mod tests {
    use crate::shapes::SQUARE;
    use crate::geometry::*;

    #[test]
    fn plane() {
        let xy = Plane {
            normal: Vector3::z(),
            offset: Vector3::zeros()
        };
        assert_eq!(xy.signed_distance(&Vector3::z()), 1.0);
        assert_eq!(xy.signed_distance(&-Vector3::z()), -1.0);
        assert_eq!(xy.signed_distance(&(0.5 * Vector3::z())), 0.5);

        let xy = Plane::fit_matrix(SQUARE.coordinates.clone());
        assert_eq!(xy.signed_distance(&Vector3::z()).abs(), 1.0);
    }

}
