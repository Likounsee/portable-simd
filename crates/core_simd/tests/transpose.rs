#![feature(portable_simd)]

use core_simd::simd::{Simd, transpose};

#[test]
fn transpose_rectangular_rows_to_columns() {
    let rows = [
        Simd::from_array([1_i32, 2, 3, 4]),
        Simd::from_array([5_i32, 6, 7, 8]),
    ];

    let columns = transpose(rows);
    assert_eq!(
        columns.map(|column| column.to_array()),
        [[1, 5], [2, 6], [3, 7], [4, 8]],
    );
}

#[test]
fn transpose_square_matrix() {
    let rows = [
        Simd::from_array([0_u32, 1, 2, 3]),
        Simd::from_array([4_u32, 5, 6, 7]),
        Simd::from_array([8_u32, 9, 10, 11]),
        Simd::from_array([12_u32, 13, 14, 15]),
    ];

    let columns = transpose(rows);
    assert_eq!(
        columns.map(|column| column.to_array()),
        [
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15],
        ],
    );
}

#[test]
fn transpose_one_lane_dimension() {
    let rows = [
        Simd::from_array([10_i16]),
        Simd::from_array([20_i16]),
        Simd::from_array([30_i16]),
    ];

    let [column] = transpose(rows);
    assert_eq!(column.to_array(), [10, 20, 30]);

    let [row] = transpose([Simd::from_array([7_i16, 8, 9])]);
    assert_eq!(row.to_array(), [7, 8, 9]);
}

#[test]
fn transpose_floating_point_vectors() {
    let rows = [
        Simd::from_array([1.25_f32, 2.5]),
        Simd::from_array([3.75_f32, 5.0]),
        Simd::from_array([6.25_f32, 7.5]),
    ];

    let columns = transpose(rows);
    assert_eq!(
        columns.map(|column| column.to_array()),
        [[1.25, 3.75, 6.25], [2.5, 5.0, 7.5]],
    );
}
