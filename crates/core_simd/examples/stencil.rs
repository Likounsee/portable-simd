#![feature(portable_simd)]

//! Two-dimensional five-point heat stencil using portable SIMD.
//! Run with: cargo test -p core_simd --example stencil

use core_simd::simd::prelude::*;

const LANES: usize = 4;

/// Apply one heat-diffusion step to a row-major image.
///
/// Boundary cells are kept unchanged. Four adjacent interior cells are
/// updated together, with a scalar tail for widths not divisible by four.
pub fn stencil_2d(input: &[f32], width: usize, height: usize) -> Vec<f32> {
    assert_eq!(
        input.len(),
        width
            .checked_mul(height)
            .expect("image dimensions overflow"),
        "input length must match image dimensions",
    );

    let mut output = input.to_vec();
    if width < 3 || height < 3 {
        return output;
    }

    let quarter = f32x4::splat(4.0);
    let scale = f32x4::splat(0.125);

    for row in 1..height - 1 {
        let mut column = 1;
        while column < width - 1 {
            let columns = core::array::from_fn(|lane| (column + lane).min(width - 2));
            let center = f32x4::from_array(columns.map(|col| input[row * width + col]));
            let up = f32x4::from_array(columns.map(|col| input[(row - 1) * width + col]));
            let down = f32x4::from_array(columns.map(|col| input[(row + 1) * width + col]));
            let left = f32x4::from_array(columns.map(|col| input[row * width + col - 1]));
            let right = f32x4::from_array(columns.map(|col| input[row * width + col + 1]));

            let values = ((center * quarter + up + down + left + right) * scale).to_array();
            for (lane, value) in values.into_iter().enumerate() {
                let col = column + lane;
                if col < width - 1 {
                    output[row * width + col] = value;
                }
            }
            column += LANES;
        }
    }

    output
}

#[cfg(test)]
fn stencil_2d_scalar(input: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut output = input.to_vec();
    if width < 3 || height < 3 {
        return output;
    }

    for row in 1..height - 1 {
        for column in 1..width - 1 {
            let index = row * width + column;
            output[index] = (input[index] * 4.0
                + input[index - width]
                + input[index + width]
                + input[index - 1]
                + input[index + 1])
                * 0.125;
        }
    }
    output
}

fn main() {
    let width = 1024;
    let height = 768;
    let input = (0..width * height)
        .map(|index| ((index * 17 % 251) as f32) / 251.0)
        .collect::<Vec<_>>();
    let output = stencil_2d(&input, width, height);
    println!("Updated {} grid cells", output.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_matches_scalar_for_varied_dimensions() {
        for (width, height) in [(0, 0), (1, 8), (2, 5), (3, 3), (7, 5), (8, 6), (17, 9)] {
            let input = (0..width * height)
                .map(|index| ((index * 31 % 97) as f32) / 97.0)
                .collect::<Vec<_>>();
            assert_eq!(
                stencil_2d(&input, width, height),
                stencil_2d_scalar(&input, width, height),
                "grid dimensions {width}x{height}",
            );
        }
    }

    #[test]
    fn boundary_cells_are_preserved() {
        let width = 7;
        let height = 6;
        let input = (0..width * height).map(|n| n as f32).collect::<Vec<_>>();
        let output = stencil_2d(&input, width, height);

        for row in 0..height {
            assert_eq!(output[row * width], input[row * width]);
            assert_eq!(
                output[row * width + width - 1],
                input[row * width + width - 1]
            );
        }
        for column in 0..width {
            assert_eq!(output[column], input[column]);
            assert_eq!(
                output[(height - 1) * width + column],
                input[(height - 1) * width + column],
            );
        }
    }

    #[test]
    #[should_panic(expected = "input length must match image dimensions")]
    fn rejects_inconsistent_dimensions() {
        let _ = stencil_2d(&[0.0; 8], 3, 3);
    }
}
