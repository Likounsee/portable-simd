#![feature(portable_simd)]

//! SIMD Mandelbrot renderer, ported as a practical example for issue #110.
//! Run with: cargo test -p core_simd --example mandelbrot

use core_simd::simd::prelude::*;

const LANES: usize = 4;
const MAX_ITERATIONS: u32 = 96;

/// Returns the number of iterations needed for each point to escape.
///
/// Points are processed four at a time. Once a lane escapes, its state is
/// frozen while the remaining lanes continue iterating.
pub fn escape_counts_simd(points_x: [f32; LANES], point_y: f32) -> [u32; LANES] {
    let x = f32x4::from_array(points_x);
    let y = f32x4::splat(point_y);
    let mut zx = f32x4::splat(0.0);
    let mut zy = f32x4::splat(0.0);
    let mut counts = u32x4::splat(0);
    let mut active = Mask::<i32, LANES>::splat(true);

    for _ in 0..MAX_ITERATIONS {
        // Count every iteration entered by a live lane, including the one
        // which first crosses the escape radius.
        counts += active.select(u32x4::splat(1), u32x4::splat(0));

        let next_zx = zx * zx - zy * zy + x;
        let next_zy = f32x4::splat(2.0) * zx * zy + y;
        let magnitude_squared = next_zx * next_zx + next_zy * next_zy;
        let still_bounded = magnitude_squared.simd_le(f32x4::splat(4.0));
        active &= still_bounded;

        zx = active.select(next_zx, zx);
        zy = active.select(next_zy, zy);
    }

    // Lanes which never escaped are represented by MAX_ITERATIONS.
    counts.to_array()
}

/// Render a grayscale Mandelbrot image into a row-major byte buffer.
pub fn render(width: usize, height: usize) -> Vec<u8> {
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let mut pixels = vec![0; width * height];
    for row in 0..height {
        let y = -1.2 + 2.4 * row as f32 / height.saturating_sub(1).max(1) as f32;
        for start in (0..width).step_by(LANES) {
            let xs = core::array::from_fn(|lane| {
                let column = (start + lane).min(width - 1);
                -2.0 + 3.0 * column as f32 / width.saturating_sub(1).max(1) as f32
            });
            let counts = escape_counts_simd(xs, y);
            for (lane, count) in counts.iter().copied().enumerate() {
                let column = start + lane;
                if column < width {
                    let shade = 255 - (count * 255 / MAX_ITERATIONS) as u8;
                    pixels[row * width + column] = shade;
                }
            }
        }
    }
    pixels
}

fn main() {
    let image = render(640, 400);
    println!("Rendered {} Mandelbrot pixels", image.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn escape_count_scalar(x: f32, y: f32) -> u32 {
        let (mut zx, mut zy) = (0.0_f32, 0.0_f32);
        for iteration in 0..MAX_ITERATIONS {
            let next_zx = zx * zx - zy * zy + x;
            let next_zy = 2.0 * zx * zy + y;
            zx = next_zx;
            zy = next_zy;
            if zx * zx + zy * zy > 4.0 {
                return iteration + 1;
            }
        }
        MAX_ITERATIONS
    }

    #[test]
    fn simd_escape_counts_match_scalar_reference() {
        for y_step in 0..17 {
            let y = -1.2 + 2.4 * y_step as f32 / 16.0;
            let xs = [-2.0, -0.75, -0.25, 1.0];
            let actual = escape_counts_simd(xs, y);
            let expected = xs.map(|x| escape_count_scalar(x, y));
            assert_eq!(actual, expected, "row coordinate {y}");
        }
    }

    #[test]
    fn renderer_handles_empty_single_and_non_multiple_dimensions() {
        assert!(render(0, 10).is_empty());
        assert!(render(10, 0).is_empty());
        assert_eq!(render(1, 1).len(), 1);
        assert_eq!(render(7, 5).len(), 35);
        assert_eq!(render(7, 5), render(7, 5));
    }

    #[test]
    fn known_interior_and_exterior_points_are_classified() {
        let counts = escape_counts_simd([0.0, -0.75, 1.0, 2.0], 0.0);
        assert_eq!(counts[0], MAX_ITERATIONS);
        assert_eq!(counts[1], MAX_ITERATIONS);
        assert!(counts[2] < MAX_ITERATIONS);
        assert_eq!(counts[3], 2);
    }
}
