#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]


use oxiblas::prelude::*;

mod matrixops;
use crate::matrixops::*;


/// Example: `vct_y = mat2 * mat1 * vct_x` with abstract `mat1` and `mat2`
fn mult_mult_vct(
	mat2: &dyn MatrixOps1<f64>,
	mat1: &dyn MatrixOps1<f64>,
	vct_x: &[f64],
	vct_y: &mut[f64],
	vct_wsp: &mut[f64], // workspace with same size as vct_x and vct_y
) {
    assert_eq!(vct_x.len(), vct_wsp.len(), "wrong size");
    assert_eq!(vct_y.len(), vct_wsp.len(), "wrong size");
	mat1.mult_vct(vct_x, vct_wsp);
	mat2.mult_vct(vct_wsp, vct_y);
}


fn main() {
	let mat_a = Mat::from_rows( &[
	   	&[   1.0,   2.0 ],
	    &[ 100.0, 200.0 ],
	   	]);
	let mat_b = DiagonalMatrixOps {
	    	diag: &[ 1.0, 30.0 ],
		};
	let vct_x = &[ 10.0, 0.1];
	let vct_y = &mut[ 0.0; 2];
	let vct_wsp = &mut[ 0.0; 2];

	// vct_y = mat_a * mat_b * vct_x:
	mult_mult_vct( &mat_a.as_ref(), &mat_b, vct_x, vct_y, vct_wsp );
	
	// vct_y = mat_b * mat_a * vct_x:
	mult_mult_vct( &mat_b, &mat_a.as_ref(), vct_x, vct_y, vct_wsp );

    println!("Done!");
}
