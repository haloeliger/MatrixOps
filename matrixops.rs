//! Traits for the multiplication of vectors and matrices by abstract matrices 
//! (i.e., linear operators),
//! with implementations for generic OxiBLAS matrices. 
//!

//! Only the operators are abstract; 
//! vector operands and results are slices, matrix operands and results are OxiBLAS matrices. 
//!
//! Writing an algorithm in terms of these abstract operators
//! enables users of that algorithm to provide their own specialized methods
//! for the actual operator. 
//!
//! Using such an algorithm requires the implementation of the traits 
//! `MatrixOps1` or `MatrixOps2` or both (= `MatrixOps`).
//! As examples, such implementations are provided for generic matrices and for diagonal matrices.
//!

//! Trait `MatrixOps1` requires the following methods:
//!
//! ```text
//!	nrows:                  number of rows
//!	ncols:                  number of columns
//! mult_vct:               vct_c =  self * vct_b
//! mult_vct_addto:         vct_c += self * vct_b
//! mult_vct_scaled_addto:  vct_c += alpha * self * mat_b + beta * vct_c
//! tmult_vct:              vct_c =  self_t * vct_b
//!	tmult_vct_addto:        vct_c += self_t * vct_b
//!	tmult_vct_scaled_addto: vct_c += alpha * self_t * mat_b + beta * vct_c
//!
//!	where `vct_` denotes a vector (= a slice of scalars), `_t` means transposed
//! and `self` is the abstract matrix.
//! ```
//!

//! Trait `MatrixOps2` requires the following methods:
//! ```text
//!	mult_mat:                mat_c =  self * mat_b
//! mult_mat_addto:          mat_c += self * mat_b
//!	mult_mat_scaled_addto:   mat_c =  alpha * self * mat_b + beta * mat_c
//!	tmult_mat:               mat_c =  self_t * mat_b
//!	tmult_mat_addto:         mat_c += self_t * mat_b
//!	tmult_mat_scaled_addto:  mat_c =  alpha * self_t * mat_b + beta * mat_c
//!
//! where `mat_` denotes an OxiBLAS matrix, `_t` means transposed, 
//! and `self` is the abstract matrix.
//! ```
//! \

//!	**Example:** `vct_c = mat2 * mat1 * vct_b` with abstract `mat1` and `mat2`:
//!
//! ```
//! use oxiblas::prelude::*;
//! use matrixops::*;
//!
//!	fn mult_mult_vct(
//!		mat1: &dyn MatrixOps1<f64>,
//!		mat2: &dyn MatrixOps1<f64>,
//!		vct_b: &[f64],
//!		vct_c: &mut[f64],
//!		wsp:   &mut[f64], // workspace with same size as vct_b and vct_c
//!	) {
//!		mat1.mult_vct(vct_b, wsp);
//!		mat2.mult_vct(wsp, vct_c);
//!	}
//!
//! let mat_a = Mat::from_rows( &[
//!    	&[   1.0,   2.0 ],
//!     &[ 100.0, 200.0 ],
//!    	]);
//!
//!	let mat_b = DiagonalMatrixOps {
//!     	diag: &[ 1.0, 30.0 ],
//!    	};
//!
//!	let vct_b = &[ 10.0, 0.1];
//!	let vct_c = &mut[ 0.0; 2];
//!	let wsp =   &mut[ 0.0; 2];
//!
//!	// vct_c = mat_a * mat_b * vct_b:
//!	mult_mult_vct( &mat_b, &mat_a.as_ref(), vct_b, vct_c, wsp );
//!
//!	// vct_c = mat_b * mat_a * vct_b:
//!	mult_mult_vct( &mat_a.as_ref(), &mat_b, vct_b, vct_c, wsp );
//! ```


use oxiblas::prelude::*;
// => `Scalar` is `oxiblas::Scalar`


pub trait MatrixOps1<T: Scalar> {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn mult_vct(&self, vct_b: &[T], vct_c: &mut[T])
    {	// dumb default:
    	self.mult_vct_scaled_addto(T::one(), vct_b, T::zero(), vct_c);
    }
    fn mult_vct_addto(&self, vct_b: &[T], vct_c: &mut[T])
    {	// dumb default:
    	self.mult_vct_scaled_addto(T::one(), vct_b, T::one(), vct_c);
    }
    fn mult_vct_scaled_addto(&self, alpha: T, vct_b: &[T], beta: T, vct_c: &mut[T]);

    fn tmult_vct(&self, vct_b: &[T], vct_c: &mut[T])
    {	// dumb default:
    	self.tmult_vct_scaled_addto(T::one(), vct_b, T::zero(), vct_c);
    }
    fn tmult_vct_addto(&self, vct_b: &[T], vct_c: &mut[T])
    {	// dumb default:
    	self.tmult_vct_scaled_addto(T::one(), vct_b, T::one(), vct_c);
    }
    fn tmult_vct_scaled_addto(&self, alpha: T, vct_b: &[T], beta: T, vct_c: &mut[T]);
}


pub trait MatrixOps2<T: Scalar> {
    fn mult_mat(&self, mat_b: &MatRef<T>, mat_c: &mut MatMut<T>)
    {	// dumb default:
    	self.mult_mat_scaled_addto(T::one(), mat_b, T::zero(), mat_c);
    }
    fn mult_mat_addto(&self, mat_b: &MatRef<T>, mat_c: &mut MatMut<T>)
    {	// dumb default:
    	self.mult_mat_scaled_addto(T::one(), mat_b, T::one(), mat_c);
    }
    fn mult_mat_scaled_addto(&self, alpha: T, mat_b: &MatRef<T>, beta: T, mat_c: &mut MatMut<T>);

    fn tmult_mat(&self, mat_b: &MatRef<T>, mat_c: &mut MatMut<T>)
    {	// dumb default:
    	self.tmult_mat_scaled_addto(T::one(), mat_b, T::zero(), mat_c);
    }
    fn tmult_mat_addto(&self, mat_b: &MatRef<T>, mat_c: &mut MatMut<T>)
    {	// dumb default:
    	self.tmult_mat_scaled_addto(T::one(), mat_b, T::one(), mat_c);
    }
    fn tmult_mat_scaled_addto(&self, alpha: T, mat_b: &MatRef<T>, beta: T, mat_c: &mut MatMut<T>);
}


pub trait MatrixOps<T: Scalar>: MatrixOps1<T> + MatrixOps2<T> {}
impl<Mops, T: Scalar> MatrixOps<T> for Mops where Mops: MatrixOps1<T> + MatrixOps2<T> {}


/// diagonal matrix as MatrixOp

#[derive(Debug, Clone, PartialEq)]
pub struct DiagonalMatrixOps<'a, T: Scalar> {
    pub diag: &'a [T]
}


// ---------------------------------------------------------------------------------
/// Implementation for generic OxiBLAS matrices (views)

impl<'a, T> MatrixOps1<T> for MatRef<'a, T> 
	where T: Field + GemmKernel + bytemuck::Zeroable
{
    fn nrows(&self) -> usize
    {
        self.nrows()
    }

    fn ncols(&self) -> usize
    {
        self.ncols()
    }

    fn mult_vct_scaled_addto(&self, alpha: T, vct_b: &[T], beta: T, vct_c: &mut[T])
    {
        gemv(
            GemvTrans::NoTrans,
            alpha,
            *self,
            vct_b,
            beta,
            vct_c
        );
    }

    fn tmult_vct_scaled_addto(&self, alpha: T, vct_b: &[T], beta: T, vct_c: &mut[T])
    {
        gemv(
            GemvTrans::Trans,
            alpha,
            *self,
            vct_b,
            beta,
            vct_c
        );
    }
}


impl<'a, T> MatrixOps2<T> for MatRef<'a, T> 
	where T: Field + GemmKernel + bytemuck::Zeroable
{
    fn mult_mat_scaled_addto(&self, alpha: T, mat_b: &MatRef<T>, beta: T, mat_c: &mut MatMut<T>)
    {
        gemm(
            alpha, 
            *self, 
            *mat_b, 
            beta, 
            (*mat_c).rb_mut()
        );
    }

    fn tmult_mat_scaled_addto(&self, alpha: T, mat_b: &MatRef<T>, beta: T, mat_c: &mut MatMut<T>)
    {
        gemm(
            alpha, 
            (*self).transpose(), 
            *mat_b, 
            beta, 
            (*mat_c).rb_mut()
        );
    }
}


// ---------------------------------------------------------------------------------
// Implementation for diagonal matrices

impl<'a, T: Scalar> MatrixOps1<T> for DiagonalMatrixOps<'a, T> 
{
    fn nrows(&self) -> usize
    {
        self.diag.len()
    }

    fn ncols(&self) -> usize
    {
        self.diag.len()
    }
    
    fn mult_vct(&self, vct_b: &[T], vct_c: &mut [T])
    {
        assert_eq!(self.diag.len(), vct_b.len());
        assert_eq!(self.diag.len(), vct_c.len());
		self.diag.iter().zip(vct_b).zip(vct_c).for_each( |((a,b),c)| *c = (*a) * (*b) );
    }

    fn mult_vct_addto(&self, vct_b: &[T], vct_c: &mut [T])
    {
        assert_eq!(self.diag.len(), vct_b.len());
        assert_eq!(self.diag.len(), vct_c.len());
    	self.diag.iter().zip(vct_b).zip(vct_c).for_each( |((a,b),c)| *c += (*a) * (*b) );
    }

    fn mult_vct_scaled_addto(&self, alpha: T, vct_b: &[T], beta: T, vct_c: &mut[T])
    {
        assert_eq!(self.diag.len(), vct_b.len());
        assert_eq!(self.diag.len(), vct_c.len());
    	self.diag.iter().zip(vct_b).zip(vct_c).for_each( |((a,b),c)| *c = alpha * (*a) * (*b) + beta * (*c));
    }

    fn tmult_vct(&self, vct_b: &[T], vct_c: &mut [T])
    {
    	self.mult_vct(vct_b, vct_c);
    }

    fn tmult_vct_addto(&self, vct_b: &[T], vct_c: &mut [T])
    {
    	self.mult_vct_addto(vct_b, vct_c);
    }

    fn tmult_vct_scaled_addto(&self, alpha: T, vct_b: &[T], beta: T, vct_c: &mut[T])
    {
    	self.mult_vct_scaled_addto(alpha, vct_b, beta, vct_c);
    }
}


impl<'a, T: Scalar> MatrixOps2<T> for DiagonalMatrixOps<'a, T> 
{
    fn mult_mat(&self, mat_b: &MatRef<T>, mat_c: &mut MatMut<T>)
    {
        let nrows = mat_b.nrows();
        let ncols = mat_b.ncols();
        assert_eq!(self.diag.len(), nrows);
        assert_eq!(nrows, mat_c.nrows());
        assert_eq!(ncols, mat_c.ncols());
        for m in 0..nrows {
            let a = self.diag[m];
            /*  // this works, of course:
            for n in 0..ncols {
                mat_c[ (m, n) ] = a * mat_b[ (m, n) ];
            }
            */
            let b_row = mat_b.row(m);
            let mut c_row = mat_c.rb_mut().row_mut(m);
            //b_row.iter().zip( c_row ).for_each( |(_b,_c)| _c = a*_b );   // b_row.iter() does not exist
            for n in 0..ncols {
                c_row[(0,n)] = a * b_row[(0,n)];
            }
        }
    }

    fn mult_mat_addto(&self, mat_b: &MatRef<T>, mat_c: &mut MatMut<T>)
    {
        let nrows = mat_b.nrows();
        let ncols = mat_b.ncols();
        assert_eq!(self.diag.len(), nrows);
        assert_eq!(nrows, mat_c.nrows());
        assert_eq!(ncols, mat_c.ncols());
        for m in 0..nrows {
            let a = self.diag[m];
            let b_row = mat_b.row(m);
            let mut c_row = mat_c.rb_mut().row_mut(m);
            for n in 0..ncols {
                c_row[(0,n)] += a * b_row[(0,n)];
            }
        }
    }

    fn mult_mat_scaled_addto(&self, alpha: T, mat_b: &MatRef<T>, beta: T, mat_c: &mut MatMut<T>)
	{
        let nrows = mat_b.nrows();
        let ncols = mat_b.ncols();
        assert_eq!(self.diag.len(), nrows);
        assert_eq!(nrows, mat_c.nrows());
        assert_eq!(ncols, mat_c.ncols());
        for m in 0..nrows {
            let a = self.diag[m] * alpha;
            let b_row = mat_b.row(m);
            let mut c_row = mat_c.rb_mut().row_mut(m);
            for n in 0..ncols {
                c_row[(0,n)] = a * b_row[(0,n)] + beta * c_row[(0,n)];
            }
        }
	}

    fn tmult_mat(&self, mat_b: &MatRef<T>, mat_c: &mut MatMut<T>)
    {
        self.mult_mat(mat_b, &mut mat_c.rb_mut());
    }

    fn tmult_mat_addto(&self, mat_b: &MatRef<T>, mat_c: &mut MatMut<T>)
    {
        self.mult_mat_addto(mat_b, &mut mat_c.rb_mut());
    }

    fn tmult_mat_scaled_addto(&self, alpha: T, mat_b: &MatRef<T>, beta: T, mat_c: &mut MatMut<T>)
	{
		self.mult_mat_scaled_addto(alpha, mat_b, beta, &mut mat_c.rb_mut());
	}
}


// -------------------------------------------------------------------------------------
// testing:

#[cfg(test)]
mod tests {
    use super::*;
    type Float = f64;


fn test_eq(a: &MatRef<Float>, b: &MatRef<Float>, tolerance: Float) -> bool
{
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    for m in 0..a.nrows() {
        for n in 0..a.ncols() {
            let diff = a[(m,n)] - b[(m,n)];
            if diff.abs() > tolerance { return false; }
        }
    }
    return true;
}

fn test_mult_vct(mat: &dyn MatrixOps1<Float>, vct_b: &[Float], vct_c: &mut[Float], shouldbe: &[Float])
{
    println!("mat.mult_vct(vct_b, vct_c) results in ");
    mat.mult_vct(vct_b, vct_c);
    println!("  vct_c = {:?}", vct_c);
    assert_eq!(vct_c, shouldbe, "test_mult_vct() failed");
}

fn test_mult_vct_addto(mat: &dyn MatrixOps1<Float>, vct_b: &[Float], vct_c: &mut[Float], shouldbe: &[Float])
{
    println!("mat.mult_vct_addto(vct_b, vct_c) results in ");
    mat.mult_vct_addto(vct_b, vct_c);
    println!("  vct_c = {:?}", vct_c);
    assert_eq!(vct_c, shouldbe, "test_mult_vct_addto() failed");
}

fn test_mult_vct_scaled_addto(mat: &dyn MatrixOps1<Float>, alpha: Float, vct_b: &[Float], beta: Float, vct_c: &mut[Float], shouldbe: &[Float])
{
    println!("mat.mult_vct_scaled_addto(alpha, vct_b, beta, vct_c) results in ");
    mat.mult_vct_scaled_addto(alpha, vct_b, beta, vct_c);
    println!("  vct_c = {:?}", vct_c);
    assert_eq!(vct_c, shouldbe, "test_mult_vct_addto() failed");
}


fn test_tmult_vct(mat: &dyn MatrixOps1<Float>, vct_b: &[Float], vct_c: &mut[Float], shouldbe: &[Float])
{
    println!("mat.tmult_vct(vct_b, vct_c) results in ");
    mat.tmult_vct(vct_b, vct_c);
    println!("  vct_c = {:?}", vct_c);
    assert_eq!(vct_c, shouldbe, "test_tmult_vct() failed");
}

fn test_tmult_vct_addto(mat: &dyn MatrixOps1<Float>, vct_b: &[Float], vct_c: &mut[Float], shouldbe: &[Float])
{
    println!("mat.tmult_vct_addto(vct_b, vct_c) results in ");
    mat.tmult_vct_addto(vct_b, vct_c);
    println!("  vct_c = {:?}", vct_c);
    assert_eq!(vct_c, shouldbe, "test_tmult_vct_addto() failed");
}

fn test_tmult_vct_scaled_addto(mat: &dyn MatrixOps1<Float>, alpha: Float, vct_b: &[Float], beta: Float, vct_c: &mut[Float], shouldbe: &[Float])
{
    println!("mat.tmult_vct_scaled_addto(alpha, vct_b, beta, vct_c) results in ");
    mat.tmult_vct_scaled_addto(alpha, vct_b, beta, vct_c);
    println!("  vct_c = {:?}", vct_c);
    assert_eq!(vct_c, shouldbe, "test_mult_vct_addto() failed");
}


fn test_mult_mat(mat: &dyn MatrixOps2<Float>, mat_b: &MatRef<Float>, mat_c: &mut MatMut<Float>, shouldbe: &MatRef<Float>)
{
    println!("mat.mult_mat(mat_b, mat_c) results in ");
    mat.mult_mat(mat_b, mat_c);
    println!("  mat_c = {:?}", mat_c);
    if !test_eq(&mat_c.rb(), shouldbe, 0.0) { panic!("test_mult_mat() failed"); }
}

fn test_mult_mat_addto(mat: &dyn MatrixOps2<Float>, mat_b: &MatRef<Float>, mat_c: &mut MatMut<Float>, shouldbe: &MatRef<Float>)
{
    println!("mat.mult_mat_addto(mat_b, mat_c) results in ");
    mat.mult_mat_addto(mat_b, mat_c);
    println!("  mat_c = {:?}", mat_c);
    if !test_eq(&mat_c.rb(), shouldbe, 0.0) { panic!("test_mult_mat_addto() failed"); }
}

fn test_mult_mat_scaled_addto(mat: &dyn MatrixOps2<Float>, alpha:Float, mat_b: &MatRef<Float>, beta: Float, mat_c: &mut MatMut<Float>, shouldbe: &MatRef<Float>)
{
    println!("alpha = {}, beta = {}", alpha, beta);
    println!("  mat_c = {:?}", mat_c);
    println!("mat.mult_mat_scaled_addto(alpha, mat_b, beta, mat_c) results in ");
    mat.mult_mat_scaled_addto(alpha, mat_b, beta, mat_c);
    println!("  mat_c = {:?}", mat_c);
    if !test_eq(&mat_c.rb(), shouldbe, 0.0) { panic!("test_mult_mat_scaled_addto() failed"); }
}

fn test_tmult_mat(mat: &dyn MatrixOps2<Float>, mat_b: &MatRef<Float>, mat_c: &mut MatMut<Float>, shouldbe: &MatRef<Float>)
{
    println!("mat.tmult_mat(mat_b, mat_c) results in ");
    mat.tmult_mat(mat_b, mat_c);
    println!("  mat_c = {:?}", mat_c);
    if !test_eq(&mat_c.rb(), shouldbe, 0.0) { panic!("test_tmult_mat() failed"); }
}

fn test_tmult_mat_addto(mat: &dyn MatrixOps2<Float>, mat_b: &MatRef<Float>, mat_c: &mut MatMut<Float>, shouldbe: &MatRef<Float>)
{
    println!("mat.tmult_mat_addto(mat_b, mat_c) results in ");
    mat.tmult_mat_addto(mat_b, mat_c);
    println!("  mat_c = {:?}", mat_c);
    if !test_eq(&mat_c.rb(), shouldbe, 0.0) { panic!("test_tmult_mat_addto() failed"); }
}

fn test_tmult_mat_scaled_addto(mat: &dyn MatrixOps2<Float>, alpha:Float, mat_b: &MatRef<Float>, beta: Float, mat_c: &mut MatMut<Float>, shouldbe: &MatRef<Float>)
{
    println!("alpha = {}, beta = {}", alpha, beta);
    println!("  mat_c = {:?}", mat_c);
    println!("mat.tmult_mat_scaled_addto(alpha, mat_b, beta, mat_c) results in ");
    mat.tmult_mat_scaled_addto(alpha, mat_b,beta, mat_c);
    println!("  mat_c = {:?}", mat_c);
    if !test_eq(&mat_c.rb(), shouldbe, 0.0) { panic!("test_tmult_mat_scaled_addto() failed"); }
}


#[test]
fn test_generic_matrix_ops()
{
	println!("testing MatrixOp for generic matrices");
    let mat = Mat::from_rows( &[
    	&[   1.0,   2.0,   3.0 ],
        &[ 100.0, 200.0, 300.0 ],
    ]);
	println!("mat = {:?}", mat);
	const M: usize = 2;
	const N: usize = 3;
	assert_eq!(mat.nrows(), M);
	assert_eq!(mat.ncols(), N);
	println!("");

    println!("generic matrix (as above): testing MatrixOps1");
    let vct_b: &[Float; N] = &[ 1.0, 2.0, 3.0 ];
    println!("vct_b = {:?}", vct_b);
    let vct_c: &mut[Float; M] = &mut[ 0.0; M ];
    let shouldbe: &mut[Float; M] = &mut[ 14.0, 1400.0 ];
	test_mult_vct(&mat.as_ref(), vct_b, vct_c, shouldbe);
    shouldbe.iter_mut().for_each( |s| *s *= 2.0 );
	test_mult_vct_addto(&mat.as_ref(), vct_b, vct_c, shouldbe);

    let vct_b: &[Float; M] = &[ 2.0, 3.0 ];
    println!("vct_b = {:?}", vct_b);
    let vct_c: &mut[Float; N] = &mut[ 0.0; N ];
    let shouldbe: &mut[Float; N] = &mut[ 302.0, 604.0, 906.0 ];
	//test_mult_vct(&mat.as_ref().transpose(), vct_b, vct_c, shouldbe);
	test_tmult_vct(&mat.as_ref(), vct_b, vct_c, shouldbe);

    shouldbe.iter_mut().for_each( |s| *s *= 2.0 );
	//test_mult_vct_addto(&mat.as_ref().transpose(), vct_b, vct_c, shouldbe);
	test_tmult_vct_addto(&mat.as_ref(), vct_b, vct_c, shouldbe);
	println!("");

    let vct_b: &[Float; N] = &[ 1.0, 2.0, 3.0 ];
    println!("vct_b = {:?}", vct_b);
    let vct_c: &mut[Float; M] = &mut[ 0.3, 0.2 ];
    let alpha: Float = 0.5;
    let beta: Float = 0.1;
    println!("alpha = {}, beta = {}", alpha, beta);
    let shouldbe = &mut[ 7.03, 700.02 ];
	test_mult_vct_scaled_addto(&mat.as_ref(), alpha, vct_b, beta, vct_c, shouldbe);
	println!("");

    println!("generic matrix (as above): testing MatrixOps2");
    let mat_b = Mat::from_rows( &[
    	&[ 1.0, 1.0 ],
    	&[ 1.0, 2.0 ],
    	&[ 1.0, 3.0 ],
    ]);
	println!("mat_b = {:?}", mat_b);
    let mut mat_c = Mat::<Float>::zeros(2,2);
    let shouldbe = Mat::from_rows( &[
        &[   6.0,   14.0 ],
        &[ 600.0, 1400.0 ],
    ]);
	test_mult_mat(&mat.as_ref(), &mat_b.as_ref(), &mut mat_c.as_mut(), &shouldbe.as_ref());
    let shouldbe = Mat::from_rows( &[
        &[   12.0,   28.0 ],
        &[ 1200.0, 2800.0 ],
    ]);
	test_mult_mat_addto(&mat.as_ref(), &mat_b.as_ref(), &mut mat_c.as_mut(), &shouldbe.as_ref());

    println!("");
    let mut mat_c = Mat::from_rows( &[
    	&[ 1.0, 2.0 ],
    	&[ 3.0, 4.0 ],
    ]);
    let alpha: Float = 0.5;
    let beta: Float = 0.1;
    let shouldbe = Mat::from_rows( &[
    	&[   3.1,   7.2 ],
    	&[ 300.3, 700.4 ],
    ]);
	test_mult_mat_scaled_addto(&mat.as_ref(), alpha, &mat_b.as_ref(), beta, &mut mat_c.as_mut(), &shouldbe.as_ref());
    println!("");

    let mat_b = Mat::from_rows( &[
    	&[ 1.0, 2.0 ],
    	&[ 1.0, 4.0 ],
    ]);
	println!("mat_b = {:?}", mat_b);
    let mut mat_c = Mat::<Float>::zeros(3,2);
    let shouldbe = Mat::from_rows( &[
        &[ 101.0, 402.0 ],
        &[ 202.0, 804.0 ],
        &[ 303.0, 1206.0 ],
    ]);
	test_tmult_mat(&mat.as_ref(), &mat_b.as_ref(), &mut mat_c.as_mut(), &shouldbe.as_ref());
    let shouldbe = Mat::from_rows( &[
        &[ 202.0,  804.0 ],
        &[ 404.0, 1608.0 ],
        &[ 606.0, 2412.0 ],
    ]);
	test_tmult_mat_addto(&mat.as_ref(), &mat_b.as_ref(), &mut mat_c.as_mut(), &shouldbe.as_ref());

    println!("");
    let mut mat_c = Mat::from_rows( &[
    	&[ 1.0, 2.0 ],
    	&[ 3.0, 4.0 ],
    	&[ 5.0, 6.0 ],
    ]);
    let alpha: Float = 0.5;
    let beta: Float = 0.01;
    let shouldbe = Mat::from_rows( &[
        &[  50.51, 201.02 ],
        &[ 101.03, 402.04 ],
        &[ 151.55, 603.06 ],
    ]);
	test_tmult_mat_scaled_addto(&mat.as_ref(), alpha, &mat_b.as_ref(), beta, &mut mat_c.as_mut(), &shouldbe.as_ref());
    println!("");
}


#[test]
fn test_diagonal_matrix_ops()
{
    println!("testing DiagonalMatrixOps");
    const N: usize = 2;
    let diag: &[Float; N] = &[ 2.0, 100.0 ];
    let mat = DiagonalMatrixOps {
        diag: diag
    };
    println!("mat.nrows() = {}", mat.nrows());
    println!("mat.ncols() = {}", mat.ncols());
    println!("mat.diag = {:?}", mat.diag);
	println!("");

    println!("diagonal matrix (as above): testing MatrixOps1");
    let vct_b: &[Float; N] = &[ 1.0, 3.0 ];
    println!("vct_b = {:?}", vct_b);
    let vct_c: &mut[Float; N] = &mut[ 0.0; N ];
    let shouldbe: &mut[Float; N] = &mut[ 2.0, 300.0 ];
	test_mult_vct(&mat, vct_b, vct_c, shouldbe);
	test_tmult_vct(&mat, vct_b, vct_c, shouldbe);

    shouldbe.iter_mut().for_each( |s| *s *= 2.0 );
	test_mult_vct_addto(&mat, vct_b, vct_c, shouldbe);
    shouldbe.iter_mut().for_each( |s| *s *= 1.5 );
	test_tmult_vct_addto(&mat, vct_b, vct_c, shouldbe);
	println!("");

    let vct_c: &mut[Float; N] = &mut[ 3.0, 4.0 ];
    println!("vct_c = {:?}", vct_c);
    let alpha: Float = 10.0;
    let beta: Float = 0.1;
    let shouldbe = &mut[ 20.3, 3000.4 ];
    test_mult_vct_scaled_addto(&mat, alpha, vct_b, beta, vct_c, shouldbe);
    let vct_c: &mut[Float; N] = &mut[ 3.0, 4.0 ];
    println!("vct_c = {:?}", vct_c);
    test_tmult_vct_scaled_addto(&mat, alpha, vct_b, beta, vct_c, shouldbe);
	println!("");

    println!("diagonal matrix (as above): testing MatrixOps2");
    let mat_b = Mat::from_rows( &[
    	&[ 1.0, 2.0, 3.0 ],
    	&[ 4.0, 5.0, 6.0 ],
    ]);
    println!("mat_b = {:?}", mat_b);
    let mut mat_c = Mat::<Float>::zeros(2,3);
    let shouldbe = Mat::from_rows( &[
    	&[ 2.0, 4.0, 6.0 ],
    	&[ 400.0, 500.0, 600.0 ],
    ]);
    test_mult_mat(&mat, &mat_b.as_ref(), &mut mat_c.as_mut(), &shouldbe.as_ref());
    let shouldbe2 = Mat::from_rows( &[
    	&[ 4.0, 8.0, 12.0 ],
    	&[ 800.0, 1000.0, 1200.0 ],
    ]);
    test_mult_mat_addto(&mat, &mat_b.as_ref(), &mut mat_c.as_mut(), &shouldbe2.as_ref());
    
    test_tmult_mat(&mat, &mat_b.as_ref(), &mut mat_c.as_mut(), &shouldbe.as_ref());
    test_tmult_mat_addto(&mat, &mat_b.as_ref(), &mut mat_c.as_mut(), &shouldbe2.as_ref());

    println!("");
    let mut mat_c = Mat::from_rows( &[
    	&[ 1.0, 2.0, 3.0 ],
    	&[ 4.0, 5.0, 6.0 ],
    ]);
    println!("mat_c = {:?}", mat_c);
    let alpha: Float = 10.0;
    let beta: Float = 0.1;
    let shouldbe = Mat::from_rows( &[
    	&[   20.1,   40.2,   60.3 ],
    	&[ 4000.4, 5000.5, 6000.6 ],
    ]);
	test_mult_mat_scaled_addto(&mat, alpha, &mat_b.as_ref(), beta, &mut mat_c.as_mut(), &shouldbe.as_ref());    

    println!("");
    let mut mat_c = Mat::from_rows( &[
    	&[ 1.0, 2.0, 3.0 ],
    	&[ 4.0, 5.0, 6.0 ],
    ]);
    println!("mat_c = {:?}", mat_c);
	test_tmult_mat_scaled_addto(&mat, alpha, &mat_b.as_ref(), beta, &mut mat_c.as_mut(), &shouldbe.as_ref());    
}

} // mod tests

