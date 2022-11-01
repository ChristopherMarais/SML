// [[Rcpp::depends(RcppGSL)]]

#include <RcppGSL.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>

// [[Rcpp::export]]

Rcpp::NumericVector getEigenValues(RcppGSL::Matrix & M) {
    
    int k = M.ncol();
    
    RcppGSL::Vector ev(k);  	// instead of gsl_vector_alloc(k);
    gsl_eigen_symm_workspace *w = gsl_eigen_symm_alloc(k);
    gsl_eigen_symm (M, ev, w);
    gsl_eigen_symm_free (w);
    
    return Rcpp::wrap(ev);				// return results vector
}

/*** R

 set.seed(8915)
 
 X <- matrix(rnorm(4*4), 4, 4)
 Z <- X %*% t(X)
 
 getEigenValues(Z)
 
 eigen(Z)

*/