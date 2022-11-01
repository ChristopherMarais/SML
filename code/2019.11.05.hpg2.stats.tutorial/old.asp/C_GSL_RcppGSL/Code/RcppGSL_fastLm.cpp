// [[Rcpp::depends(RcppGSL)]]

#include <RcppGSL.h>

#include <gsl/gsl_multifit.h>
#include <cmath>

// [[Rcpp::export]]
Rcpp::List fastLm(const RcppGSL::Matrix & X, const RcppGSL::Vector & y) {
    
    int n = X.nrow(), k = X.ncol();
    double chisq;
    
    RcppGSL::Vector coef(k);                // to hold the coefficient vector
    RcppGSL::Matrix cov(k,k);               // and the covariance matrix
    
    // the actual fit requires working memory we allocate and free
    gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc (n, k);
    gsl_multifit_linear (X, y, coef, cov, &chisq, work);
    gsl_multifit_linear_free (work);
    
    // assign diagonal to a vector, then take square roots to get std.error
    Rcpp::NumericVector std_err;
    std_err = gsl_matrix_diagonal(cov); // need two step decl. and assignment
    std_err = Rcpp::sqrt(std_err);         	// sqrt() is an Rcpp sugar function
    
    return Rcpp::List::create(Rcpp::Named("coefficients") = coef,
                              Rcpp::Named("stderr")       = std_err,
                              Rcpp::Named("df.residual")  = n - k);
    
}

/*** R

y <- log(trees$Volume)
X <- cbind(1, log(trees$Girth))
frm <- formula(log(Volume) ~ log(Girth))
 
gsl_results <- fastLm(X, y)

R_results <- lm(frm, data=trees)
 
gsl_results
 
summary(R_results)

*/