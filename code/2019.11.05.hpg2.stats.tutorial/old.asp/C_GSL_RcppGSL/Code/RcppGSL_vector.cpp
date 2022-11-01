// [[Rcpp::depends(RcppGSL)]]

#include <RcppGSL.h>

#include <gsl/gsl_vector.h>
#include <cmath>


// The following example defines a .Call compatible function called sum_gsl_vector_int that operates on a gsl_vector_int through the RcppGSL::vector<int> template specialization

// [[Rcpp::export]]
int sum_gsl_vector_int(const RcppGSL::vector<int> & vec){
    int res = std::accumulate(vec.begin(), vec.end(), 0);
    return res;
}


// A second example shows a simple function that grabs elements of an R list as gsl_vector objects using implicit conversion mechanisms of Rcpp

// [[Rcpp::export]]
double gsl_vector_sum_2(const Rcpp::List & data) {
    // grab "x" as a gsl_vector through the RcppGSL::vector<double> class
    const RcppGSL::vector<double> x = data["x"];
    
    // grab "y" as a gsl_vector through the RcppGSL::vector<int> class
    const RcppGSL::vector<int> y = data["y"];
    double res = 0.0;
    for (size_t i=0; i< x->size; i++) {
        res += x[i] * y[i];
    }
    
    return res;    // return the result, memory freed automatically
}

/*** R

 sum_gsl_vector_int(1:10)
 
 data <- list( x = seq(0,1,length=10), y = 1:10 )
 gsl_vector_sum_2(data)

*/