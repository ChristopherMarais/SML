#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <armadillo>

using namespace arma;

void cov_fun_C (const mat & A, const mat & B, vec theta, mat & sigma)
{
    int m, n, i, j;
    m = A.n_rows;
    n = B.n_rows;
    double a1, a2, b1, b2, tmp = 0.0;
    
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            a1 = A (i, 0);
            a2 = A (i, 1);
            b1 = B (j, 0);
            b2 = B (j, 1);
            
            if (a1==b1 && a2==b2) tmp = 2.0;
            else tmp = exp(-1.0 * theta(0) * fabs(a1 - b1) - theta(1) * (a2 - b2) * (a2 - b2));
            sigma (i, j) = tmp;
        }
    }
}

double log_Gaussian_SVD_C (const vec & x, const vec & mu, const mat & U, const mat & V, const vec & S)
{
    vec b, y1, y2;
    int p, i;
    double results=0.0;
    
    p = x.n_elem;
    
    b = x - mu;
    
    y1 = U.t() * b;
    y2 = V.t() * b;
    
    results = p * log(2.0 * M_PI);
    for (i = 0; i < p; i++) results += log(S(i)) + y1(i) * y2(i) / S(i);
    
    return -0.5 * results;
    
}

double log_Gaussian_Chol_C (const vec & x, const vec & mu, const mat & cholesky)
{
    vec b, y;
    int p, i;
    double results=0.0;
    
    p = x.n_elem;
    
    b = x - mu;
    
    y = solve(trimatl(cholesky), b);
    
    results = p * log(2.0 * M_PI);
    for (i = 0; i < p; i++) results += 2.0 * log(cholesky(i, i)) + y(i) * y(i);
    
    return -0.5 * results;
}

double log_gaussian_QR_C (vec & x, vec & mu, mat & Q, mat & R)
{
    vec b, y;
    int p, i;
    double results = 0.0;
    
    p = x.n_elem;
    
    b = x - mu;
    y = Q.t() * b;
    y = solve(trimatu(R), y);
    
    results = p * log (2 * M_PI);
    for (i = 0; i < p; i++) results += log(fabs(R(i,i))) + b(i) * y(i);

    return -0.5 * results;
    
}

int main ()
{
    arma_rng::set_seed(SEED);
    
    wall_clock timer;
    
    mat sigma, A, B;
    vec mu, x;
    vec theta;
    
    int n, i=0, index1=0, index2,j;
    
    struct timespec tstart={0,0}, tend={0,0};
    
    FILE * file1, * file2, * file3, * file4;
    
    mat U, V;
    vec S;
   
    mat Q, R;

    
    mat cholesky;
    
    file1 = fopen("time1_arma_INDEX.txt", "a");
    file2 = fopen("time2_arma_INDEX.txt", "a");
    file3 = fopen("time3_arma_INDEX.txt", "a");
    file4 = fopen("results_arma_INDEX.txt", "a");
    
    for (i=0; i <= 5; i++)
    {
        n = 100;
        j=0;
        for (j=0; j < i; j++) n = n * 2;
        
        cholesky = zeros<mat>(n, n);
        
        theta = randu<vec>(2);
        A = randu<mat>(n, 2);
        for (index1 = 0; index1 < n; index1++) for (index2 = 0; index2 < 2; index2++) A (index1, index2) *= sqrt(n);
        B = A;
        
        sigma = zeros<mat>(n, n);
        x = randu<vec>(n);
        mu = randu<vec>(n);
        
        // calculate covariance matrix
//        clock_gettime(CLOCK_MONOTONIC, &tstart);
        timer.tic();
        cov_fun_C(A, B, theta, sigma);
        
//        clock_gettime(CLOCK_MONOTONIC, &tend);
        double time_elp = timer.toc();
        // fprintf(file1, "%f\n", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        fprintf(file1, "%f\n", time_elp);
        
        // factorization
//        clock_gettime(CLOCK_MONOTONIC, &tstart);
        timer.tic();
        cholesky = chol(sigma, "lower");
        time_elp = timer.toc();
//        clock_gettime(CLOCK_MONOTONIC, &tend);
//        fprintf(file2, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        fprintf(file2, "%f ", time_elp);
        

//        clock_gettime(CLOCK_MONOTONIC, &tstart);
        timer.tic();
        qr(Q, R, sigma);
        time_elp = timer.toc();
//        clock_gettime(CLOCK_MONOTONIC, &tend);
//        fprintf(file2, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        fprintf(file2, "%f ", time_elp);
        
//        clock_gettime(CLOCK_MONOTONIC, &tstart);
        timer.tic();
        svd(U, S, V, sigma);
        time_elp = timer.toc();
//        clock_gettime(CLOCK_MONOTONIC, &tend);
//        fprintf(file2, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        fprintf(file2, "%f ", time_elp);
        fprintf(file2, "\n");
        
        // log-likelihood
//        clock_gettime(CLOCK_MONOTONIC, &tstart);
        timer.tic();
        fprintf(file4, "%f ", log_Gaussian_Chol_C(x, mu, cholesky));
        time_elp = timer.toc();
//        clock_gettime(CLOCK_MONOTONIC, &tend);
//        fprintf(file3, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        fprintf(file3, "%f ", time_elp);


//        clock_gettime(CLOCK_MONOTONIC, &tstart);
        timer.tic();
        fprintf(file4, "%f ", log_gaussian_QR_C(x, mu, Q, R));
        time_elp = timer.toc();
//        clock_gettime(CLOCK_MONOTONIC, &tend);
//        fprintf(file3, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        fprintf(file3, "%f ", time_elp);

//        clock_gettime(CLOCK_MONOTONIC, &tstart);
        timer.tic();
        fprintf(file4, "%f ", log_Gaussian_SVD_C(x, mu, U, V, S));
        time_elp = timer.toc();
//        clock_gettime(CLOCK_MONOTONIC, &tend);
//        fprintf(file3, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        fprintf(file3, "%f ", time_elp);
        
        fprintf(file4, "\n");
        fprintf(file3, "\n");
        
    }
    
    fclose(file1);
    fclose(file2);
    fclose(file3);
    fclose(file4);
    
    return 0;
}