#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

void cov_fun_C (gsl_matrix *A, gsl_matrix *B, gsl_vector *theta, gsl_matrix * sigma)
{
    int m, n, i, j;
    m = A -> size1;
    n = B -> size1;
    double a1, a2, b1, b2, tmp = 0.0;
    
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            a1 = gsl_matrix_get (A, i, 0);
            a2 = gsl_matrix_get (A, i, 1);
            b1 = gsl_matrix_get (B, j, 0);
            b2 = gsl_matrix_get (B, j, 1);
            
            if (a1==b1 && a2==b2) tmp = 2.0;
            else tmp = exp(-1.0 * gsl_vector_get (theta,0) * fabs(a1 - b1) - gsl_vector_get (theta, 1) * (a2 - b2) * (a2 - b2));
            gsl_matrix_set (sigma, i, j, tmp);
        }
    }
}

double log_Gaussian_SVD_C (gsl_vector *x, gsl_vector *mu, gsl_matrix *U, gsl_matrix *V, gsl_vector *S)
{
    gsl_vector *b, *y;
    int p, i;
    double results=0.0;
    
    p = x -> size;
    
    b = gsl_vector_calloc (p);
    y = gsl_vector_calloc (p);
    
    gsl_vector_memcpy (b, x);
    gsl_vector_sub (b, mu); // b is replaced by x-mu
    gsl_linalg_SV_solve (U, V, S, b, y); // y = inv(sigma)(x-mu)
    gsl_vector_mul (b, y); // b = (x-mu) inv(sigma) (x-mu)
    
    results = p * log(2.0 * M_PI);
    for (i = 0; i < p; i++) results += log(gsl_vector_get (S, i)) + gsl_vector_get (b, i);
    
    gsl_vector_free (b);
    gsl_vector_free (y);
    
    return -0.5 * results;
    
}

double log_Gaussian_Chol_C (gsl_vector *x, gsl_vector *mu, gsl_matrix *cholesky)
{
    gsl_vector * b, * y;
    int p, i;
    double results=0.0;
    
    p = x -> size;
    
    b = gsl_vector_calloc (p);
    y = gsl_vector_calloc (p);
    
    gsl_vector_memcpy (b, x);
    gsl_vector_sub (b, mu); // b = x - mu
    gsl_linalg_cholesky_solve (cholesky, b, y);
    gsl_vector_mul(b, y);
    
    results = p * log(2.0 * M_PI);
    for (i = 0; i < p; i++) results += 2.0 * log(gsl_matrix_get (cholesky, i, i)) + gsl_vector_get (b, i);
    
    gsl_vector_free (b);
    gsl_vector_free (y);
    
    return -0.5 * results;
}

double log_gaussian_QR_C (gsl_vector *x, gsl_vector *mu, gsl_matrix * QR, gsl_vector * tau)
{
    gsl_vector * b, * y;
    int p, i;
    double results = 0.0;
    
    p = x -> size;
    
    b = gsl_vector_calloc (p);
    y = gsl_vector_calloc (p);
    
    gsl_vector_memcpy (b, x);
    gsl_vector_sub (b, mu); // b = x - mu
    gsl_linalg_QR_solve (QR, tau, b, y);
    gsl_vector_mul (b, y);
    
    results = p * log (2 * M_PI);
    for (i = 0; i < p; i++) results += log(fabs(gsl_matrix_get(QR, i,i))) + gsl_vector_get (b, i);
    
    gsl_vector_free(b);
    gsl_vector_free(y);
    
    return -0.5 * results;
    
}

int main ()
{
    const gsl_rng_type * T;
    gsl_rng * r;
    
    gsl_rng_env_setup();
    
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    
    gsl_rng_set (r, 8915);
    
    gsl_matrix * sigma, * A, * B;
    gsl_vector * mu, * x;
    gsl_vector * theta;
    
    int n, i=0, index1=0, index2,j;
    
    struct timespec tstart={0,0}, tend={0,0};
    
    FILE * file1, * file2, * file3, * file4;
    
    gsl_matrix * U, * V;
    gsl_vector * S, * work;
    
    gsl_matrix * QR;
    gsl_vector * tau;
    
    gsl_matrix * cholesky;
    
    file1 = fopen("time1_simple.txt", "a");
    file2 = fopen("time2_simple.txt", "a");
    file3 = fopen("time3_simple.txt", "a");
    file4 = fopen("results_simple.txt", "a");
    
    for (i=0; i <= 3; i++)
    {
        n = 100;
        j=0;
        for (j=0; j < i; j++) n = n * 2;
        
        U = gsl_matrix_calloc(n, n);
        V = gsl_matrix_calloc(n, n);
        S = gsl_vector_calloc(n);
        work = gsl_vector_calloc(n);
        
        QR = gsl_matrix_calloc(n, n);
        tau = gsl_vector_calloc(n);
        
        cholesky = gsl_matrix_calloc(n, n);
        
        theta = gsl_vector_calloc (2);
        A = gsl_matrix_calloc (n, 2);
        B = gsl_matrix_calloc (n, 2);
        
        sigma = gsl_matrix_calloc (n, n);
        x = gsl_vector_calloc (n);
        mu = gsl_vector_calloc (n);
        
        for (index1=0; index1 < 2; index1++) gsl_vector_set (theta, index1, gsl_rng_uniform(r));
        for (index1=0; index1 < n; index1++) for (index2 = 0; index2 < 2; index2++) gsl_matrix_set (A, index1, index2, gsl_rng_uniform(r) * sqrt(n));
        gsl_matrix_memcpy (B, A);
        
        // calculate covariance matrix
        clock_gettime(CLOCK_MONOTONIC, &tstart);
        cov_fun_C(A, B, theta, sigma);
        clock_gettime(CLOCK_MONOTONIC, &tend);
        fprintf(file1, "%f\n", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        
        for (index1 = 0; index1 < n; index1++) gsl_vector_set (x, index1, gsl_ran_ugaussian(r));
        
        // factorization
        gsl_matrix_memcpy (cholesky, sigma);
        clock_gettime(CLOCK_MONOTONIC, &tstart);
        gsl_linalg_cholesky_decomp (cholesky);
        clock_gettime(CLOCK_MONOTONIC, &tend);
        fprintf(file2, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        
        gsl_matrix_memcpy (QR, sigma);
        clock_gettime(CLOCK_MONOTONIC, &tstart);
        gsl_linalg_QR_decomp (QR, tau);
        clock_gettime(CLOCK_MONOTONIC, &tend);
        fprintf(file2, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        
        gsl_matrix_memcpy (U, sigma);
        clock_gettime(CLOCK_MONOTONIC, &tstart);
        gsl_linalg_SV_decomp (U, V, S, work);
        clock_gettime(CLOCK_MONOTONIC, &tend);
        fprintf(file2, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        
        fprintf(file2, "\n");
        
        // log-likelihood
        clock_gettime(CLOCK_MONOTONIC, &tstart);
        fprintf(file4, "%f ", log_Gaussian_Chol_C(x, mu, cholesky));
        clock_gettime(CLOCK_MONOTONIC, &tend);
        fprintf(file3, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        
        clock_gettime(CLOCK_MONOTONIC, &tstart);
        fprintf(file4, "%f ", log_gaussian_QR_C(x, mu, QR, tau));
        clock_gettime(CLOCK_MONOTONIC, &tend);
        fprintf(file3, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        
        clock_gettime(CLOCK_MONOTONIC, &tstart);
        fprintf(file4, "%f ", log_Gaussian_SVD_C(x, mu, U, V, S));
        clock_gettime(CLOCK_MONOTONIC, &tend);
        fprintf(file3, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        
        fprintf(file4, "\n");
        fprintf(file3, "\n");

        gsl_vector_free (mu);
        gsl_vector_free (x);
        gsl_matrix_free (sigma);
        gsl_matrix_free (A);
        gsl_matrix_free (B);
        gsl_vector_free (theta);
        
        gsl_matrix_free (U);
        gsl_matrix_free (V);
        gsl_vector_free (S);
        gsl_vector_free (work);
        
        gsl_matrix_free (cholesky);
        
        gsl_matrix_free (QR);
        gsl_vector_free (tau);
    
    }
    
    fclose(file1);
    fclose(file2);
    fclose(file3);
    fclose(file4);
    
    gsl_rng_free (r);
    
    return 0;
}