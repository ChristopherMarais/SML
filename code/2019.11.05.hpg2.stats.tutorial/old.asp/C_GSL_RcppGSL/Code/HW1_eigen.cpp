#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/SVD>

using namespace Eigen;

void cov_fun_C (const MatrixXd & A, const MatrixXd & B, Vector2d theta, MatrixXd & sigma)
{
    int m, n, i, j;
    m = A.rows();
    n = B.rows();
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

double log_Gaussian_SVD_C (const VectorXd & x, const VectorXd & mu, const MatrixXd & U, const MatrixXd & V, const VectorXd & S)
{
    VectorXd b, y1, y2;
    int p, i;
    double results=0.0;
    
    p = x.size();
    
    b = x - mu;
    
    y1 = U.transpose() * b;
    y2 = V.transpose() * b;
    
    results = p * log(2.0 * M_PI);
    for (i = 0; i < p; i++) results += log(S(i)) + y1(i) * y2(i) / S(i);
    
    return -0.5 * results;
    
}

double log_Gaussian_Chol_C (const VectorXd & x, const VectorXd & mu, const MatrixXd & cholesky)
{
    VectorXd b, y;
    int p, i;
    double results=0.0;
    
    p = x.size();
    
    b = x - mu;
    
    y = cholesky.triangularView<Lower>().solve(b);
    
    results = p * log(2.0 * M_PI);
    for (i = 0; i < p; i++) results += 2.0 * log(cholesky(i, i)) + y(i) * y(i);
    
    return -0.5 * results;
}

double log_gaussian_QR_C (VectorXd & x, VectorXd & mu, MatrixXd & Q, MatrixXd & R)
{
    VectorXd b, y;
    int p, i;
    double results = 0.0;
    
    p = x.size();
    
    b = x - mu;
    y = Q.transpose() * b;
    y = R.triangularView<Upper>().solve(y);
    
    results = p * log (2 * M_PI);
    for (i = 0; i < p; i++) results += log(fabs(R(i,i))) + b(i) * y(i);

    return -0.5 * results;
    
}

int main ()
{
    srand(SEED);
    
    int n, i=0, index1=0, index2, j;
    
    struct timespec tstart={0,0}, tend={0,0};
    
    FILE * file1, * file2, * file3, * file4;
    
    MatrixXd sigma, A, B;
    
    VectorXd mu, x;
    Vector2d theta;
    
    MatrixXd U, V;
    VectorXd S;
   
    MatrixXd Q, R;
    
    MatrixXd cholesky;
    
    file1 = fopen("time1_eigen_INDEX.txt", "a");
    file2 = fopen("time2_eigen_INDEX.txt", "a");
    file3 = fopen("time3_eigen_INDEX.txt", "a");
    file4 = fopen("results_eigen_INDEX.txt", "a");
    
    for (i=0; i <= 2; i++)
    {
        n = 100;
        j=0;
        for (j=0; j < i; j++) n = n * 2;
        
        cholesky = MatrixXd::Zero(n, n);
        
        theta = Vector2d::Random().cwiseAbs();
        A = MatrixXd::Random(n, 2);
        A *= sqrt(n);
        B = A;
        
        sigma = MatrixXd::Zero(n, n);
        x = VectorXd::Random(n);
        mu = VectorXd::Random(n);
        
        // calculate covariance matrix
        clock_gettime(CLOCK_MONOTONIC, &tstart);
        
        cov_fun_C(A, B, theta, sigma);
        
        clock_gettime(CLOCK_MONOTONIC, &tend);
        
        fprintf(file1, "%f\n", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        
        // factorization
        clock_gettime(CLOCK_MONOTONIC, &tstart);
        cholesky = sigma.llt().matrixL();
        clock_gettime(CLOCK_MONOTONIC, &tend);
        fprintf(file2, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        

        clock_gettime(CLOCK_MONOTONIC, &tstart);
        HouseholderQR<MatrixXd> qr(sigma);
        Q = qr.householderQ();
        R = qr.matrixQR().triangularView<Upper>();
        clock_gettime(CLOCK_MONOTONIC, &tend);
        fprintf(file2, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        
        clock_gettime(CLOCK_MONOTONIC, &tstart);
        JacobiSVD<MatrixXd> svd(sigma, ComputeFullU | ComputeFullV);
        S = svd.singularValues();
        U = svd.matrixU();
        V = svd.matrixV();
        clock_gettime(CLOCK_MONOTONIC, &tend);
        fprintf(file2, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        fprintf(file2, "\n");
        
        // log-likelihood
        clock_gettime(CLOCK_MONOTONIC, &tstart);
        fprintf(file4, "%f ", log_Gaussian_Chol_C(x, mu, cholesky));
        clock_gettime(CLOCK_MONOTONIC, &tend);
        fprintf(file3, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));

        clock_gettime(CLOCK_MONOTONIC, &tstart);
        fprintf(file4, "%f ", log_gaussian_QR_C(x, mu, Q, R));
        clock_gettime(CLOCK_MONOTONIC, &tend);
        fprintf(file3, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));

        clock_gettime(CLOCK_MONOTONIC, &tstart);
        fprintf(file4, "%f ", log_Gaussian_SVD_C(x, mu, U, V, S));
        clock_gettime(CLOCK_MONOTONIC, &tend);
        fprintf(file3, "%f ", ((double)tend.tv_sec + 1.0e-9*tend.tv_nsec) - ((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
        
        fprintf(file4, "\n");
        fprintf(file3, "\n");
        
    }
    
    fclose(file1);
    fclose(file2);
    fclose(file3);
    fclose(file4);
    
    return 0;
}