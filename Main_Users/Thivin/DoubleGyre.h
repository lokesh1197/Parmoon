// Navier-Stokes problem, Driven Cavity
// 
// u(x,y) = ?
// p(x,y) = ?

#include<cmath>

void ExampleFile()
{
  OutPut("Example: Driven.h" << endl) ;
}
// ========================================================================
// initial solution
// ========================================================================
void InitialU1(double x, double y, double *values)
{
  double time   =  TDatabase::TimeDB->CURRENTTIME;
  double eps    = 0.25;
  double omega  = (2*M_PI)/10;
  double a      = eps*sin(omega*time);
  double b      = 1.0 - 2*eps*sin(omega*time);
  double A      = 0.1;
  double f      = a*x*x + b*x;
  double f_x    = 2*a*x + b;
  double val    = -M_PI*A*sin(M_PI*f)*cos(M_PI*y);
  values[0] = val;
  // values[0] = 0.01;
}


void InitialU2(double x, double y, double *values)
{
  double time   =  TDatabase::TimeDB->CURRENTTIME;
  double eps    = 0.25;
  double omega  = (2*M_PI)/10;
  double a      = eps*sin(omega*time);
  double b      = 1.0 - 2*eps*sin(omega*time);
  double A      = 0.1;
  double f      = a*x*x + b*x;
  double f_x    = 2*a*x + b;
  double val    = M_PI*A*cos(M_PI*f)*sin(M_PI*y)*f_x;

  values[0] = val;
  // values[0] = 0.0;
}

void InitialP(double x, double y, double *values)
{
  values[0] = 0;
}


// ========================================================================
// exact solution
// ========================================================================
void ExactU1(double x, double y, double *values)
{
  values[0] = 0.0;
  values[1] = 0;
  values[2] = 0;
  values[3] = 0;
}

void ExactU2(double x, double y, double *values)
{
  values[0] = 0;
  values[1] = 0;
  values[2] = 0;
  values[3] = 0;
}

void ExactP(double x, double y, double *values)
{
  values[0] = 0;
  values[1] = 0;
  values[2] = 0;
  values[3] = 0;
}

// ========================================================================
// boundary conditions
// ========================================================================
void BoundCondition(int i, double t, BoundCond &cond)
{
  cond = DIRICHLET;
}

void U1BoundValue(int BdComp, double Param, double &value)
{
  double t = TDatabase::TimeDB->CURRENTTIME;
  double eps = 1e-8;

  switch(BdComp)
  {
    case 0: 
            value=0;
            break;
    case 1: 
            value=0;
            break;
    case 2:  
          if( (abs(Param) - 0.0 )<eps || (abs(1.0-Param) - 0.0 )<eps  )
            value=0; // top moving side velocity
          else
            value =0;
            break;
    case 3: 
            value=0;
            break;
    default: cout << "wrong boundary part number" << endl;
            break;
  }
}

void U2BoundValue(int BdComp, double Param, double &value)
{
  value = 0;
}

// ========================================================================
// coefficients for Stokes form: A, B1, B2, f1, f2
// ========================================================================
void LinCoeffs(int n_points, double *X, double *Y,
               double **parameters, double **coeffs)
{
  int i;
  double *coeff, x, y; 
  static double eps=1/TDatabase::ParamDB->RE_NR;

  for(i=0;i<n_points;i++)
  {
    coeff = coeffs[i];
    x = X[i];
    y = Y[i];

    coeff[0] = eps;

    coeff[1] = 0;  // f1
    coeff[2] = 0;  // f2
  }
}


