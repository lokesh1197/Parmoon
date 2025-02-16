// =======================================================================
//
// Purpose:     main program for scalar equations for computing Error Norms
//
// Author:      Thivin Anandh	
//
// History:     Implementation started on Jan 2023

// =======================================================================

#include <Domain.h>
#include <Database.h>
#include <SystemTCD2D.h>
#include <SystemTCD2D_ALE.h>
#include <FEDatabase2D.h>
#include <FESpace2D.h>
#include <SquareStructure2D.h>
#include <Structure2D.h>
#include <QuadAffin.h>
#include <DirectSolver.h>
#include <Assemble2D.h>
#include <Output2D.h>
#include <LinAlg.h>
#include <CD2DErrorEstimator.h>
#include <MainUtilities.h>
#include <TimeDiscRout.h>

#include <string.h>
#include <sstream>
#include <MooNMD_Io.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>
#include <algorithm>

// =======================================================================
// include current example
// =======================================================================
#include "../Examples/CD_2D/TempDistribution.h"
// #include "../Examples/TCD_2D/SinCos1.h"
// #include "../Examples_All/TCD_2D/Time3.h"
// #include "../Examples/TCD_2D/exp_0.h"
//    #include "../Examples/TCD_2D/exp_2.h"
// #include "../Examples_All/TCD_2D/exp_1.h"
// #include "../Main_Users/Sashi/TCD_2D/Hemker.h"

int main(int argc, char *argv[])
{
	int i, j, l, m, N_SubSteps, ORDER, N_Cells, N_DOF, img = 1, N_G;
	int N_Active;

	double *sol, *rhs, *oldrhs, t1, t2, errors[5], Linfty;
	double tau, end_time, *defect, olderror, olderror1, hmin, hmax;

	bool UpdateStiffnessMat, UpdateRhs, ConvectionFirstTime;
	char *VtkBaseName;
	const char vtkdir[] = "VTK";

	TDomain *Domain;
	TDatabase *Database = new TDatabase();
	TFEDatabase2D *FEDatabase = new TFEDatabase2D();
	TCollection *coll;
	TFESpace2D *Scalar_FeSpace, *fesp[1];
	TFEFunction2D *Scalar_FeFunction;
	TOutput2D *Output;
	TSystemTCD2D *SystemMatrix;
	TAuxParam2D *aux;
	MultiIndex2D AllDerivatives[3] = {D00, D10, D01};

	std::ostringstream os;
	os << " ";

	// ======================================================================
	// set the database values and generate mesh
	// ======================================================================
	// set variables' value in TDatabase using argv[1] (*.dat file), and generate the MESH based
	Domain = new TDomain(argv[1]);

	

	if (TDatabase::ParamDB->PROBLEM_TYPE == 0)
		TDatabase::ParamDB->PROBLEM_TYPE = 2;
	OpenFiles();

	Database->WriteParamDB(argv[0]);
	Database->WriteTimeDB();
	ExampleFile();

	/* include the mesh from a mesh generator, for a standard mesh use the
	 * build-in function. The GEOFILE describes the boundary of the domain. */
	if (TDatabase::ParamDB->MESH_TYPE == 0)
	{
		Domain->ReadGeo(TDatabase::ParamDB->GEOFILE);
		OutPut("PRM-GEO used for meshing !!!" << endl);
	} // ParMooN  build-in Geo mesh
	else if (TDatabase::ParamDB->MESH_TYPE == 1)
	{
		Domain->GmshGen(TDatabase::ParamDB->GEOFILE);
		OutPut("GMSH used for meshing !!!" << endl);
	} // gmsh mesh
	// else if (TDatabase::ParamDB->MESH_TYPE == 2) //triangle mesh
	// {
	// 	OutPut("Triangle.h used for meshing !!!" << endl);
	// 	TriaReMeshGen(Domain);
	// }
	else
	{
		OutPut("Mesh Type not known, set MESH_TYPE correctly!!!" << endl);
		exit(0);
	}

	// #if defined(__HEMKER__) || defined(__BEAM__)
	// 	TriaReMeshGen(Domain);
	// 	TDatabase::ParamDB->UNIFORM_STEPS = 0;
	// #endif

	// refine grid up to the coarsest level
	for (i = 0; i < TDatabase::ParamDB->UNIFORM_STEPS; i++)
		Domain->RegRefineAll();

	// write grid into an Postscript file
	if (TDatabase::ParamDB->WRITE_PS)
		Domain->PS("Domain.ps", It_Finest, 0);

	// create output directory, if not already existing
	if (TDatabase::ParamDB->WRITE_VTK)
		mkdir(vtkdir, 0777);

	//=========================================================================
	// construct all finite element spaces
	//=========================================================================
	ORDER = TDatabase::ParamDB->ANSATZ_ORDER;

	coll = Domain->GetCollection(It_Finest, 0);
	N_Cells = coll->GetN_Cells();
	OutPut("N_Cells (space) : " << N_Cells << endl);

	// fespaces for scalar equation
	Scalar_FeSpace = new TFESpace2D(coll, (char *)"fe space", (char *)"solution space",
									BoundCondition, 1, NULL);

	N_DOF = Scalar_FeSpace->GetN_DegreesOfFreedom();
	N_Active = Scalar_FeSpace->GetActiveBound();
	OutPut("dof all      : " << setw(10) << N_DOF << endl);
	OutPut("dof active   : " << setw(10) << N_Active << endl);

	//======================================================================
	// construct all finite element functions
	//======================================================================
	sol = new double[N_DOF];
	rhs = new double[N_DOF];
	oldrhs = new double[N_DOF];

	memset(sol, 0, N_DOF * SizeOfDouble);
	memset(rhs, 0, N_DOF * SizeOfDouble);

	Scalar_FeFunction = new TFEFunction2D(Scalar_FeSpace, (char *)"sol", (char *)"sol", sol, N_DOF);

	// interpolate the initial value
	Scalar_FeFunction->Interpolate(InitialCondition);

	//======================================================================
	// SystemMatrix construction and solution
	//======================================================================
	// Disc type: GALERKIN (or) SDFEM  (or) UPWIND (or) SUPG (or) LOCAL_PROJECTION
	// Solver: AMG_SOLVE (or) GMG  (or) DIRECT
	SystemMatrix = new TSystemTCD2D(Scalar_FeSpace, GALERKIN, DIRECT);

	// initilize the system matrix with the functions defined in Example file
	SystemMatrix->Init(BilinearCoeffs, BoundCondition, BoundValue);

	// assemble the system matrix with given aux, sol and rhs
	// aux is used to pass  addition fe functions (eg. mesh velocity) that is nedded for assembling,
	// otherwise, just pass with NULL
	SystemMatrix->AssembleMRhs(NULL, sol, rhs);

	//======================================================================
	// produce outout at t=0
	//======================================================================
	VtkBaseName = TDatabase::ParamDB->VTKBASENAME;
	Output = new TOutput2D(2, 2, 1, 1, Domain);

	Output->AddFEFunction(Scalar_FeFunction);

	//     Scalar_FeFunction->Interpolate(Exact);
	if (TDatabase::ParamDB->WRITE_VTK)
	{
		os.seekp(std::ios::beg);
		if (img < 10)
			os << "VTK/" << VtkBaseName << ".0000" << img << ".vtk" << ends;
		else if (img < 100)
			os << "VTK/" << VtkBaseName << ".000" << img << ".vtk" << ends;
		else if (img < 1000)
			os << "VTK/" << VtkBaseName << ".00" << img << ".vtk" << ends;
		else if (img < 10000)
			os << "VTK/" << VtkBaseName << ".0" << img << ".vtk" << ends;
		else
			os << "VTK/" << VtkBaseName << "." << img << ".vtk" << ends;
		Output->WriteVtk(os.str().c_str());
		img++;
	}

	// measure errors to known solution
	if (TDatabase::ParamDB->MEASURE_ERRORS)
	{
		fesp[0] = Scalar_FeSpace;
		aux = new TAuxParam2D(1, 0, 0, 0, fesp, NULL, NULL, NULL, NULL, 0, NULL);

		for (j = 0; j < 5; j++)
			errors[j] = 0;

		Scalar_FeFunction->GetErrors(Exact, 3, AllDerivatives, 2, L2H1Errors, BilinearCoeffs, aux, 1, fesp, errors);

		olderror = errors[0];
		olderror1 = errors[1];

		OutPut("time: " << TDatabase::TimeDB->CURRENTTIME);
		OutPut(" L2: " << errors[0]);
		OutPut(" H1-semi: " << errors[1] << endl);
		Linfty = errors[0];
	} //  if(TDatabase::ParamDB->MEASURE_ERRORS)

	coll->GetHminHmax(&hmin, &hmax);
	OutPut("h_min : " << hmin << " h_max : " << hmax << endl);

	//    TDatabase::TimeDB->TIMESTEPLENGTH =  hmax;

	// TDatabase::TimeDB->TIMESTEPLENGTH =  hmax;
	//======================================================================
	// time disc loop
	//======================================================================
	// parameters for time stepping scheme
	m = 0;
	N_SubSteps = GetN_SubSteps();
	end_time = TDatabase::TimeDB->ENDTIME;

	UpdateStiffnessMat = FALSE; // check BilinearCoeffs in example file
	UpdateRhs = TRUE;			// check BilinearCoeffs in example file
	ConvectionFirstTime = TRUE;

	// Read the mapping file
	double *CoOrdinates = new double[N_DOF * 2]();
	TFEVectFunct2D *CoOrdinates_Fevect = new TFEVectFunct2D(Scalar_FeSpace, "c", "c", CoOrdinates, N_DOF, 2);

	CoOrdinates_Fevect->GridToData();
	std::vector<int> mapping = GenerateMapping("UnitSquare_Coord.txt", CoOrdinates_Fevect);

	// Store PINNS Solution
	double *PinnsSolution = new double[N_DOF]();
	TFEVectFunct2D *PINNS_FeVect = new TFEVectFunct2D(Scalar_FeSpace, "p_c", "p_c", PinnsSolution, N_DOF, 1);
	TFEFunction2D *Pinns_FEFunc = PINNS_FeVect->GetComponent(0);

	// Store DMD Solution
	double *DMDSolution = new double[N_DOF]();
	TFEVectFunct2D *DMD_FeVect = new TFEVectFunct2D(Scalar_FeSpace, "d_c", "d_c", DMDSolution, N_DOF, 1);
	TFEFunction2D *DMD_FEFunc = DMD_FeVect->GetComponent(0);

	// Store PINNS FEM Error
	double *Error_pinns_fem = new double[N_DOF]();
	TFEVectFunct2D *Error_pinns_fem_FeVect = new TFEVectFunct2D(Scalar_FeSpace, "diff_fem_vs_pinns", "diff_fem_vs_pinns", Error_pinns_fem, N_DOF, 1);
	TFEFunction2D *Error_pinns_fem_FeVect_FEFunc = Error_pinns_fem_FeVect->GetComponent(0);

	// Store PINNS DMD Error
	double *Error_dmd_fem = new double[N_DOF]();
	TFEVectFunct2D *Error_dmd_fem_FeVect = new TFEVectFunct2D(Scalar_FeSpace, "diff_fem_vs_dmd", "diff_fem_vs_dmd", Error_dmd_fem, N_DOF, 1);
	TFEFunction2D *Error_dmd_fem_FeVect_FEFunc = Error_dmd_fem_FeVect->GetComponent(0);

	// Store PINNS DMD Error
	double *Error_pinns_dmd = new double[N_DOF]();
	TFEVectFunct2D *Error_pinns_dmd_FeVect = new TFEVectFunct2D(Scalar_FeSpace, "diff_dmd_vs_pinns", "diff_dmd_vs_pinns", Error_pinns_dmd, N_DOF, 1);
	TFEFunction2D *Error_pinns_dmd_FeVect_FEFunc = Error_pinns_dmd_FeVect->GetComponent(0);

	char *VtkBaseName_pinns = "Pinns";
	TOutput2D *Outputpinns = new TOutput2D(2, 2, 1, 1, Domain);
	Outputpinns->AddFEFunction(Pinns_FEFunc);

	char *VtkBaseName_dmd = "DMD";
	TOutput2D *Outputdmd = new TOutput2D(2, 2, 1, 1, Domain);
	Outputdmd->AddFEFunction(DMD_FEFunc);

	char *VtkBaseName_fem_pinns = "ERROR-fem-pinns";
	TOutput2D *OutputError_fem_pinns = new TOutput2D(2, 2, 1, 1, Domain);
	OutputError_fem_pinns->AddFEFunction(Error_pinns_fem_FeVect_FEFunc);

	char *VtkBaseName_fem_dmd = "ERROR-fem-dmd";
	TOutput2D *OutputError_fem_dmd = new TOutput2D(2, 2, 1, 1, Domain);
	OutputError_fem_dmd->AddFEFunction(Error_dmd_fem_FeVect_FEFunc);

	char *VtkBaseName_pinns_dmd = "ERROR-dmd-pinns";
	TOutput2D *OutputError_pinns_dmd = new TOutput2D(2, 2, 1, 1, Domain);
	OutputError_pinns_dmd->AddFEFunction(Error_pinns_dmd_FeVect_FEFunc);

	std::vector<int> N_Modes = {1, 2, 4, 8, 16,32,64};

	int PercentageOfDMD = atoi(argv[2]);
	// Generate File pointers for all the NModes
	std::ofstream ErrorNormFiles[6];
	std::ofstream Er;

	mkdir("ErrorNorms", 0777);
	
	std::string s = "ErrorNorms/Percentage_" + std::to_string(PercentageOfDMD) + ".err";
	Er.open(s.c_str());

	if(!Er)
	{
		cout << "[ERROR] - Cannot open the file/folder : ErrorNorms/Percentage" <<endl;
		exit(0);
	}
	
	// time loop starts
	while (TDatabase::TimeDB->CURRENTTIME < end_time - TDatabase::TimeDB->CURRENTTIMESTEPLENGTH )
	{
		m++;
		TDatabase::TimeDB->INTERNAL_STARTTIME = TDatabase::TimeDB->CURRENTTIME;

		for (l = 0; l < N_SubSteps; l++) // sub steps of fractional step theta
		{
			SetTimeDiscParameters(1);

			if (m == 1)
			{
				OutPut("Theta1: " << TDatabase::TimeDB->THETA1 << endl);
				OutPut("Theta2: " << TDatabase::TimeDB->THETA2 << endl);
				OutPut("Theta3: " << TDatabase::TimeDB->THETA3 << endl);
				OutPut("Theta4: " << TDatabase::TimeDB->THETA4 << endl);
			}

			tau = TDatabase::TimeDB->CURRENTTIMESTEPLENGTH;
			TDatabase::TimeDB->CURRENTTIME += tau;

			OutPut(endl
				   << "CURRENT TIME: ");
			OutPut(TDatabase::TimeDB->CURRENTTIME << endl);

			// copy rhs to oldrhs
			memcpy(oldrhs, rhs, N_DOF * SizeOfDouble);

			// unless the stiffness matrix or rhs change in time, it is enough to
			// assemble only once at the begning
			if (UpdateStiffnessMat || UpdateRhs || ConvectionFirstTime)
			{
				SystemMatrix->AssembleARhs(NULL, sol, rhs);

				// M:= M + (tau*THETA1)*A
				// rhs: =(tau*THETA4)*rhs +(tau*THETA3)*oldrhs +[M-(tau*THETA2)A]*oldsol
				// note! sol contains only the previous time step value, so just pass
				// sol for oldsol
				SystemMatrix->AssembleSystMat(oldrhs, sol, rhs, sol);
				ConvectionFirstTime = FALSE;
			}

			// solve the system matrix
			SystemMatrix->Solve(sol, rhs);

			// restore the mass matrix for the next time step
			// unless the stiffness matrix or rhs change in time, it is not necessary to assemble the system matrix in every time step
			if (UpdateStiffnessMat || UpdateRhs)
			{
				SystemMatrix->RestoreMassMat();
			}

			std::stringstream ss;
			ss << std::fixed << std::setprecision(2) << TDatabase::TimeDB->CURRENTTIME;
			std::string time = ss.str();
			// std::string time = std::to_string(TDatabase::TimeDB->CURRENTTIME);
			time.erase(std::remove(time.begin(), time.end(), '.'), time.end());
			std::string s = time;
			unsigned int number_of_zeros = 6 - s.length(); // add 2 zeros
			s.insert(0, number_of_zeros, '0');

			std::string filename = "/home/thivin/Shell/cmg_PINNs/Output/TempDistPaper/5/dat_logs/PinnsData_" + s + ".dat";

			// ---------- OBTAIN PINNS NORM WITH FEM ----------------- //

			// Read Pinns Data
			readPinnsData(Pinns_FEFunc, mapping, filename.c_str());

			//Make sure that the dirichlet values are equal to the galerkin FEM values So that they 
			// are not included in error comptuationa
			double* pinSol = Pinns_FEFunc->GetValues();
			for(int i=N_Active; i < N_DOF  ; i++)
			{
				pinSol[i] = sol[i];
			} 

			// Get the  PINNS Norm
			double l2Norm_Pinns = ComputeL2ErrorDifference(Scalar_FeSpace, Scalar_FeFunction, Pinns_FEFunc);
			cout << "Pinns     : " << l2Norm_Pinns << endl;

			// ---------- OBTAIN DMD NORM WITH FEM ---------- //

			for ( int k = 0 ; k < N_Modes.size() ; k++)
			{
				// Since N Modes is Not available to the 40 and 50 % data
				if(N_Modes[k] == 64 && (PercentageOfDMD == 40 || PercentageOfDMD == 50) )
					continue;

				std::string a = "/home/thivin/Shell/cmg_PINNs/Output/TempDistPaper/5/dmd_" + std::to_string(PercentageOfDMD) + "/DMD_Recon_Scalar_" + std::to_string(N_Modes[k]);
				filename = a + "_c_" + s + ".dat";
				readPinnsData(DMD_FEFunc, mapping, filename.c_str());

				double* dmdSol = DMD_FEFunc->GetValues();
				for(int i=N_Active; i < N_DOF  ; i++)
				{
					dmdSol[i] = sol[i];
				} 


				double l2Norm_DMD = ComputeL2ErrorDifference(Scalar_FeSpace, Scalar_FeFunction, DMD_FEFunc);
				cout <<N_Modes[k] <<" - DMD       : " << l2Norm_DMD << endl;

				double l2Norm_DMD_PINNS = ComputeL2ErrorDifference(Scalar_FeSpace, Pinns_FEFunc, DMD_FEFunc);
				cout  <<N_Modes[k] << " - DMD_PINNS : " << l2Norm_DMD_PINNS << endl;

				// Store the Error values in Arrays for vtk Visualisation
				double *Error_pinns_fem_Array = Error_pinns_fem_FeVect_FEFunc->GetValues();
				double *Error_dmd_fem_Array = Error_dmd_fem_FeVect_FEFunc->GetValues();
				double *Error_pinns_dmd_Array = Error_pinns_dmd_FeVect_FEFunc->GetValues();

				double *pinnsArray = Pinns_FEFunc->GetValues();
				double *dmdArray = DMD_FEFunc->GetValues();

				for (int i = 0; i < N_DOF; i++)
				{
					Error_pinns_fem_Array[i] 	= sol[i] - pinnsArray[i];
					Error_dmd_fem_Array[i] 		= sol[i] - dmdArray[i];
					Error_pinns_dmd_Array[i] 	= pinnsArray[i] - dmdArray[i];
				}

				Er << N_Modes[k]<<"," << l2Norm_Pinns <<"," <<l2Norm_DMD <<"," << l2Norm_DMD_PINNS<<"\n";
				
			}
			


		} // for(l=0;l<N_SubSteps;l++)

		//======================================================================
		// produce outout
		//======================================================================
		if (m == 1 || m % TDatabase::TimeDB->STEPS_PER_IMAGE == 0)
			if (TDatabase::ParamDB->WRITE_VTK)
			{
				os.seekp(std::ios::beg);
				if (img < 10)
					os << "VTK/" << VtkBaseName << ".0000" << img << ".vtk" << ends;
				else if (img < 100)
					os << "VTK/" << VtkBaseName << ".000" << img << ".vtk" << ends;
				else if (img < 1000)
					os << "VTK/" << VtkBaseName << ".00" << img << ".vtk" << ends;
				else if (img < 10000)
					os << "VTK/" << VtkBaseName << ".0" << img << ".vtk" << ends;
				else
					os << "VTK/" << VtkBaseName << "." << img << ".vtk" << ends;
				Output->WriteVtk(os.str().c_str());
			}

		if (m == 1 || m % TDatabase::TimeDB->STEPS_PER_IMAGE == 0)
			if (TDatabase::ParamDB->WRITE_VTK)
			{
				os.seekp(std::ios::beg);
				if (img < 10)
					os << "VTK/" << VtkBaseName_pinns << ".0000" << img << ".vtk" << ends;
				else if (img < 100)
					os << "VTK/" << VtkBaseName_pinns << ".000" << img << ".vtk" << ends;
				else if (img < 1000)
					os << "VTK/" << VtkBaseName_pinns << ".00" << img << ".vtk" << ends;
				else if (img < 10000)
					os << "VTK/" << VtkBaseName_pinns << ".0" << img << ".vtk" << ends;
				else
					os << "VTK/" << VtkBaseName_pinns << "." << img << ".vtk" << ends;
				Outputpinns->WriteVtk(os.str().c_str());
				// img++;
			}

		if (m == 1 || m % TDatabase::TimeDB->STEPS_PER_IMAGE == 0)
			if (TDatabase::ParamDB->WRITE_VTK)
			{
				os.seekp(std::ios::beg);
				if (img < 10)
					os << "VTK/" << VtkBaseName_dmd << ".0000" << img << ".vtk" << ends;
				else if (img < 100)
					os << "VTK/" << VtkBaseName_dmd << ".000" << img << ".vtk" << ends;
				else if (img < 1000)
					os << "VTK/" << VtkBaseName_dmd << ".00" << img << ".vtk" << ends;
				else if (img < 10000)
					os << "VTK/" << VtkBaseName_dmd << ".0" << img << ".vtk" << ends;
				else
					os << "VTK/" << VtkBaseName_dmd << "." << img << ".vtk" << ends;
				Outputdmd->WriteVtk(os.str().c_str());
				// img++;
			}

		if (m == 1 || m % TDatabase::TimeDB->STEPS_PER_IMAGE == 0)
			if (TDatabase::ParamDB->WRITE_VTK)
			{
				os.seekp(std::ios::beg);
				if (img < 10)
					os << "VTK/" << VtkBaseName_fem_pinns << ".0000" << img << ".vtk" << ends;
				else if (img < 100)
					os << "VTK/" << VtkBaseName_fem_pinns << ".000" << img << ".vtk" << ends;
				else if (img < 1000)
					os << "VTK/" << VtkBaseName_fem_pinns << ".00" << img << ".vtk" << ends;
				else if (img < 10000)
					os << "VTK/" << VtkBaseName_fem_pinns << ".0" << img << ".vtk" << ends;
				else
					os << "VTK/" << VtkBaseName_fem_pinns << "." << img << ".vtk" << ends;
				OutputError_fem_pinns->WriteVtk(os.str().c_str());
				// img++;
			}

		if (m == 1 || m % TDatabase::TimeDB->STEPS_PER_IMAGE == 0)
			if (TDatabase::ParamDB->WRITE_VTK)
			{
				os.seekp(std::ios::beg);
				if (img < 10)
					os << "VTK/" << VtkBaseName_fem_dmd << ".0000" << img << ".vtk" << ends;
				else if (img < 100)
					os << "VTK/" << VtkBaseName_fem_dmd << ".000" << img << ".vtk" << ends;
				else if (img < 1000)
					os << "VTK/" << VtkBaseName_fem_dmd << ".00" << img << ".vtk" << ends;
				else if (img < 10000)
					os << "VTK/" << VtkBaseName_fem_dmd << ".0" << img << ".vtk" << ends;
				else
					os << "VTK/" << VtkBaseName_fem_dmd << "." << img << ".vtk" << ends;
				OutputError_fem_dmd->WriteVtk(os.str().c_str());
				// img++;
			}

		if (m == 1 || m % TDatabase::TimeDB->STEPS_PER_IMAGE == 0)
			if (TDatabase::ParamDB->WRITE_VTK)
			{
				os.seekp(std::ios::beg);
				if (img < 10)
					os << "VTK/" << VtkBaseName_pinns_dmd << ".0000" << img << ".vtk" << ends;
				else if (img < 100)
					os << "VTK/" << VtkBaseName_pinns_dmd << ".000" << img << ".vtk" << ends;
				else if (img < 1000)
					os << "VTK/" << VtkBaseName_pinns_dmd << ".00" << img << ".vtk" << ends;
				else if (img < 10000)
					os << "VTK/" << VtkBaseName_pinns_dmd << ".0" << img << ".vtk" << ends;
				else
					os << "VTK/" << VtkBaseName_pinns_dmd << "." << img << ".vtk" << ends;
				OutputError_pinns_dmd->WriteVtk(os.str().c_str());
				// img++;
			}

		img++;
		//======================================================================
		// measure errors to known solution
		//======================================================================
		if (TDatabase::ParamDB->MEASURE_ERRORS)
		{
			Scalar_FeFunction->GetErrors(Exact, 3, AllDerivatives, 2, L2H1Errors, BilinearCoeffs, aux, 1, fesp, errors);

			OutPut("time: " << TDatabase::TimeDB->CURRENTTIME);
			OutPut(" L2: " << errors[0]);
			OutPut(" H1-semi: " << errors[1] << endl);

			errors[3] += (errors[0] * errors[0] + olderror * olderror) * TDatabase::TimeDB->TIMESTEPLENGTH / 2.0;
			olderror = errors[0];
			OutPut(TDatabase::TimeDB->CURRENTTIME << " L2(0,T;L2) " << sqrt(errors[3]) << " ");

			errors[4] += (errors[1] * errors[1] + olderror1 * olderror1) * TDatabase::TimeDB->TIMESTEPLENGTH / 2.0;
			OutPut("L2(0,T;H1) " << sqrt(errors[4]) << endl);
			olderror1 = errors[1];

			if (Linfty < errors[0])
				Linfty = errors[0];

			OutPut("Linfty " << Linfty << endl);
		} //  if(TDatabase::ParamDB->MEASURE_ERRORS)

	} // while(TDatabase::TimeDB->CURRENTTIME< end_time)

	

	//======================================================================
	// produce final outout
	//======================================================================

	if (TDatabase::ParamDB->WRITE_VTK)
	{
		os.seekp(std::ios::beg);
		if (img < 10)
			os << "VTK/" << VtkBaseName << ".0000" << img << ".vtk" << ends;
		else if (img < 100)
			os << "VTK/" << VtkBaseName << ".000" << img << ".vtk" << ends;
		else if (img < 1000)
			os << "VTK/" << VtkBaseName << ".00" << img << ".vtk" << ends;
		else if (img < 10000)
			os << "VTK/" << VtkBaseName << ".0" << img << ".vtk" << ends;
		else
			os << "VTK/" << VtkBaseName << "." << img << ".vtk" << ends;
		Output->WriteVtk(os.str().c_str());
		img++;
	}

	CloseFiles();

	return 0;
} // end main
