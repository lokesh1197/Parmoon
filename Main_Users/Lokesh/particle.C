// =======================================================================
//
// Purpose:    Run the siminhale particle deposition using the loaded data file. 
// =======================================================================
#include <Domain.h>
#include <Database.h>
#include <FEDatabase3D.h>
#include <LinAlg.h>
#include <FESpace3D.h>
#include <SystemTNSE3D.h>
#include <SquareStructure3D.h>
#include <Structure3D.h>
#include <Output3D.h>
#include <LinAlg.h>
#include <MainUtilities.h>
#include <TimeDiscRout.h>

#include <tetgen.h>

#include <string.h>
#include <sstream>
#include <MooNMD_Io.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <cstdlib>
#include <sys/types.h>

#include <fstream>

#include <omp.h>

#include<Particle.h>
#include<algorithm>
#include<cstring>

#include "siminhale.h"

#define AMG 0
#define GMG 1
#define DIRECT 2

double bound = 0;
double timeC = 0;

void printall_array(double *Arr1, double *Arr2, int SizeOfArr)
{

	std::ofstream myfile;
	myfile.open("entries/Arr1.txt");
	for (int ii = 0; ii < SizeOfArr; ii++)
		myfile << " " << Arr1[ii] << endl;
	myfile.close();

	myfile.open("entries/Arr2.txt");
	for (int ii = 0; ii < SizeOfArr; ii++)
		myfile << " " << Arr2[ii] << endl;
	myfile.close();
}

void norm_array(double *Arr1, double *Arr2, int SizeOfArr)
{
	double sum_Arr1 = 0;
	for (int ii = 0; ii < SizeOfArr; ii++)
		{
			sum_Arr1 = sum_Arr1 + (Arr1[ii] * Arr1[ii]);
		}
	cout << "sum_Arr1::" << sum_Arr1 << endl;

	double sum_Arr2 = 0;
	for (int ii = 0; ii < SizeOfArr; ii++)
		{
			sum_Arr2 = sum_Arr2 + (Arr2[ii] * Arr2[ii]);
		}
	cout << "sum_Arr2::" << sum_Arr2 << endl;
}
// =======================================================================
// main program
// =======================================================================
int main(int argc, char *argv[])
{
	// ======================================================================
	//  declaration of variables
	// ======================================================================
	int i, j, l, m, N_Cells, N_U, N_P, N_TotalDOF, img = 1, pressure_space_code;
	int Max_It, NSEType, velocity_space_code;
	int LEVELS, mg_level, mg_type;
	int N_SubSteps, N_L;

	double *sol, *rhs, *oldrhs, t1, t2, errors[4], residual, impuls_residual;
	double **Sol_array, **Rhs_array;
	double limit, AllErrors[7];
	double tau, oldtau, end_time;

	double start_time, stop_time, start_assembling_solving = 0, end_assembling_solving = 0,
		total_assembling_solving = 0, start_int = 0, end_int = 0, total_int = 0;

	double Cd, Cl;
	int checker = 1;
	TDomain *Domain;
	TDatabase *Database = new TDatabase();

	int profiling;

	TFEDatabase3D *FEDatabase = new TFEDatabase3D();
	TCollection *coll, *mortarcoll = NULL;
	TFESpace3D *velocity_space, *pressure_space, **Velocity_FeSpace, **Pressure_FeSpace, *fesp[1];
	TFEVectFunct3D **Velocity, *u;
	TFEFunction3D *p, *u1, *u2, *u3, **Pressure, *fefct[2];
	TOutput3D *Output;
	TSystemTNSE3D *SystemMatrix;
	TAuxParam3D *aux;
	MultiIndex3D AllDerivatives[4] = {D000, D100, D010, D001};

	TFESpace3D *projection_space;
	TFESpace3D **Projection_FeSpace;

	const char vtkdir[] = "VTK";
	char *PsBaseName, *VtkBaseName, *GEO, *PRM, *SMESH;

	char Name[] = "name";
	char UString[] = "u";
	char PString[] = "p";
	char NameString[] = "VMS";
	char SubID[] = "";
	std::ostringstream os;
	double stime;

	{
		os << " ";

		mkdir(vtkdir, 0777);
		
		//exit based on the number of arguments
		if (argc < 4)
			{
				cout << "Usage: " << argv[0] << " <parameter file>  <number of particles> <path to csv>" << endl;
				exit(1);
			}

		// ======================================================================
		// set the database values and generate mesh
		// ======================================================================
		/** set variables' value in TDatabase using argv[1] (*.dat file), and generate the MESH based */
		Domain = new TDomain(argv[1]);

		profiling = TDatabase::ParamDB->timeprofiling;
		profiling = 0;

		// omp_set_num_threads(42);

		if (profiling) start_time = GetTime();

		OpenFiles();
		OutFile.setf(std::ios::scientific);

		Database->CheckParameterConsistencyNSE();

		Database->WriteParamDB(argv[0]);
		ExampleFile();

		/** needed in the new implementation */
		if (TDatabase::ParamDB->FLOW_PROBLEM_TYPE == STOKES)
			TDatabase::ParamDB->SC_NONLIN_MAXIT_SADDLE = 1;

		cout << " mesh Type is : " << TDatabase::ParamDB->MESH_TYPE << endl;

		/* meshgenerator */
		if (TDatabase::ParamDB->MESH_TYPE == 0)
			Domain->Init(TDatabase::ParamDB->BNDFILE, TDatabase::ParamDB->GEOFILE);
		else if (TDatabase::ParamDB->MESH_TYPE == 1)
			Domain->GmshGen(TDatabase::ParamDB->GEOFILE);
		else
			{
				OutPut("Mesh Type not known, set MESH_TYPE correctly!!!" << endl);
				exit(0);
			}

		LEVELS = TDatabase::ParamDB->LEVELS;
		if (TDatabase::ParamDB->SOLVER_TYPE == DIRECT)
			{
				TDatabase::ParamDB->UNIFORM_STEPS += (LEVELS - 1);
				LEVELS = 1;
			}
		// refine grid up to the coarsest level  for Normal Mesh
		if(TDatabase::ParamDB->MESH_TYPE == 0)
			for (i = 0; i < TDatabase::ParamDB->UNIFORM_STEPS; i++)
				Domain->RegRefineAll();

		//=========================================================================
		// set data for multigrid
		//=========================================================================

		// set type of multilevel
		mg_type = TDatabase::ParamDB->SC_MG_TYPE_SADDLE;

		if (TDatabase::ParamDB->SOLVER_TYPE == AMG_SOLVE || TDatabase::ParamDB->SOLVER_TYPE == DIRECT)
			{
				mg_type = 0;
				TDatabase::ParamDB->SC_MG_TYPE_SADDLE = mg_type;
			}

		if (mg_type == 1) mg_level = LEVELS + 1;
		else mg_level = LEVELS;

		if (TDatabase::ParamDB->SOLVER_TYPE == GMG)
			{
				OutPut("=======================================================" << endl);
				OutPut("======           GEOMETRY  LEVEL ");
				OutPut(LEVELS << "              ======" << endl);
				OutPut("======           MULTIGRID LEVEL ");
				OutPut(mg_level << "              ======" << endl);
				OutPut("=======================================================" << endl);
			}

		Velocity_FeSpace = new TFESpace3D *[mg_level];
		Pressure_FeSpace = new TFESpace3D *[mg_level];

#ifdef __PRIVATE__
		Projection_FeSpace = new TFESpace3D *[mg_level];
#endif

		Velocity = new TFEVectFunct3D *[mg_level];
		Pressure = new TFEFunction3D *[mg_level];

		Sol_array = new double *[mg_level];
		Rhs_array = new double *[mg_level];

		//=========================================================================
		// loop over all levels (not a multigrid level but for convergence study)
		//=========================================================================
		for (i = 0; i < LEVELS; i++)
			{
				if (i)
					{
						Domain->RegRefineAll();
					}

				coll = Domain->GetCollection(It_Finest, 0);

				//=========================================================================
				// construct all finite element spaces
				//=========================================================================
				if (mg_type == 1) // lower order FE on coarse grids
					{
						Velocity_FeSpace[i] = new TFESpace3D(coll, Name, UString, BoundCondition, Non_USpace, 1);
						Pressure_FeSpace[i] = new TFESpace3D(coll, Name, PString, BoundCondition, DiscP_PSpace, 0);

						if (i == LEVELS - 1) // higher order on fine level
							{
								GetVelocityAndPressureSpace3D(coll, BoundCondition, velocity_space,
																							pressure_space, &pressure_space_code,
																							TDatabase::ParamDB->VELOCITY_SPACE,
																							TDatabase::ParamDB->PRESSURE_SPACE);
								Velocity_FeSpace[i + 1] = velocity_space;
								Pressure_FeSpace[i + 1] = pressure_space;

								// defaulty inf-sup pressure space will be selected based on the velocity space, so update it in database
								TDatabase::ParamDB->INTERNAL_PRESSURE_SPACE = pressure_space_code;
								velocity_space_code = TDatabase::ParamDB->VELOCITY_SPACE;
							}
					}
				else
					{
						GetVelocityAndPressureSpace3D(coll, BoundCondition, velocity_space,
																					pressure_space, &pressure_space_code,
																					TDatabase::ParamDB->VELOCITY_SPACE,
																					TDatabase::ParamDB->PRESSURE_SPACE);

						Velocity_FeSpace[i] = velocity_space;
						Pressure_FeSpace[i] = pressure_space;

						TDatabase::ParamDB->INTERNAL_PRESSURE_SPACE = pressure_space_code;
						velocity_space_code = TDatabase::ParamDB->VELOCITY_SPACE;
					}

#ifdef __PRIVATE__
				if (TDatabase::ParamDB->DISCTYPE == VMS_PROJECTION)
					{
						if (TDatabase::ParamDB->VMS_LARGE_VELOCITY_SPACE == 0)
							projection_space = new TFESpace3D(coll, NameString, UString, BoundCondition,
																								DiscP_PSpace, 0);
						else
							projection_space = new TFESpace3D(coll, NameString, UString, BoundCondition,
																								DiscP_PSpace, 1);

						Projection_FeSpace[i] = projection_space;
						N_L = Projection_FeSpace[i]->GetN_DegreesOfFreedom();
						OutPut("Dof Projection : " << setw(10) << N_L << endl);
					}
#endif

				//======================================================================
				// construct all finite element functions
				//======================================================================
				N_U = Velocity_FeSpace[i]->GetN_DegreesOfFreedom();
				N_P = Pressure_FeSpace[i]->GetN_DegreesOfFreedom();
				N_TotalDOF = 3 * N_U + N_P;

				sol = new double[N_TotalDOF];
				memset(sol, 0, N_TotalDOF * SizeOfDouble);
				Sol_array[i] = sol;

				rhs = new double[N_TotalDOF];
				memset(rhs, 0, N_TotalDOF * SizeOfDouble);
				Rhs_array[i] = rhs;

				u = new TFEVectFunct3D(Velocity_FeSpace[i], UString, UString, sol, N_U, 3);
				Velocity[i] = u;
				p = new TFEFunction3D(Pressure_FeSpace[i], PString, PString, sol + 3 * N_U, N_P);
				Pressure[i] = p;

				if (i == LEVELS - 1 && mg_type == 1)
					{
						N_U = Velocity_FeSpace[i + 1]->GetN_DegreesOfFreedom();
						N_P = Pressure_FeSpace[i + 1]->GetN_DegreesOfFreedom();
						N_TotalDOF = 3 * N_U + N_P;

						sol = new double[N_TotalDOF];
						memset(sol, 0, N_TotalDOF * SizeOfDouble);
						Sol_array[i + 1] = sol;

						rhs = new double[N_TotalDOF];
						memset(rhs, 0, N_TotalDOF * SizeOfDouble);
						Rhs_array[i + 1] = rhs;

						u = new TFEVectFunct3D(Velocity_FeSpace[i + 1], UString, UString, sol, N_U, 3);
						Velocity[i + 1] = u;
						p = new TFEFunction3D(Pressure_FeSpace[i + 1], PString, PString, sol + 3 * N_U, N_P);
						Pressure[i + 1] = p;
					} // if(i==LEVELS-1 && mg_type==1)
			} //  for(i=0;i<LEVELS;i++)

		u1 = Velocity[mg_level - 1]->GetComponent(0);
		u2 = Velocity[mg_level - 1]->GetComponent(1);
		u3 = Velocity[mg_level - 1]->GetComponent(2);

		oldrhs = new double[N_TotalDOF];

		//======================================================================
		// SystemMatrix construction and solution
		//======================================================================
		NSEType = TDatabase::ParamDB->NSTYPE;

		if (profiling)
			{
				start_int = GetTime();
			}

		// get a  TNSE3D system
		SystemMatrix = new TSystemTNSE3D(mg_level, Velocity_FeSpace, Pressure_FeSpace, Velocity, Pressure,
																		 Sol_array, Rhs_array, TDatabase::ParamDB->DISCTYPE, NSEType, TDatabase::ParamDB->SOLVER_TYPE
#ifdef __PRIVATE__
																		 ,
																		 Projection_FeSpace
#endif
																		 );

		// initilize the system matrix with the functions defined in Example file
		SystemMatrix->Init(LinCoeffs, BoundCondition, U1BoundValue, U2BoundValue, U3BoundValue);

		printf("SystemMatrix constructed\n");

		if (profiling)
			{
				end_int = GetTime();
				total_int = end_int - start_int;
			}

		if (profiling) start_assembling_solving = GetTime();

		u1->Interpolate(InitialU1);
		u2->Interpolate(InitialU2);
		u3->Interpolate(InitialU3);

		//======================================================================
		// prepare for outout
		//======================================================================
		VtkBaseName = TDatabase::ParamDB->VTKBASENAME;
		Output = new TOutput3D(2, 2, 1, 1, Domain);

		Output->AddFEVectFunct(u);
		Output->AddFEFunction(p);

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

		//======================================================================
		// time disc loop
		//======================================================================
		// parameters for time stepping scheme
		cout << " INT PROJ PRESSURE : " << TDatabase::ParamDB->INTERNAL_PROJECT_PRESSURE << endl;
		m = 0;
		N_SubSteps = GetN_SubSteps();
		oldtau = 1.;
		end_time = TDatabase::TimeDB->ENDTIME;
		limit = TDatabase::ParamDB->SC_NONLIN_RES_NORM_MIN_SADDLE;
		Max_It = TDatabase::ParamDB->SC_NONLIN_MAXIT_SADDLE;
		memset(AllErrors, 0, 7. * SizeOfDouble);
	}


	// INTIALISE THE PARTICLES 
	int numPart = std::atoi(argv[2]);
	TParticles* particleObject =  new TParticles(numPart,0.0,0.0,0.4,Velocity_FeSpace[0]);
	cout << " Particles Initialised " <<endl;



	particleObject->OutputFile("siminhale_0000.csv");
	int StartNo = 2;


	// time loop starts
	while (TDatabase::TimeDB->CURRENTTIME < end_time) // time cycle
		{
			{
				m++;
				TDatabase::TimeDB->INTERNAL_STARTTIME = TDatabase::TimeDB->CURRENTTIME;
			}
				
			for (l = 0; l < N_SubSteps; l++) // sub steps of fractional step theta
				{

					{
						SetTimeDiscParameters(1);
						{
							if (m == 1)
								{
									OutPut("Theta1: " << TDatabase::TimeDB->THETA1 << endl);
									OutPut("Theta2: " << TDatabase::TimeDB->THETA2 << endl);
									OutPut("Theta3: " << TDatabase::TimeDB->THETA3 << endl);
									OutPut("Theta4: " << TDatabase::TimeDB->THETA4 << endl);
								}
						}
						tau = TDatabase::TimeDB->CURRENTTIMESTEPLENGTH;
						TDatabase::TimeDB->CURRENTTIME += tau;

						{
							OutPut(endl
										 << "CURRENT TIME: ");
							OutPut(TDatabase::TimeDB->CURRENTTIME << endl);
						}



					}

				}

			// Considering the simulation has saturated upto 1000 time  steps, If the particle moves more than 
			int lineNo=0;
			if(StartNo >  995)
				StartNo = 995;
			cout << "Start No : " << StartNo <<endl;
			std::string prefix(argv[3]); 
			std::string baseFileName = "Solution_";

			// Save the u Solution, the number of char in the img should be 6, remaing space is padded by zeros
			int padding = 6 - std::to_string(StartNo).length();

			// create the file name
			std::string u1FileName = prefix + baseFileName + "u_" + std::string(padding, '0')
				+ std::to_string(StartNo) + ".bin";
			std::string u2FileName = prefix + baseFileName + "v_"+ std::string(padding, '0')
				+ std::to_string(StartNo) + ".bin";
			std::string u3FileName = prefix + baseFileName + "w_"+ std::string(padding, '0')
				+ std::to_string(StartNo) + ".bin";
			std::string pFileName = prefix + baseFileName + "p_"+ std::string(padding, '0')
				+ std::to_string(StartNo) + ".bin";

			std::string filename = prefix + std::to_string(StartNo) + ".bin";

			// Read the file into the solution array
			std::ifstream u1File(u1FileName, std::ios::in | std::ios::binary);
			std::ifstream u2File(u2FileName, std::ios::in | std::ios::binary);
			std::ifstream u3File(u3FileName, std::ios::in | std::ios::binary);
			std::ifstream pFile(pFileName, std::ios::in | std::ios::binary);

			// throw an error if the file is not found
			if (!u1File.is_open())
				throw std::runtime_error("Could not open file " + u1FileName);
			if (!u2File.is_open())
				throw std::runtime_error("Could not open file " + u2FileName);
			if (!u3File.is_open())
				throw std::runtime_error("Could not open file " + u3FileName);
			if (!pFile.is_open())
				throw std::runtime_error("Could not open file " + pFileName);
				
			// Read the file into the solution array
			u1File.read((char *)sol, sizeof(double) * N_U);
			u2File.read((char *)sol + sizeof(double) * N_U, sizeof(double) * N_U);
			u3File.read((char *)sol + sizeof(double) * 2 * N_U, sizeof(double) * N_U);
			pFile.read((char *)sol + sizeof(double) * 3 * N_U, sizeof(double) * N_P);

			// Close the file
			u1File.close();
			u2File.close();
			u3File.close();
			pFile.close();

			// Increment the file number
			StartNo++;

			for (int i=0 ; i < 3*N_U; i++) sol[i] *= 3.18;

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
				
			if (m >= 20 )  // To start the interpolation after 20 time steps ( to ensure that flow has propogated )
				{
					cout << " Interpolation Started" <<endl;
					//Compute the Particle Displacement for the FTLE values 
					particleObject->interpolateNewVelocity_Parallel(TDatabase::TimeDB->TIMESTEPLENGTH,Velocity[0],Velocity_FeSpace[0]);
					std::string old_str = std::to_string(img-2);
					size_t n_zero = 4;
					auto new_str = std::string(n_zero - std::min(n_zero, old_str.length()), '0') + old_str;
					std::string name =  "siminhale_" + new_str + ".csv";
					particleObject->OutputFile(name.c_str());

				int depositedCount = 0;
				for (int i = 0; i < particleObject->isParticleDeposited.size(); i++) {
						if (particleObject->isParticleDeposited[i]) depositedCount++;
				}
				if (depositedCount == numPart) {
						cout << "All particles deposited/escaped" << endl;
						cout << "Total particles: " << depositedCount << endl;
						break;
				}
				}
		} // while(TDatabase::TimeDB->CURRENTTIME< e

	CloseFiles();
	return 0;
} // end main
