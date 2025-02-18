import os
import numpy as np
import matplotlib.pyplot as plt


def plotError(projectName, runNumber, figName, errorBar = False):


    curDir = os.getcwd();
    #_______________________________________________________
    # Set paths and variables
    #_______________________________________________________
    # Output path for sensitivity analysis (./output)
    outputPath = os.getcwd()+'/output';
    # Location for storing test case (name is specified in inputData.py)
    projectOutputDir = outputPath+'/'+projectName; 
    runDir = projectOutputDir+'/'+str(runNumber); 

    # Change into the run directory inside output/caseName
    os.chdir(runDir);




    #_______________________________________________________
    # Prepare Sample data
    #_______________________________________________________


    zValue = 1.96;

    # Read metadata
    myfile = open("metadata.dat",'rt')
    contents = contents = myfile.read();
    myfile.close();
    trainingDataSize = int((contents.split('Training data size :')[1]).split('\n')[0]);

    # Load input space for sensitivity analysis
    inputData = np.loadtxt("inputSpace.dat");
    # Total number of samples
    numberOfSamples = inputData.shape[0];
    # Total number of input parameters i.e. dimension of Input space for sensitivity analysis
    numberOfInputParam = inputData.shape[1]-1;
    # Total number of output results (4) 
    # i.e. min error, max error, L1 error, L2 error
    numberOfOutputParam = 4;

    # Extract useful information about arrays etc. Refer Run.py to find out the sequence of data
    sampleNumber = inputData[:,0];
    NHL = inputData[:,1];
    OPLTYPE = inputData[:,2];
    HL_0_DIM = inputData[:,3];
    HL_0_TYPE = inputData[:,4];
    HL_1_DIM = inputData[:,5];
    HL_1_TYPE = inputData[:,6];
    HL_2_DIM = inputData[:,7];
    HL_2_TYPE = inputData[:,8];
    EPOCHS = inputData[:,9];

    # Load the output space for sensitivity analysis
    outputData = np.loadtxt("outputSpace.dat");
    L1Error = outputData[:,0];
    L2Error = outputData[:,1];
    MinError = outputData[:,2];
    MaxError = outputData[:,3];
    MSError = outputData[:,4];


    # Standard error
    if (errorBar == True):
        error = np.loadtxt("standardError.dat");
    else:
        error = outputData.copy();
    L1StE = error[:,0];
    L2StE = error[:,1];
    MinStE = error[:,2];
    MaxStE = error[:,3];
    MSEStE = error[:,4];


    # Find out the NHL1, NHL2 and NHL3 data
    flag1 = 0;
    flag2 = 0;
    for i in range(numberOfSamples):
        if (inputData[i,1] == 1):
            flag1 = i+1;
        elif (inputData[i,1] == 2):
            flag2 = i+1;

    #_______________________________________________________
    # Process the data 
    #_______________________________________________________
    p5 = np.percentile(outputData,5, axis=0);
    p5={'L1Error':p5[0], 'L2Error':p5[1], 'MinError':p5[2], 'MaxError':p5[3], 'MSError':p5[4]};

    p95 = np.percentile(outputData,95, axis=0);
    p95={'L1Error':p95[0], 'L2Error':p95[1], 'MinError':p95[2], 'MaxError':p95[3], 'MSError':p95[4]};

    pMean = np.mean(outputData, axis=0);

    lowerbound = 0.1 * np.min(outputData, axis = 0);
    lowerbound = {'L1Error':lowerbound[0], 'L2Error':lowerbound[1], 'MinError':lowerbound[2], 'MaxError':lowerbound[3], 'MSError':lowerbound[4]};

    upperbound = 10. * np.max(outputData, axis = 0) ;
    upperbound = {'L1Error':upperbound[0], 'L2Error':upperbound[1], 'MinError':upperbound[2], 'MaxError':upperbound[3], 'MSError':upperbound[4]};


    #_______________________________________________________
    # Set color codes
    #_______________________________________________________


    #shadeColor = 'beige';
    shadeColor = 'gainsboro';
    s1color = 'red';
    s2color = 'green';
    s3color = 'darkblue';
    s4color = 'black';

    s1size = '2';
    s2size = '1.5';
    s3size = '1';
    s4size = '1.5';

    LW = '0.8';

    #_______________________________________________________
    # Plot the errors for all the networks (design space)
    #_______________________________________________________

    fig, axs = plt.subplots(2,2, figsize=(6.4,4.8), dpi=300, constrained_layout=True);

    # Title:
    plt.suptitle("Training data size: "+str(trainingDataSize));

    #_______________________________________________________
    # L1 Error plot, location 0,0
    #_______________________________________________________
    location = (0,0)
    axs[location].semilogy(np.arange(0,flag1), L1Error[0:flag1],"^", color=s1color, markersize=s1size, label="NHL = 1");
    axs[location].semilogy(np.arange(flag1,flag2), L1Error[flag1:flag2],"s",color=s2color, markersize=s2size, label="NHL = 2");
    axs[location].semilogy(np.arange(flag2, numberOfSamples), L1Error[flag2:numberOfSamples],"o", color=s3color, markersize=s3size, label="NHL = 3");

    axs[location].legend(loc=0,ncol=3, fontsize=6.3);

    axs[location].set_xlabel(r"ANN ID");
    axs[location].set_ylabel(r"$L_1$ Error");
    axs[location].set_ylim(lowerbound['L1Error'],upperbound['L1Error']);
    #axs[location].set_yticks([0.5,1]);

    # highlight 5 and 95 percentile zones with demarkation line
    axs[location].axhspan(lowerbound['L1Error'],p5['L1Error'],facecolor=shadeColor)
    axs[location].axhspan(p95['L1Error'],upperbound['L1Error'],facecolor=shadeColor)
    axs[location].axhline(p5['L1Error'],color=s4color,linewidth=LW);
    axs[location].axhline(p95['L1Error'],color=s4color,linewidth=LW);

    if (errorBar == True):
        axs[location].set_yscale("log", nonpositive='clip')
        axs[location].errorbar(np.arange(0,flag1), L1Error[0:flag1],yerr=zValue*L1StE[0:flag1], color=s1color, alpha=0.1);
        axs[location].errorbar(np.arange(flag1,flag2), L1Error[flag1:flag2],yerr=zValue*L1StE[flag1:flag2], color=s2color, alpha=0.1);
        axs[location].errorbar(np.arange(flag2, numberOfSamples), L1Error[flag2:numberOfSamples],yerr=zValue*L1StE[flag2:numberOfSamples], color=s3color, alpha=0.1);
    

    #_______________________________________________________
    # Mean Squared Error plot
    #_______________________________________________________
    location = (1,0)
    axs[location].semilogy(np.arange(0,flag1), MSError[0:flag1],"^", color=s1color, markersize=s1size, label="NHL = 1");
    axs[location].semilogy(np.arange(flag1,flag2), MSError[flag1:flag2],"s",color=s2color, markersize=s2size, label="NHL = 2");
    axs[location].semilogy(np.arange(flag2, numberOfSamples), MSError[flag2:numberOfSamples],"o", color=s3color, markersize=s3size, label="NHL = 3");

    axs[location].legend(loc=0,ncol=3, fontsize=6.3);

    axs[location].set_xlabel(r"ANN ID");

    axs[location].set_ylabel(r"MSE");
    axs[location].set_ylim(lowerbound['MSError'],upperbound['MSError']);
    #axs[location].set_yticks([0.001,0.01]);

    # highlight 5 and 95 percentile zones with demarkation line
    axs[location].axhspan(lowerbound['MSError'],p5['MSError'],facecolor=shadeColor)
    axs[location].axhspan(p95['MSError'],max(0.01,upperbound['MSError']),facecolor=shadeColor)
    axs[location].axhline(p5['MSError'],color=s4color,linewidth=LW);
    axs[location].axhline(p95['MSError'],color=s4color,linewidth=LW);

    if (errorBar == True):
        axs[location].errorbar(np.arange(0,flag1), MSError[0:flag1],yerr=zValue*MSEStE[0:flag1], color=s1color, alpha=0.1);
        axs[location].errorbar(np.arange(flag1,flag2), MSError[flag1:flag2],yerr=zValue*MSEStE[flag1:flag2], color=s2color, alpha=0.1);
        axs[location].errorbar(np.arange(flag2, numberOfSamples), MSError[flag2:numberOfSamples],yerr=zValue*MSEStE[flag2:numberOfSamples], color=s3color, alpha=0.1);
    
    '''
    #_______________________________________________________
    # L2 Error plot
    #_______________________________________________________
    location = (1,0)
    axs[location].semilogy(np.arange(0,flag1), L2Error[0:flag1],"^", color=s1color, markersize=s1size, label="NHL = 1");
    axs[location].semilogy(np.arange(flag1,flag2), L2Error[flag1:flag2],"s",color=s2color, markersize=s2size, label="NHL = 2");
    axs[location].semilogy(np.arange(flag2, numberOfSamples), L2Error[flag2:numberOfSamples],"o", color=s3color, markersize=s3size, label="NHL = 3");

    axs[location].legend(loc=0,ncol=3, fontsize=6.3);

    axs[location].set_xlabel(r"ANN ID");

    axs[location].set_ylabel(r"$L_2$ Error");
    axs[location].set_ylim(lowerbound['L2Error'],upperbound['L2Error']);
    axs[location].set_yticks([0.5,1]);

    # highlight 5 and 95 percentile zones with demarkation line
    axs[location].axhspan(lowerbound['L2Error'],p5['L2Error'],facecolor=shadeColor)
    axs[location].axhspan(p95['L2Error'],upperbound['L2Error'],facecolor=shadeColor)
    axs[location].axhline(p5['L2Error'],color=s4color,linewidth=LW);
    axs[location].axhline(p95['L2Error'],color=s4color,linewidth=LW);

    if (errorBar == True):
        axs[location].set_yscale("log", nonpositive='clip')
        axs[location].errorbar(np.arange(0,flag1), L2Error[0:flag1],yerr=zValue*L2StE[0:flag1], color=s1color, alpha=0.1);
        axs[location].errorbar(np.arange(flag1,flag2), L2Error[flag1:flag2],yerr=zValue*L2StE[flag1:flag2], color=s2color, alpha=0.1);
        axs[location].errorbar(np.arange(flag2, numberOfSamples), L2Error[flag2:numberOfSamples],yerr=zValue*L2StE[flag2:numberOfSamples], color=s3color, alpha=0.1);
    

    '''


    #_______________________________________________________
    # L_Min Error plot
    #_______________________________________________________
    location = (0,1);
    axs[location].semilogy(np.arange(0,flag1), MinError[0:flag1],"^", color=s1color, markersize=s1size, label="NHL = 1");
    axs[location].semilogy(np.arange(flag1,flag2), MinError[flag1:flag2],"s",color=s2color, markersize=s2size, label="NHL = 2");
    axs[location].semilogy(np.arange(flag2, numberOfSamples), MinError[flag2:numberOfSamples],"o", color=s3color, markersize=s3size, label="NHL = 3");

    axs[location].legend(loc=1,ncol=3, fontsize=6.3);


    axs[location].set_xlabel(r"ANN ID");
    axs[location].set_ylabel(r"Min Error");
    axs[location].set_ylim(lowerbound['MinError'],upperbound['MinError']*100);
    #axs[location].set_yticks([1e-10,1e-5,1e-1]);

    # highlight 5 and 95 percentile zones with demarkation line
    axs[location].axhspan(lowerbound['MinError'],p5['MinError'],facecolor=shadeColor)
    axs[location].axhspan(p95['MinError'],upperbound['MinError']*100,facecolor=shadeColor)
    axs[location].axhline(p5['MinError'],color=s4color,linewidth=LW);
    axs[location].axhline(p95['MinError'],color=s4color,linewidth=LW);

    if (errorBar == True):
        axs[location].set_yscale("log", nonpositive='clip')
        axs[location].errorbar(np.arange(0,flag1), MinError[0:flag1],yerr=zValue*MinStE[0:flag1], color=s1color, alpha=0.1);
        axs[location].errorbar(np.arange(flag1,flag2), MinError[flag1:flag2],yerr=zValue*MinStE[flag1:flag2], color=s2color, alpha=0.1);
        axs[location].errorbar(np.arange(flag2, numberOfSamples), MinError[flag2:numberOfSamples],yerr=zValue*MinStE[flag2:numberOfSamples], color=s3color, alpha=0.1);

    #_______________________________________________________
    # L_Max Error plot
    #_______________________________________________________
    location = (1,1);
    axs[location].semilogy(np.arange(0,flag1), MaxError[0:flag1],"^", color=s1color, markersize=s1size, label="NHL = 1");
    axs[location].semilogy(np.arange(flag1,flag2), MaxError[flag1:flag2],"s",color=s2color, markersize=s2size, label="NHL = 2");
    axs[location].semilogy(np.arange(flag2, numberOfSamples), MaxError[flag2:numberOfSamples],"o", color=s3color, markersize=s3size, label="NHL = 3");

    axs[location].legend(loc=1,ncol=3, fontsize=6.3);

    axs[location].set_xlabel(r"ANN ID");

    axs[location].set_ylabel(r"Max Error");
    axs[location].set_ylim(lowerbound['MaxError'],upperbound['MaxError']*10);
    #axs[location].set_yticks([1e-1,1e0]);

    # highlight 5 and 95 percentile zones with demarkation line
    axs[location].axhspan(lowerbound['MaxError'],p5['MaxError'],facecolor=shadeColor)
    axs[location].axhspan(p95['MaxError'],upperbound['MaxError']*10,facecolor=shadeColor)
    axs[location].axhline(p5['MaxError'],color=s4color,linewidth=LW);
    axs[location].axhline(p95['MaxError'],color=s4color,linewidth=LW);

    if (errorBar == True):
        axs[location].set_yscale("log", nonpositive='clip')
        axs[location].errorbar(np.arange(0,flag1), MaxError[0:flag1],yerr=zValue*MaxStE[0:flag1], color=s1color, alpha=0.1);
        axs[location].errorbar(np.arange(flag1,flag2), MaxError[flag1:flag2],yerr=zValue*MaxStE[flag1:flag2], color=s2color, alpha=0.1);
        axs[location].errorbar(np.arange(flag2, numberOfSamples), MaxError[flag2:numberOfSamples],yerr=zValue*MaxStE[flag2:numberOfSamples], color=s3color, alpha=0.1);

    plt.savefig(figName);

    
    os.chdir(curDir);


def plotAllErrors(projectName,figName,errorBar):

    TotalRuns = 8;

    projectDir = os.getcwd()+"/output/"+projectName+"/";

    for runNumber in range(TotalRuns):
        print (runNumber);
        plotError(projectName, runNumber,figName,errorBar);
        pass;

    curDir = os.getcwd();
    os.chdir(projectDir);
    os.system("pdftk 0/"+figName+" 1/"+figName+" 2/"+figName+ " 3/"+figName+ " 4/"+figName+ " 5/"+figName+ " 6/"+figName+ " 7/"+figName+ " cat output ErrorPlots.pdf");
    os.chdir(curDir);


if __name__=="__main__":
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern"]
    # Name of the project
    projectName = "Avg";
    errorBar = True;
    figName = "Error"+projectName+".pdf"
    plotAllErrors(projectName, figName,errorBar);
