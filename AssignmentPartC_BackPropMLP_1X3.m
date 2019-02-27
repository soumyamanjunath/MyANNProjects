%function [  ] = AssignmentPartC_BackPropMLP_1X3(  )
% Name of Function : BackPropMLP
% Author           : Soumya M
% This Function implements the Multi Layer Neural Network using Back
% Propagation Algorithm
%
clear all;
close all ;
clc;
%~~~~~~~~~~~~~~~~~~~Initialization~~~~~~~~~~~~~~~~~~~~~~~
MaxTrials=30;

BIAS_INPUT =  1;
    NumberOfInputs_N=1;
    NumberOfOutputs_R=3; 
    NumOfWeights_W=NumberOfInputs_N + 1 ;
    NumberOfNeuronsInHidden_M=50;
    MaxCycles= 50000;
    InputVector_X=zeros(NumOfWeights_W,1);
    Weights_W=0.2*rand(NumberOfNeuronsInHidden_M,NumOfWeights_W) - 0.1;%[-0.0503 0.0919; -0.0490 0.0090; 0.0012 -0.0723;0.0398 -0.0701; 0.0782 -0.0485]; %
    Weights_V=0.2*(rand(NumberOfOutputs_R,NumberOfNeuronsInHidden_M))-0.1;%[0.0681 -0.0491 -0.0629 -0.0513 0.0859];%
    ExemplarInput=[0:0.05:1]; %Input data
    Omega=2*pi;
    NumberOfSamples=numel(ExemplarInput);
    
    GammaLearningFactor = 0.01;
    GammaMomentumGain = 0.8;

    Delta_W = zeros(NumberOfNeuronsInHidden_M,NumberOfInputs_N);
    Delta_V = zeros(NumberOfOutputs_R,NumberOfNeuronsInHidden_M);

    t=0:0.05:1;
    y1=1+3.*t.^2+sin(Omega.*t)./2;
    y2=3.*(sin(Omega.*t)./2)-(t.^2)./2;
    y3=27.*t.^4-60.*t.^3+39.*t.^2-6.*t;
    
    Output_Y=[y1' y2' y3'];
    for trial=1:MaxTrials
    ErrorForEverySample = zeros(NumberOfSamples,1);
    for epoch=1:MaxCycles

        CurrentDelta_W = zeros(NumberOfNeuronsInHidden_M,NumberOfInputs_N);
        CurrentDelta_V = zeros(NumberOfOutputs_R,NumberOfNeuronsInHidden_M);
        
        %~~~~~~~~~~~~~For every Sample in the Exemplar~~~~~~~~~~~~~
        for(Sample=1:NumberOfSamples)
           %ErrorAtOutput_EY = zeros(NumberOfOutputs_R,1);
            %~~~~~~~~~~~~~~ Output of the Aggregator~~~~~~~~~~~~~~~~~~~~~~~
            InputVector_X=[ExemplarInput(Sample) BIAS_INPUT ]';%nX1 
            ActivationVector_V=Weights_W*InputVector_X;% W_mXn * X_nX1 = V_mX1 

            %~~~~~~~~~~~~Apply Logsigmoid~~~~~~~~~~~~~~~~~~``
            DecisionVector_D= 1./(1+exp(-ActivationVector_V)); %V_mX1=> D-mX1


            %~~~~~~~~~~~~~~~~~~~~Calculate Estimated Output~~~~~~~~~~~~~~~
            EstimatedOutputVector_yHat= Weights_V*DecisionVector_D; %yHat_rX1 =rXm * mX1  

			ActualOutput_Y = Output_Y(Sample,:);
            
            ErrorAtOutput_EY = ActualOutput_Y'- EstimatedOutputVector_yHat;

            ErrorForEverySample(Sample)=sum(ErrorAtOutput_EY.^2); %for plotting

            ErrorAtHidden_ED=  Weights_V'*ErrorAtOutput_EY;

            ActivationError_EA=DecisionVector_D.*(1-DecisionVector_D).*ErrorAtHidden_ED;

            %~~~~~~~~~~~~~~~~Calculate Weight Corrections ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
            CurrentDelta_V=CurrentDelta_V+ErrorAtOutput_EY*DecisionVector_D';
            CurrentDelta_W=CurrentDelta_W+ActivationError_EA*InputVector_X';
            
            
            % Apply Correction to the weights after the SAMPLE - ONLINE TESTING 
%                Weights_V=Weights_V+Delta_V;
%             Weights_W=Weights_W+Delta_W;
        end
       Delta_V=GammaLearningFactor*CurrentDelta_V+GammaMomentumGain*Delta_V;
       Delta_W=GammaLearningFactor*CurrentDelta_W+GammaMomentumGain*Delta_W;
        %~~~~~~~~~~~~~~Find Training Mean Square Error
        trainingMSE(epoch)=(mean(ErrorForEverySample.^2));
       % Apply Correction to the weights after the Entire Batch is processed ~~~~~~~~~~~~~~~~~~~~~~
            Weights_V=Weights_V+Delta_V;
             Weights_W=Weights_W+Delta_W;
%            
       
       
       %
            
    end
    
    %%%% Training Completed Testing to be done Below 
    
    % ~~~~~Generate Testing Data~~~~~~~~~~~
    
    TestingData= 0:0.001:1;
    TestingDataSize = numel(TestingData);
    ZeroOut=zeros(TestingDataSize,1);
    Actual_y1=1+3.*TestingData.^2+sin(Omega.*TestingData)./2;
    Actual_y2=3.*(sin(Omega.*TestingData)./2)-(TestingData.^2)./2;
    Actual_y3=27.*TestingData.^4-60.*TestingData.^3+39.*TestingData.^2-6.*TestingData;
    ActualTestingOutput_Y=[Actual_y1' Actual_y2' Actual_y3'];
    EstimatedOutputVector_yHat = zeros(TestingDataSize,NumberOfOutputs_R);
    BiasInput=ones(TestingDataSize,1);
    for testingsample = 1:TestingDataSize
      
        InputVector_test=[TestingData(testingsample) BIAS_INPUT]';
        
        ActivationVector_V=Weights_W*InputVector_test;

        %~~~~~~~~~~~~Apply Logsigmoid~~~~~~~~~~~~~~~~~~``
        DecisionVector_D= 1./(1+exp(-ActivationVector_V));

        %~~~~~~~~~~~~~~~~~~~~Calculate Estimated Output~~~~~~~~~~~~~~~
        OutputVector_yHat= Weights_V*DecisionVector_D;
        TestingEstimatedOutputVector_yHat(testingsample,:)=OutputVector_yHat;
        TestingErrorAtOutput_EY = ActualTestingOutput_Y(testingsample,:)'- OutputVector_yHat;
       
        MeanError = sum(TestingErrorAtOutput_EY.^2);
        TestingError(testingsample)=MeanError;

    end
    
    %~~~~~~~~~Find Testing Mean Square Error
    TestingMSE = (mean(TestingError))^2;
    
    %~~~~~~~~~`Plot the output from Neural Network and Desired output for
    %comparison
    colors=['b' 'g' 'm'];
    for o=1:NumberOfOutputs_R
        if( trial==MaxTrials)
             figure;
            plot(TestingData,TestingEstimatedOutputVector_yHat(:,o) ,'r',TestingData, ActualTestingOutput_Y(:,o),colors(o));
            grid on ;
            xlabel('Testing Data Input');
            ylabel('Output of Neural Network');
            legend('output','desiredoutput','location','north');
            figure;
            plot3(TestingData,ZeroOut,TestingEstimatedOutputVector_yHat(:,o) ,'r',ZeroOut,TestingData, ActualTestingOutput_Y(:,o),colors(o));
            grid on;
            xlabel('Testing Data Input');
            ylabel('Output of Neural Network');
            ylabel('Testing Data Input');
            zlabel('Output of Neural Network');
            legend('output','desiredoutput','location','north');
        end
    end   
    
    %Plot the Error Trends to compare error trends between Training and
    %Testing
    sample=1:numel(TestingError);
    trainingsample=1:numel(ErrorForEverySample);
    %subplot (2,1,1);
    figure;
    if( trial==MaxTrials)
        plot(sample,TestingError,'b');
    end
    xlabel('Number of samples');
     ylabel('Errors during testing ');
     title('Error Trends during testing');
    %subplot(2,1,2);
    %plot(trainingsample, ErrorForEverySample,'r');
     %xlabel('Number of Epochs');
     %ylabel('Errors during TRaining ');
     %title('Error Trends during training');
    %figure;
     fprintf('Tesing Mean square error = %d\n',TestingMSE);
     fprintf('Training Mean square error = %d\n',trainingMSE(epoch));
     TestingMeanSqErr(trial)=TestingMSE;
     TrainingMeanSqErr(trial)=trainingMSE(epoch);
    end
    MeanTestingMSE=mean(TestingMeanSqErr)
    MeanTrainingMSE=mean(TrainingMeanSqErr);
    figure;
    plot(TestingMeanSqErr,'r');
    xlabel('No of Trials');
    ylabel('MEan Square Error of Trials');
    title('Testing MEan Square Error Variations with Trials');
    
    figure;
    plot(TrainingMeanSqErr,'r');
    xlabel('No of Trials');
    ylabel('MEan Square Error of Trials');
    title('Training MEan Square Error Variations with Trials');
%end

