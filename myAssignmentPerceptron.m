function [ Weights] = myAssignmentPerceptron( TrainingData,NumberOfInputs,TrDataSize )
%myPerceptron:  Creates a Perceptron , Trains the Perceptron with the
%training data and generates the Hyperrplane to classify the training Data
%into 2 categories
%   TrainingData   : Consists of Training Inputs and Actual Classification 
%                    Output 
%   NumberOfInputs : Tells the number of input variables or properties we
%   are considering to classify
%                    Line is represented by y=mx+C
%
%   HyperPlaneSlope : Gives the Slope of the HyperPlane (m)
%   HyperPlaneConst : Gives the Constant of the HyperPlane (C)
%   In a Perceptron the Hyperplane is represented by W'X=0; 
%                    where  W' = Vector of Weights including the Bias
%                           X  = Vector of Inputs
%
%~~~~~~~~~~~~~~~~~~~~Initialization~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    EtaMax=1;
    EtaMin = 0.4;
    Eta=0.8;
    
    Bias = 1.5;
    MaxIterations=50000;
    NumberOfDendrites= NumberOfInputs + 1 ;%(Bias + Number of Inputs)
    PerceptronOutput=zeros(TrDataSize,1)';
    Weights =0.2*randn(NumberOfDendrites,1)-0.1;% ones(NumberOfDendrites,1)';% Initial Value being set to 1 just for trial
    Error = zeros(MaxIterations,TrDataSize);
    for iterations = 1:MaxIterations
          RandomDataPicker=ceil(TrDataSize*rand);
          PerceptronInput=[TrainingData(RandomDataPicker,[1:NumberOfInputs]),1];
          PerceptronOutput= PerceptronInput * Weights >=0;
          
          Error(iterations)=TrainingData(TrDataSize,NumberOfInputs+1)- PerceptronOutput ; % Error = Actual Output - Perceptron Output
          Weights=Weights+(2*Eta*(TrainingData(RandomDataPicker,NumberOfInputs+1)-PerceptronOutput)).*PerceptronInput';
    end
    x1= -Weights(NumberOfInputs)/(sum(Weights(1)+Weights(3)));
    x2=-Weights(NumberOfInputs)/(sum(Weights(2)+Weights(3)));
    x=-2:0.01:2;
    %Hyperplane = TrainingData(:,[1:NumberOfInputs]) * Weights; %x2*x/x1;
    
%     plot(TrainingData(1),'b*');
%     hold on  ; 
%     grid on ;
%     plot(TrainingData(2).'r*');
%     x=[0:0.1:4]
%     axis([0,4,0,5]);
%     y=HyperPlaneSlope*x+HyperPlaneConst;
    %plot(x,Hyperplane,'g');
    %hold on;
end

