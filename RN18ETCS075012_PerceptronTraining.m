function [  ] = RN18ETCS075012_PerceptronTraining(  )
%Function Name : RN18ETCS075012_PerceptronTraining
%Author        : Soumya M 
%Date          : 9 Jan 2018
%This function creates Linearly non-separable data and trains the Neural
%Network to create a Hyperplane to separate this data
clear all;
close all;
trainingRadius1 = 3;SeparationRadius=1;
trainingRadius2=7;
TrDataSize=10000;
Tr_Data_Green=CreateDataWithinRadius(trainingRadius1,0,TrDataSize/2,0);
Tr_Data_Red=CreateDataWithinRadius(trainingRadius2,trainingRadius1,TrDataSize/2,1);
plot3(Tr_Data_Green(:,1),Tr_Data_Green(:,2),Tr_Data_Green(:,3),'g*');
hold on
plot3(Tr_Data_Red(:,1),Tr_Data_Red(:,2),Tr_Data_Red(:,3),'r*');
grid on
%Draw the bordering circle
Center=[0 0];
radius=1;
color='r';
%viscircles(Center,radius,'Color',color);

TrainingData = [ Tr_Data_Green;Tr_Data_Red];
[TrDataSize,NumberOfInputs]=size(TrainingData);

NumberOfInputs =NumberOfInputs-1;
Weights = zeros(NumberOfInputs+1,1);
Weights = myAssignmentPerceptron( TrainingData,NumberOfInputs,TrDataSize );
BiasIP=ones(TrDataSize,1);
TrInput=[TrainingData(:,1:NumberOfInputs),BiasIP ];
TrHyperplane =TrainingData*Weights;
% Testing the Trained Perceptron
% plot3(TrInput(:,1),TrInput(:,2),TrHyperplane,'k*');
%First Generate data
TestingDataSize= 5000; TestingRadius=10;
TestingData = CreateDataWithinRadius(TestingRadius,0,TestingDataSize,0);

figure;
%Output=zeros(TestingDataSize,1)';
for i=1:TestingDataSize
     X = [TestingData(i,1:3) 1];
     Y(i,:)=X;
     Output= X*Weights >=0;%
     Hyperplane(i)=X*Weights;
    
    if Output == 0
  
          plot3(TestingData(i,1),TestingData(i,2),TestingData(i,3),'g*');
          hold on
    else
          plot3(TestingData(i,1),TestingData(i,2),TestingData(i,3),'r*');
          hold on
    end
   
        plot3(TestingData(i,1),TestingData(i,2),Hyperplane(i),'k*');

end

title('Testing Patterns')
grid on
axis([-10 10 -10 10 -10 10]);
figure;

Center=[0 0];
Radius=1;
% subplot(1,2,1);
% viscircles(Center,Radius,'Color',color);
% subplot(1,2,2);
% x1= -Weights(NumberOfInputs)/(sum(Weights(1)+Weights(3)));
%     x2=-Weights(NumberOfInputs)/(sum(Weights(2)+Weights(3)));
    x=-2:0.01:2;
    %Weights(1:3)*X(1:3)=Weights(4);
     %    plot(x,SeparationRadius,'b');
     %plot(Hyperplane,'k');
    % hold on ;
     
     %y=Weights'*X; 
   %  plot(Hyperplane,'k');
     grid on ;
      
% y=(-Weights(2)*x-Weights(1))/Weights(3);
% plot(x,y,'k*')
end


