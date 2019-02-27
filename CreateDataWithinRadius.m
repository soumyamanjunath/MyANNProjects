function [ SampleData ] = CreateDataWithinRadius( MaxRadius,MinRadius, SizeOfData, Classification)
%Name of Function : CreateDataWithinRadius
%Arguments        : Radius : Radius of the circle which is the boundary
%inside which the data has to be generated
%                   SizeOfData : Number of Datapoints to be generated
% This function generates <SizeOfData> number of  data points within the specified radius
%   
SampleData = zeros(SizeOfData,4);
i=1;
while(i<=SizeOfData)
    NewData=randn(2,1)*(MaxRadius-MinRadius)+MinRadius;
    Distance = sqrt(NewData(1)^2+NewData(2)^2);
    if(Distance < MaxRadius && Distance > MinRadius)
       SampleData(i,:)= [NewData(1),NewData(2), Distance, Classification];
       i=i+1;
    end
end
end

