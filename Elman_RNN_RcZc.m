load('data_oneshot_train_test_normalization.mat');
%��������  
P=train_input;  %ѵ������
T=train_output;%���ʵ��ֵ
TestInput=test_input;%��������
TestOutput=test_output; %���ʵ��ֵ

[pn,minp,maxp,tn,mint,maxt]=premnmx(P,T);
p2= tramnmx(TestInput,minp,maxp);
%����Elman������  
net_1 = newelm(minmax(pn),[88,2],{'tansig','purelin'},'traingda');% traingd/traingdm/   %trainlm

%����ѵ������  
net_1.trainParam.show = 50;  %��ʾƵ�ʣ���������Ϊÿѵ��20����ʾһ��
net_1.trainParam.lr = 0.02;  %ѧϰ��
net_1.trainParam.mc = 0.9;  %��������
net_1.trainParam.epochs =30000;%10000;  
net_1.trainParam.goal = 1e-3;%2*1e-3;  
net=init(net_1);%��ʼ������  
%ѵ������  
net = train(net,pn,tn);  
%ʹ��ѵ���õ����磬�Զ�������  
PN = sim(net,p2); 
TestResult= postmnmx(PN,mint,maxt);%����ֵ����һ��
%���������ѵ������Ľ�����бȽ�  
E =TestOutput - TestResult; 
%�������  
MSE=mse(E);%����������

figure(1);
plot(TestOutput(1,1:20:end),'bo');
hold on;
plot(TestResult(1,1:20:end),'r*');
legend('R��ʵֵ','RԤ��ֵ');
ylim([0.98,1.02]);%ylim([min(TestOutput(1,:))-0.01,max(TestOutput(1,:))+0.01]);

figure(2);
plot(TestOutput(2,1:20:end),'bo');
hold on;
plot(TestResult(2,1:20:end),'r*');
legend('Z��ʵֵ','ZԤ��ֵ');
ylim([-0.02,0.02]);%ylim([min(TestOutput(2,:))-0.01,max(TestOutput(2,:))+0.01]);

save('Elman.mat','net');