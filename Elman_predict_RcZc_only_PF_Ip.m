%Elman_predict_RcZc  only use PF and Ip
clear
load('C:\Users\Vincent\Desktop\Git\LSTM_predict_RcZc\data_oneshot.mat');
input_data=[data_IP_oneshot data_PF_oneshot];
output_data=data_XZ_oneshot;
train_input_initial=[input_data(1,:);input_data(2,:);input_data(3,:);input_data(4,:)];
for i=1:(fix(3679/5)-1)
    train_input_initial=[train_input_initial;input_data(1+i*5,:);input_data(2+i*5,:);input_data(3+i*5,:);input_data(4+i*5,:)];
end
test_input_initial=input_data(5:5:3679,:);
train_output_initial=[output_data(1,:);output_data(2,:);output_data(3,:);output_data(4,:)];
for i=1:(fix(3679/5)-1)
    train_output_initial=[train_output_initial;output_data(1+i*5,:);output_data(2+i*5,:);output_data(3+i*5,:);output_data(4+i*5,:)];
end
test_output_initial=output_data(5:5:3679,:);

data_num_train=size(train_input_initial,1);
data_num_test=size(test_input_initial,1);

train_input=train_input_initial;
train_output=train_output_initial;
test_input=test_input_initial;
test_output=test_output_initial;

train_input=train_input';
train_output=train_output';
test_input=test_input';
test_output=test_output';
%��������  
P=train_input;  %ѵ������
T=train_output;%���ʵ��ֵ
TestInput=test_input;%��������
TestOutput=test_output; %���ʵ��ֵ

[pn,minp,maxp,tn,mint,maxt]=premnmx(P,T);
p2= tramnmx(TestInput,minp,maxp);
%����Elman������  
net_1 = newelm(minmax(pn),[15,2],{'tansig','purelin'},'traingda');% traingd/traingdm/   %trainlm

%����ѵ������  
net_1.trainParam.show = 50;  %��ʾƵ�ʣ���������Ϊÿѵ��20����ʾһ��
net_1.trainParam.lr = 0.02;  %ѧϰ��
net_1.trainParam.mc = 0.9;  %��������
net_1.trainParam.epochs =50000;%10000;  
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
%ylim([1.65,1.90]);%ylim([0.98,1.02]);%
ylim([min(TestOutput(1,:))-0.01,max(TestOutput(1,:))+0.01]);

figure(2);
plot(TestOutput(2,1:20:end),'bo');
hold on;
plot(TestResult(2,1:20:end),'r*');
legend('Z��ʵֵ','ZԤ��ֵ');
%ylim([-0.05,0.05]);%ylim([-0.02,0.02]);%
ylim([min(TestOutput(2,:))-0.01,max(TestOutput(2,:))+0.01]);

save('Elman_predict_RcZc_only_PF_Ip.mat','net');