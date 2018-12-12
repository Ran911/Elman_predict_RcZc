load('data_oneshot_train_test_normalization.mat');
%输入数据  
P=train_input;  %训练数据
T=train_output;%输出实际值
TestInput=test_input;%测试数据
TestOutput=test_output; %输出实际值

[pn,minp,maxp,tn,mint,maxt]=premnmx(P,T);
p2= tramnmx(TestInput,minp,maxp);
%创建Elman神经网络  
net_1 = newelm(minmax(pn),[88,2],{'tansig','purelin'},'traingda');% traingd/traingdm/   %trainlm

%设置训练参数  
net_1.trainParam.show = 50;  %显示频率，这里设置为每训练20次显示一次
net_1.trainParam.lr = 0.02;  %学习率
net_1.trainParam.mc = 0.9;  %动量因子
net_1.trainParam.epochs =30000;%10000;  
net_1.trainParam.goal = 1e-3;%2*1e-3;  
net=init(net_1);%初始化网络  
%训练网络  
net = train(net,pn,tn);  
%使用训练好的网络，自定义输入  
PN = sim(net,p2); 
TestResult= postmnmx(PN,mint,maxt);%仿真值反归一化
%理想输出与训练输出的结果进行比较  
E =TestOutput - TestResult; 
%计算误差  
MSE=mse(E);%计算均方误差

figure(1);
plot(TestOutput(1,1:20:end),'bo');
hold on;
plot(TestResult(1,1:20:end),'r*');
legend('R真实值','R预测值');
ylim([0.98,1.02]);%ylim([min(TestOutput(1,:))-0.01,max(TestOutput(1,:))+0.01]);

figure(2);
plot(TestOutput(2,1:20:end),'bo');
hold on;
plot(TestResult(2,1:20:end),'r*');
legend('Z真实值','Z预测值');
ylim([-0.02,0.02]);%ylim([min(TestOutput(2,:))-0.01,max(TestOutput(2,:))+0.01]);

save('Elman.mat','net');