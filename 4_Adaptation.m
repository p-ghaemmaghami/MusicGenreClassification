%% Implemented by Pouya Ghaemmaghami -- p.ghaemmaghami@unitn.it
%%
clc;
clear all;

load('C:\Users\Pouya\Documents\MATLAB\DECAF\Analysis\MusicGenreClassification\Music_Genres_mat.mat');
targets = Music_Genres_mat.majority;
%load('C:\Users\Pouya\Documents\MATLAB\DECAF\Analysis\MusicGenreClassification\features.mat');
load('C:\Users\Pouya\Documents\MATLAB\DECAF\Analysis\MusicGenreClassification\feats.mat');
load('C:\Users\Pouya\Documents\MATLAB\DECAF\Analysis\MusicGenreClassification\feats_trans.mat');

%% Features Preperation

% MCA
MultimediaFt=[];
for j = 1 : 40
    MultimediaFt(j,:) = nanmean(squeeze(AllMCAft(j,:,:)),2);
end
MultimediaFt(:,46)=[];
MultimediaFt(:,54)=[];
MCA_Ft=MultimediaFt;

% EEG
features = zeros(32,40,32,4);
for userID = 1:32
    for clipID = 1:40
        for channelID = 1:32
            for bandID = 1:4
                features(userID,clipID,channelID,bandID) = log(mean(squeeze(AllEEGft(userID,clipID,channelID,bandID,:))));
            end
        end
    end
    disp(userID)
end
features = reshape(features,[32,40,32*4]);
EEG_Ft = features;

% MEG
features = zeros(size(AllMEGft,1),size(AllMEGft,2),size(AllMEGft,3),size(AllMEGft,4)); % 30,40,102,4
for userID = 1:size(AllMEGft,1)
    for clipID = 1:size(AllMEGft,2)
        for channelID = 1:size(AllMEGft,3)
            for bandID = 1:size(AllMEGft,4)
                features(userID,clipID,channelID,bandID) = log(mean(squeeze(AllMEGft(userID,clipID,channelID,bandID,:))));
            end
        end
    end
    disp(userID)
end
features = reshape(features,[size(AllMEGft,1),size(AllMEGft,2),size(AllMEGft,3)*size(AllMEGft,4)]);
MEG_Ft = features;

save('feats','MCA_Ft','EEG_Ft','MEG_Ft');% ,'resampledAllVideoFt_M','resampledAllMEGFt_M'

%

%% MCA classification
testOutputsMCA = [];
for j = 1 : 40
    trainInd = setdiff(1:40,j);
    trainFeatures = squeeze(MCA_Ft(trainInd,:));
    testFeatures = squeeze((MCA_Ft(j,:)));
    trainTargets = targets(trainInd);
    testTargets = targets(j);
    %model = NaiveBayes.fit(trainFeatures,trainTargets);
    model = fitcsvm(trainFeatures,trainTargets,'Standardize',true,'BoxConstraint',1);
    testOutputsMCA(j) = model.predict(testFeatures);
end
eval = Evaluate(targets,testOutputsMCA');
MCA_ACC = eval(1)
MCA_f1 = eval(6)      

sparseOutputsMCA = [];
for j = 1 : 40
    trainInd = setdiff(1:40,j);
    trainFeatures = squeeze(MCA_Ft_sparse(trainInd,:));
    testFeatures = squeeze((MCA_Ft_sparse(j,:)));
    trainTargets = targets(trainInd);
    testTargets = targets(j);
    %model = NaiveBayes.fit(trainFeatures,trainTargets);
    model = fitcsvm(trainFeatures,trainTargets,'Standardize',true,'BoxConstraint',1);
    sparseOutputsMCA(j) = model.predict(testFeatures);
end
eval = Evaluate(targets,sparseOutputsMCA');
MCA_Sparse_ACC = eval(1)
MCA_Sparse_f1 = eval(6)   


%% EEG-Dataset-SingleSubject
eeg_acc = []; eeg_f1 = [];
for i = 1 : 32
    for j = 1 : 40
        trainInd = setdiff(1:40,j);
        trainFeatures =squeeze(EEG_Ft(i,trainInd,:));
        testFeatures = squeeze((EEG_Ft(i,j,:)));
        trainTargets = targets(trainInd);
        testTargets = targets(j);
        %model = fitcsvm(trainFeatures,trainTargets,'Standardize',true,'BoxConstraint',1);
        model = NaiveBayes.fit(trainFeatures,trainTargets);
        testOutputsEEG(j,i) = model.predict(testFeatures');
    end
    
    eval = Evaluate(targets,testOutputsEEG(:,i));
    eeg_f1(i) = eval(6);       
    eeg_acc(i) = eval(1);
    disp(i);
end
mean(eeg_acc)
mean(eeg_f1)

std(eeg_acc)
std(eeg_f1)

[h,p]=ttest2(rand_acc,eeg_acc)
[h,p]=ttest2(rand_f1,eeg_f1)

%% EEG-Sparse
eeg_sparse_acc = []; eeg_sparse_f1 = [];
sparseOutputsEEG = []
for i = 1 : 32
    for j = 1 : 40
        trainInd = setdiff(1:40,j);
        trainFeatures =squeeze(EEG_Ft_sparse(i,trainInd,:));
        testFeatures = squeeze((EEG_Ft_sparse(i,j,:)));
        trainTargets = targets(trainInd);
        testTargets = targets(j);
        model = fitcsvm(trainFeatures,trainTargets,'Standardize',true,'BoxConstraint',1);
        %NaiveBayes.fit(trainFeatures,trainTargets);
        sparseOutputsEEG(j,i) = model.predict(testFeatures');
    end
    
    eval = Evaluate(targets,sparseOutputsEEG(:,i));
    eeg_sparse_f1(i) = eval(6);       
    eeg_sparse_acc(i) = eval(1);
    disp(i);
end
mean(eeg_sparse_acc)
mean(eeg_sparse_f1)

std(eeg_sparse_f1)
std(eeg_sparse_f1)

%% Adapted-EEG
% Parameters Setting
par.cls_num            =    32
par.nFactor            =    3;
par.step               =    2;
par.win                =    5;
par.rho = 5e-2;
par.lambda1         =       0.01;
par.lambda2         =       0.01;
par.mu              =       0.01;
par.sqrtmu          =       sqrt(par.mu);
par.nu              =       0.1;
par.nIter           =       100;
par.epsilon         =       5e-3;
par.t0              =       5;
par.K               =       5;
par.L               =       25;
param=[]
param.K = par.K;
param.lambda = 0.1;
param.iter = 500; 
param.L = 25;

OutputsEEGAdapted=[]; eeg_adapted_acc = []; eeg_adapted_f1 = [];

%for k = 2 : 20
%k = 9; % this yields best results on svm
k = 7; % this yields best results on NB (more variance than k=3)
for i = 1 : 32
    
    par.K = k;%k_subjetcs(i);
    param.K = k;%k_subjetcs(i);
    parfor j = 1 : 40
        trainInd = setdiff(1:40,j);
        trainFeatures = squeeze(EEG_Ft_sparse(i,trainInd,:));
        testFeatures = squeeze(EEG_Ft_sparse(i,j,:));
        trainTargets = targets(trainInd);
        testTargets = targets(j);

        D = mexTrainDL([trainFeatures';MCA_Ft_sparse(trainInd,:)'], param);

        Dt = D(1:size(trainFeatures,2),:);
        Ds = D(size(trainFeatures,2)+1:end,:);
        W = eye(size(Dt, 2));
        Alpha = mexLasso([trainFeatures';MCA_Ft_sparse(trainInd,:)'], D, param);

        [AlphaT, AlphaS, Xt, Xs, Dt, Ds, Wt, Ws, f] = coupled_DL(Alpha, Alpha, trainFeatures', MCA_Ft_sparse(trainInd,:)', Dt, Ds, W, W, par);

        trainFeatures = AlphaT';
        testFeatures = (Dt' * testFeatures)';% testFeatures * Dt;% * Wt;
        
        %model = fitcsvm(full(trainFeatures),trainTargets,'Standardize',true,'BoxConstraint',1);
        model = NaiveBayes.fit(trainFeatures,trainTargets);
        OutputsEEGAdapted(j,i) = model.predict(testFeatures); %testFeatures
        disp(j)

    end
    
    eval = Evaluate(targets,OutputsEEGAdapted(:,i));
    eeg_adapted_acc(i,k) = eval(1);       
    eeg_adapted_f1(i,k) = eval(6);
    str = sprintf('subject %d accuracy for k=%d is %d',i, k, eeg_adapted_acc(i,k));
    disp(str)
end
%end

mean(eeg_adapted_acc)
[h,p]=ttest2(squeeze(eeg_adapted_acc(:,7)),eeg_acc) % p = 0.02 h=1
[h,p]=ttest2(f,eeg_f1) % p = 0.02 h=1
%end

% mean(maximum_val)
% std(maximum_val)
% 
for i=1:32
    [maximum_val(i),maximum_index(i)] = max(eeg_adapted_acc(i,:))
end
[h,p]=ttest2(maximum_val,MEGacc) % p = 0.02 h=1

save('eeg_adaptation_all','eeg_adapted_acc','eeg_adapted_f1');% ,'resampledAllVideoFt_M','resampledAllMEGFt_M'

k=9
for i=1:30
    acc(i) = eeg_adapted_acc(i,k);
    f(i) = eeg_adapted_f1(i,k);
end

mean(acc)
mean(f)

std(eeg_acc)
std(eeg_f1)
std(f)

[h,p]=ttest2(acc,eeg_acc) % p = 0.02 h=1
[h,p]=ttest2(f,eeg_f1) % p = 0.02 h=1

for k=1:9
    mean(squeeze(eeg_adapted_acc(:,k)))
end
    



%% MEG-Dataset-SingleSubject
testOutputsMEG=[]; meg_acc = []; meg_f1 = [];
for i = 1 : 30
    for j = 1 : 40
        trainInd = setdiff(1:40,j);
        trainFeatures =squeeze(MEG_Ft(i,trainInd,:));
        testFeatures = squeeze((MEG_Ft(i,j,:)));
        trainTargets = targets(trainInd);
        testTargets = targets(j);
        %model = fitcsvm(trainFeatures,trainTargets,'Standardize',true,'BoxConstraint',1);
        model = NaiveBayes.fit(trainFeatures,trainTargets);
        testOutputsMEG(j,i) = model.predict(testFeatures');
    end
    
    eval = Evaluate(targets,testOutputsMEG(:,i));
    meg_f1(i) = eval(6);       
    meg_acc(i) = eval(1);
    disp(i);
end
mean(meg_acc)
mean(meg_f1)

std(meg_acc)
std(meg_f1)

[h,p]=ttest2(rand_acc,meg_acc)
[h,p]=ttest2(rand_f1,meg_f1)


%% MEG-Sparse
sparseOutputsMEG=[]; meg_sparse_acc = []; meg_sparse_f1 = [];
for i = 1 : 30
    for j = 1 : 40
        trainInd = setdiff(1:40,j);
        trainFeatures =squeeze(MEG_Ft_sparse(i,trainInd,:));
        testFeatures = squeeze((MEG_Ft_sparse(i,j,:)));
        trainTargets = targets(trainInd);
        testTargets = targets(j);
        %model = fitcsvm(trainFeatures,trainTargets,'Standardize',true,'BoxConstraint',1);
        model = NaiveBayes.fit(trainFeatures,trainTargets);
        sparseOutputsMEG(j,i) = model.predict(testFeatures');
    end
    
    eval = Evaluate(targets,sparseOutputsMEG(:,i));
    meg_sparse_acc(i) = eval(1);       
    meg_sparse_f1(i) = eval(6);
    disp(i);
end
mean(meg_sparse_acc)
mean(meg_sparse_f1)

std(meg_sparse_acc)
std(meg_sparse_f1)


%% Adapted-MEG
% Parameters Setting
par.cls_num            =    32
par.nFactor            =    3;
par.step               =    2;
par.win                =    5;
par.rho = 5e-2;
par.lambda1         =       0.01;
par.lambda2         =       0.01;
par.mu              =       0.01;
par.sqrtmu          =       sqrt(par.mu);
par.nu              =       0.1;
par.nIter           =       100;
par.epsilon         =       5e-3;
par.t0              =       5;
par.K               =       5;
par.L               =       25;
param=[]
param.K = par.K;
param.lambda = 0.1;
param.iter = 500; 
param.L = 25;

OutputsMEGAdapted=[]; meg_adapted_acc = []; meg_adapted_f1 = [];
%for k = 2 : 20
k = 9; % this yields best results using SVM
k = 7; % this yields best results using NB
for i = 1 : 30
    
    par.K = k;%k_subjetcs(i);
    param.K = k;%k_subjetcs(i);
    parfor j = 1 : 40
        trainInd = setdiff(1:40,j);
        trainFeatures = squeeze(MEG_Ft_sparse(i,trainInd,:));
        testFeatures = squeeze(MEG_Ft_sparse(i,j,:));
        trainTargets = targets(trainInd);
        testTargets = targets(j);

        D = mexTrainDL([trainFeatures';MCA_Ft_sparse(trainInd,:)'], param);

        Dt = D(1:size(trainFeatures,2),:);
        Ds = D(size(trainFeatures,2)+1:end,:);
        W = eye(size(Dt, 2));
        Alpha = mexLasso([trainFeatures';MCA_Ft_sparse(trainInd,:)'], D, param);

        [AlphaT, AlphaS, Xt, Xs, Dt, Ds, Wt, Ws, f] = coupled_DL(Alpha, Alpha, trainFeatures', MCA_Ft_sparse(trainInd,:)', Dt, Ds, W, W, par);

        trainFeatures = AlphaT';
        testFeatures = (Dt' * testFeatures)';% testFeatures * Dt;% * Wt;
        
        %model = fitcsvm(full(trainFeatures),trainTargets,'Standardize',true,'BoxConstraint',1);
        model = NaiveBayes.fit(trainFeatures,trainTargets);
        OutputsMEGAdapted(j,i) = model.predict(testFeatures); %testFeatures
        disp(j)

    end
    
    eval = Evaluate(targets,OutputsMEGAdapted(:,i));
    meg_adapted_acc(i,k) = eval(1);       
    meg_adapted_f1(i,k) = eval(6);
    str = sprintf('subject %d accuracy for k=%d is %d',i, k, meg_adapted_acc(i,k));
    disp(str)
end
%end

% mean(maximum_val)
% std(maximum_val)
% 
% for i=1:30
%     [maximum_val(i),maximum_index(i)] = max(meg_adapted_f1(i,:))
% end
% [h,p]=ttest2(maximum_val,MEGacc) % p = 0.02 h=1
% 
% save('meg_adaptation_all','meg_adapted_acc','meg_adapted_f1');% ,'resampledAllVideoFt_M','resampledAllMEGFt_M'

k=7
for i=1:30
    acc(i) = meg_adapted_acc(i,k);
    f(i) = meg_adapted_f1(i,k);
end

mean(acc)
mean(f)
std(acc)
std(f)

%mean(meg_sparse_acc)
%mean(meg_sparse_f1)

mean(meg_acc)
mean(meg_f1)
std(meg_acc)
std(meg_f1)

[h,p]=ttest2(acc,meg_acc) % p = 0.02 h=1
[h,p]=ttest2(f,meg_f1)

%% Population analysis
majorityVoteEEG=[];majorityVoteMEG=[];majorityVoteEEG_adapted=[];majorityVoteMEG_adapted=[];
for i = 1 : 40
    [~,majorityVoteEEG(i)] = max([sum(testOutputsEEG(i,:)==1),sum(testOutputsEEG(i,:)==2)]);    
    [~,majorityVoteMEG(i)] = max([sum(testOutputsMEG(i,:)==1),sum(testOutputsMEG(i,:)==2)]);
    
    [~,majorityVoteEEG_adapted(i)] = max([sum(OutputsEEGAdapted(i,:)==1),sum(OutputsEEGAdapted(i,:)==2)]);    
    [~,majorityVoteMEG_adapted(i)] = max([sum(OutputsMEGAdapted(i,:)==1),sum(OutputsMEGAdapted(i,:)==2)]);
end
majorityVoteEEG=majorityVoteEEG';
majorityVoteMEG=majorityVoteMEG';
majorityVoteEEG_adapted=majorityVoteEEG_adapted';
majorityVoteMEG_adapted=majorityVoteMEG_adapted';

populationACCMEG =sum(majorityVoteMEG==targets)/40
populationACCEEG =sum(majorityVoteEEG==targets)/40

populationACCMEG_adapted =sum(majorityVoteMEG_adapted==targets)/40
populationACCEEG_adapted =sum(majorityVoteEEG_adapted==targets)/40

%% Confusion Matrix (x=predicted, y=actual)
for i=1:2
    conf_MEGACC(i,:) = normalizedTotalMEG(i,:)/sum(normalizedTotalMEG(i,:));
    conf_EEGACC(i,:) = normalizedTotalEEG(i,:)/sum(normalizedTotalEEG(i,:));
end

xlbl = repmat([1:2]',1,2);
ylbl = repmat([1:2],2,1);

clim = minmax([conf_MEGACC(:); conf_EEGACC(:)]');
figure;

subplot(1,2,1);
imagesc(conf_MEGACC,clim);
title('MEG Features','FontSize',12);
colorbar;
%colormap('Gray');
set(gca,'LineWidth',2,'FontSize',12,'XTickLabel',{'POP','ROCK'},'YTickLabel',{'POP','ROCK'},'XTick',[1,2],'YTick',[1,2]);
text(ylbl(:), xlbl(:), num2str(conf_MEGACC(:),2),'color','K',...
    'HorizontalAlignment','center','VerticalAlignment','middle');

subplot(1,2,2);
imagesc(conf_EEGACC,clim);
title('Adapted-MEG Features','FontSize',12);
colorbar;
%colormap('Gray');
set(gca,'LineWidth',2,'FontSize',12,'XTickLabel',{'POP','ROCK'},'YTickLabel',{'POP','ROCK'},'XTick',[1,2],'YTick',[1,2]);
text(ylbl(:), xlbl(:), num2str(conf_EEGACC(:),2),'color','K',...
    'HorizontalAlignment','center','VerticalAlignment','middle');


diag(conf_MEGACC)'
diag(conf_EEGACC)'








%% ICASSP-single-plots
model_series = [36 42; 54 64; 60 63];
model_error = [11 5; 10 1; 10 2];
h = bar(model_series);
ylim([0 80])
set(h,'BarWidth',1);    % The bars will now touch each other
%set(gca,'YGrid','on')
set(gca,'GridLineStyle','-')
set(gca,'XTicklabel',{'DECAF-MOVIE','DECAF-MUSIC','DEAP-MUSIC'})
set(get(gca,'YLabel'),'String','Accuracy')
lh = legend('Brain','Adapted-Brain');
set(lh,'Location','BestOutside','Orientation','horizontal')
hold on;
numgroups = size(model_series, 1); 
numbars = size(model_series, 2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      errorbar(x, model_series(:,i), model_error(:,i), 'k', 'linestyle', 'none');
end

%% Thesis-single-plots-nb
model_series = [36 42; 52 58; 55 60];
model_error = [11 5; 10 4; 10 6];
h = bar(model_series);
ylim([0 80])
set(h,'BarWidth',1);    % The bars will now touch each other
%set(gca,'YGrid','on')
set(gca,'GridLineStyle','-')
set(gca,'XTicklabel',{'DECAF-MOVIE','DECAF-MUSIC','DEAP-MUSIC'})
set(get(gca,'YLabel'),'String','Accuracy')
lh = legend('Brain','Adapted-Brain');
set(lh,'Location','BestOutside','Orientation','horizontal')
hold on;
numgroups = size(model_series, 1); 
numbars = size(model_series, 2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      errorbar(x, model_series(:,i), model_error(:,i), 'k', 'linestyle', 'none');
end
colormap('copper');


%% Thesis-single-plots-svm
model_series = [37 44; 54 64; 60 63];
model_error = [11 7; 10 1; 10 2];
h = bar(model_series);
ylim([0 80])
set(h,'BarWidth',1);    % The bars will now touch each other
%set(gca,'YGrid','on')
set(gca,'GridLineStyle','-')
set(gca,'XTicklabel',{'DECAF-MOVIE','DECAF-MUSIC','DEAP-MUSIC'})
set(get(gca,'YLabel'),'String','Accuracy')
lh = legend('Brain','Adapted-Brain');
set(lh,'Location','BestOutside','Orientation','horizontal')
hold on;
numgroups = size(model_series, 1); 
numbars = size(model_series, 2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      errorbar(x, model_series(:,i), model_error(:,i), 'k', 'linestyle', 'none');
end
colormap('copper');

%%
%% Adapted-EEG

% Parameters Setting
par.cls_num            =    32
par.nFactor            =    3;
par.step               =    2;
par.win                =    5;
par.rho = 5e-2;
par.lambda1         =       0.01;
par.lambda2         =       0.01;
par.mu              =       0.01;
par.sqrtmu          =       sqrt(par.mu);
par.nu              =       0.1;
par.nIter           =       100;
par.epsilon         =       5e-3;
par.t0              =       5;
par.K               =       5;
par.L               =       25;
param=[]
param.K = par.K;
param.lambda = 0.1;
param.iter = 500; 
param.L = 25;

total = zeros(2,2);
confMatAdaptedMEG = [];
adaptedacc = [];
%sparse_acc = [];
k_list = [2     3     4     5     6     8     9    10    11    13    14    16    17    19    21    25];
k_subjetcs = [2    13     4    11    16     3    13    19    10    11    11     2     9     8     2    25    21     6     9     3     9    14     4     5    21     5     5     3     5    17];

for i = 1 : 30
    par.K = k_subjetcs(i);
    param.K = k_subjetcs(i);
    parfor j = 1 : 36
        trainInd = setdiff(1:36,j);
        trainFeatures = features{i}(trainInd,:);
        testFeatures = features{i}(j,:);
        trainTargets = targets(trainInd);
        testTargets = targets(j);

        D = mexTrainDL([trainFeatures';movie_feat(trainInd,:)'], param);

        Dt = D(1:size(trainFeatures,2),:);
        Ds = D(size(trainFeatures,2)+1:end,:);
        W = eye(size(Dt, 2));
        Alpha = mexLasso([trainFeatures';movie_feat(trainInd,:)'], D, param);

        [AlphaT, AlphaS, Xt, Xs, Dt, Ds, Wt, Ws, f] = coupled_DL(Alpha, Alpha, trainFeatures', movie_feat(trainInd,:)', Dt, Ds, W, W, par);

        trainFeatures = AlphaT';
        testFeatures = (Dt' * testFeatures')';% testFeatures * Dt;% * Wt;

        model = NaiveBayes.fit(trainFeatures,trainTargets);
        testOutputsMEGAdapted(j,i) = model.predict(testFeatures);
        disp(j)

    end
    confMatAdaptedMEG{i} = confusionmat(targets,testOutputsMEGAdapted(:,i));
    total = total + confMatAdaptedMEG{i};
    adaptedacc(i) = sum(diag(confMatAdaptedMEG{i}))/36;
    str = sprintf('subject %d accuracy for k=%d is %d',i, k_subjetcs(i),adaptedacc(i));
    disp(str)
end
normalizedTotalAdpatedMEG = total/sum(sum(total));
AdpatedMEGACC = sum(diag(total))/sum(sum((total)))
mean(adaptedacc)
std(adaptedacc)


[h,p]=ttest2(adaptedacc,MEGacc) % p = 0.0038