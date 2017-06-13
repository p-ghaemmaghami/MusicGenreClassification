%% Implemented by Pouya Ghaemmaghami -- p.ghaemmaghami@unitn.it
%%
clc;
clear all;

load('C:\Users\Pouya\Documents\MATLAB\DECAF\Analysis\MusicGenreClassification\Music_Genres_mat.mat');
targets = Music_Genres_mat.majority;

load('C:\Users\Pouya\Documents\MATLAB\DECAF\Analysis\MusicGenreClassification\features.mat');

%% MCA features
MultimediaFt=[];
for j = 1 : 40
    MultimediaFt(j,:) = nanmean(squeeze(AllMCAft(j,:,:)),2);
end
MultimediaFt(:,46)=[];
MultimediaFt(:,54)=[];

% classification
testOutputsMCA = [];
for j = 1 : 40
    trainInd = setdiff(1:40,j);
    trainFeatures = squeeze(MultimediaFt(trainInd,:));
    testFeatures = squeeze((MultimediaFt(j,:)));
    trainTargets = targets(trainInd);
    testTargets = targets(j);
    model = NaiveBayes.fit(trainFeatures,trainTargets);
    %model = fitcsvm(trainFeatures,trainTargets,'Standardize',true,'BoxConstraint',1);
    testOutputsMCA(j) = model.predict(testFeatures);
end
eval = Evaluate(targets,testOutputsMCA');
MCA_ACC = eval(1)
MCA_f1 = eval(6)      

%% Random features
features = randn(100,40,32*4);

rand_acc = []; rand_f1 = [];
for i = 1 : 100
    for j = 1 : 40
        trainInd = setdiff(1:40,j);
        trainFeatures =squeeze(features(i,trainInd,:));
        testFeatures = squeeze((features(i,j,:)));
        trainTargets = targets(trainInd);
        testTargets = targets(j);
        %model = fitcsvm(trainFeatures,trainTargets,'Standardize',true,'BoxConstraint',1);
        model = NaiveBayes.fit(trainFeatures,trainTargets);
        testOutputsRand(j,i) = model.predict(testFeatures');
    end
    
    eval = Evaluate(targets,testOutputsRand(:,i));
    rand_f1(i) = eval(6);       
    rand_acc(i) = eval(1);
    disp(i);
end
mean(rand_f1)
mean(rand_acc)

std(rand_f1)
std(rand_acc)

%% EEG-Dataset-SingleSubject
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

eeg_acc = []; eeg_f1 = [];
for i = 1 : 32
    for j = 1 : 40
        trainInd = setdiff(1:40,j);
        trainFeatures =squeeze(features(i,trainInd,:));
        testFeatures = squeeze((features(i,j,:)));
        trainTargets = targets(trainInd);
        testTargets = targets(j);
        model = NaiveBayes.fit(trainFeatures,trainTargets);
        testOutputsEEG(j,i) = model.predict(testFeatures');
    end
    
    %model = fitcsvm(squeeze(features(i,:,:)),targets,'Standardize',true,'BoxConstraint',1,'Leaveout','on');
    %model = fitcdiscr(squeeze(features(i,:,:)),targets,'Leaveout','on','Gamma',1);
    %[labelEEG(i,:),~,~,~] = kfoldPredict(model);

    eval = Evaluate(targets,testOutputsEEG(:,i));
    %eval = Evaluate(targets,labelEEG(i,:)');
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

%% EEG-Dataset-SingleSubject + MCA
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

eeg_mca_acc = []; eeg_mca_f1 = [];
for i = 1 : 32
    for j = 1 : 40
        trainInd = setdiff(1:40,j);
        trainFeatures = [squeeze(features(i,trainInd,:)) MultimediaFt(trainInd,:)];
        testFeatures = [squeeze((features(i,j,:)))' MultimediaFt(j,:)];
        trainTargets = targets(trainInd);
        testTargets = targets(j);
        %model = fitcsvm(trainFeatures,trainTargets,'Standardize',true,'BoxConstraint',1);
        model = NaiveBayes.fit(trainFeatures,trainTargets);
        testOutputsEEG_MCA(j,i) = model.predict(testFeatures);
    end
    
    eval = Evaluate(targets,testOutputsEEG_MCA(:,i));
    eeg_mca_f1(i) = eval(6);       
    eeg_mca_acc(i) = eval(1);
    disp(i);
end
mean(eeg_mca_acc)
mean(eeg_mca_f1)

std(eeg_mca_acc)
std(eeg_mca_f1)

%% MEG-Dataset-SingleSubject
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

testOutputsMEG=[]; meg_acc = []; meg_f1 = [];
for i = 1 : size(AllMEGft,1)
    for j = 1 : size(AllMEGft,2)
        trainInd = setdiff(1:40,j);
        trainFeatures =squeeze(features(i,trainInd,:));
        testFeatures = squeeze((features(i,j,:)));
        trainTargets = targets(trainInd);
        testTargets = targets(j);
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

%% MEG-Dataset-SingleSubject + MCA
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

testOutputsMEG_MCA=[]; meg_mca_acc = []; meg_mca_f1 = [];
for i = 1 : size(AllMEGft,1)
    for j = 1 : size(AllMEGft,2)
        trainInd = setdiff(1:40,j);
        trainFeatures = [squeeze(features(i,trainInd,:)) MultimediaFt(trainInd,:)];
        testFeatures = [squeeze((features(i,j,:)))' MultimediaFt(j,:)];
        trainTargets = targets(trainInd);
        testTargets = targets(j);
        %model = fitcsvm(trainFeatures,trainTargets,'Standardize',true,'BoxConstraint',1);
        model = NaiveBayes.fit(trainFeatures,trainTargets);
        testOutputsMEG_MCA(j,i) = model.predict(testFeatures);
    end
    
    eval = Evaluate(targets,testOutputsMEG_MCA(:,i));
    meg_mca_f1(i) = eval(6);       
    meg_mca_acc(i) = eval(1);
    disp(i);
end
mean(meg_mca_acc)
mean(meg_mca_f1)

std(meg_mca_acc)
std(meg_mca_f1)

%% Population analysis
majorityVoteEEG=[];majorityVoteEEG_MCA=[];majorityVoteMEG=[];majorityVoteMEG_MCA=[];
for i = 1 : 40
    [~,majorityVoteEEG(i)] = max([sum(testOutputsEEG(i,:)==1),sum(testOutputsEEG(i,:)==2)]);
    [~,majorityVoteEEG_MCA(i)] = max([sum(testOutputsEEG_MCA(i,:)==1),sum(testOutputsEEG_MCA(i,:)==2)]);
    
    [~,majorityVoteMEG(i)] = max([sum(testOutputsMEG(i,:)==1),sum(testOutputsMEG(i,:)==2)]);
    [~,majorityVoteMEG_MCA(i)] = max([sum(testOutputsMEG_MCA(i,:)==1),sum(testOutputsMEG_MCA(i,:)==2)]);
end
majorityVoteEEG=majorityVoteEEG';
majorityVoteEEG_MCA=majorityVoteEEG_MCA';
majorityVoteMEG=majorityVoteMEG';
majorityVoteMEG_MCA=majorityVoteMEG_MCA';

populationACCMEG =sum(majorityVoteMEG==targets)/40
populationACCMEG_MCA =sum(majorityVoteMEG_MCA==targets)/40

populationACCEEG =sum(majorityVoteEEG==targets)/40
populationACCEEG_MCA =sum(majorityVoteEEG_MCA==targets)/40

finalmat = [testOutputsMCA;majorityVoteMEG';majorityVoteMEG_MCA';majorityVoteEEG';majorityVoteEEG_MCA'];

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