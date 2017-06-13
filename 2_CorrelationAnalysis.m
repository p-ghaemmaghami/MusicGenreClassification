%% Implemented by Pouya Ghaemmaghami -- p.ghaemmaghami@unitn.it
%%
clc;
clear all;

SourceFolder='C:\Users\Pouya\Documents\MATLAB\DECAF\Data\';

%% setup
dataset = 'EEG';% EEG
if strcmp(dataset,'MEG')
    num_sensors = 102;
    num_subjects = 30;
    brain_ds_ft = 'AllMEGft';
else
    num_sensors = 32;
    num_subjects = 32;
    brain_ds_ft = 'AllEEGft';
end

%% Calculating Correlations (between sensors tf and video features tf)
load('C:\Users\Pouya\Documents\MATLAB\DECAF\Analysis\MusicGenreClassification\features.mat');
allBrainFt = eval(brain_ds_ft);

rMatrix = zeros(num_subjects,40,4,num_sensors,size(AllMCAft,2));
pMatrix = zeros(num_subjects,40,4,num_sensors,size(AllMCAft,2));

for userID = 1:num_subjects%32
    for fband=1:4
        for clipID=1:40

            %ThsEEGFt=squeeze(AllEEGft(userID,clipID,:,fband,:));
            brainFT=squeeze(allBrainFt(userID,clipID,:,fband,:));
            ThsMCAFt = squeeze(AllMCAft(clipID,:,:));
            [r,p]=corr((brainFT'),ThsMCAFt');%[r,p]=corr((zscore(brainFT')),ThsMCAFt');
            rMatrix(userID,clipID,fband,:,:)=r;
            pMatrix(userID,clipID,fband,:,:)=p;
        end
    end
    disp(userID)
end

%% fusion over clips
% for fixing the nans
rMatrix(isnan(rMatrix))=0;
pMatrix(isnan(pMatrix))=1;

rMatrixFused = zeros(num_subjects,num_sensors,4,size(AllMCAft,2));
pMatrixFused = zeros(num_subjects,num_sensors,4,size(AllMCAft,2));

for userID =1:size(rMatrix,1)
    for sensor=1:size(rMatrix,4)
        for fband=1:size(rMatrix,3)
            for ftId=1:size(rMatrix,5)
                
                Ps=squeeze(pMatrix(userID,:,fband,sensor,ftId));
                Rs=squeeze(rMatrix(userID,:,fband,sensor,ftId));
                
                pMatrixFused(userID,sensor,fband,ftId)=pfast(Ps);
                rMatrixFused(userID,sensor,fband,ftId)=nanmean(Rs);
            end
        end
    end
    disp(userID)
end

%% fusion over subject

rMatrixFused_sub = zeros(num_sensors,4,size(AllMCAft,2));
pMatrixFused_sub = zeros(num_sensors,4,size(AllMCAft,2));

for sensor=1:size(rMatrix,4)
    for fband=1:size(rMatrix,3)
        for ftId=1:size(rMatrix,5)
            
            Ps=squeeze(pMatrixFused(:,sensor,fband,ftId));
            Rs=squeeze(rMatrixFused(:,sensor,fband,ftId));

            pMatrixFused_sub(sensor,fband,ftId)=pfast(Ps);
            rMatrixFused_sub(sensor,fband,ftId)=nanmean(Rs);
        end
    end
end

save('corr_EEG','pMatrixFused_sub','rMatrixFused_sub','-v7.3');% ,'resampledAllVideoFt_M','resampledAllMEGFt_M'
%% plotting 
%
% total MCA features = 168
% 58:113 >> Audio_ft(DEAP) = 56 ft: [MFCC13(13) DMFCC13(13) ACMFCC(13) Energy(1) Formants(4) SpectralFluxF(2) SpectralCentroidF(2) DSMF(2) BERF(2) Pitch(1) ZCrossRateF(2) SilenceRatio(1)]
        
% 1:57 >> Video_ft(49+8) >>
% mean([LightingKey,ColorVariance,!color_hist[40],ShadowPropotion,MedianLightness,MedianSaturation,VisualDetails,Grayness,VisualExcitement,Motion])
% std([LightingKey,ColorVariance,ShadowPropotion,MedianLightness,MedianSaturation,VisualDetails,Grayness,Motion])

selectd_MCA_ft = [1:168];
selectd_MCA_ft = [1,2,43:57,58:70,97:113];
selectd_MCA_ft = [1,44,48,49,52,97,102,106,112,113]; % LightingKey,MedianLightness,VisualExcitement,Motion,ShadowPropotion,Energy,SpectralFluxF,DSMF)

brain_template = 'EEG';% EEG
if strcmp(brain_template,'MEG')
    num_subjects = 30;
else
    num_subjects = 32;
end
th = 0.0000000001/40*num_subjects;

squeeze(pMatrixFused_sub(:,4,selectd_MCA_ft));
squeeze(rMatrixFused_sub(:,4,selectd_MCA_ft));
figure;imagesc(zscore(squeeze(pMatrixFused_sub(:,4,selectd_MCA_ft))));
figure;imagesc(squeeze(rMatrixFused_sub(:,4,selectd_MCA_ft)));


tic
warning off;
f=figure(1);
seg_id=1;
for FtID = 1:5 %168 % -1     [0, 2:4, 6:8]
    for fband = 1:4

        subplot(5,4,(FtID-1)*4+fband);
        %Ps=squeeze(pMatrixFused_sub(:,fband,(seg_id-1)*5+FtID));
        Ps=zscore(squeeze(pMatrixFused_sub(:,fband,selectd_MCA_ft((seg_id-1)*5+FtID))));
        Ps(Ps > th) = 1;
        %Ps=squeeze(rMatrixFused_sub(:,fband,selectd_MCA_ft((seg_id-1)*5+FtID)));
        %Ps(Ps < 0.001) = 1;
        %Ps(Fn_FDR(Ps,0.0001,'BH')<1)=1;
        %corr_topology(squeeze(rMatrixFused(UserID,:,fband,FtID)),Ps,num2str(fband))
        corr_topology(brain_template,squeeze(rMatrixFused_sub(:,fband,selectd_MCA_ft((seg_id-1)*5+FtID)))',Ps,num2str(fband))
    end
    disp((seg_id-1)*5+FtID)
end
toc
