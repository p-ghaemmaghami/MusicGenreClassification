clc;
clear all;


%% Calculating All EEG feats
SourceFolder='C:\Users\Pouya\Documents\MATLAB\DEAP\Data\';


AllEEGft = nan(32,40,32,4,60);
parfor userID = 1:32
    thisEEGft = load(char(strcat(SourceFolder, 'data_preprocessed_matlab\s',num2str(userID,'%02i'), '.mat')));
    feat = thisEEGft.data(:,1:32,128*3+1:end);
    for clipID = 1:40
        for channelID = 1:32
            for time_segment = 1:60
                [Pxx,F] = periodogram(squeeze(feat(clipID,channelID,time_segment*128-127:time_segment*128)),...
                    rectwin(length(feat(clipID,channelID,time_segment*128-127:time_segment*128))),...
                    length(feat(clipID,channelID,time_segment*128-127:time_segment*128)),128);
                theta = bandpower(Pxx,F,[3 7],'psd');
                alpha = bandpower(Pxx,F,[8 15],'psd');
                beta = bandpower(Pxx,F,[16 31],'psd');
                gamma = bandpower(Pxx,F,[32 45],'psd');

                AllEEGft(userID,clipID,channelID,:,time_segment) = [theta alpha beta gamma];
            end
        end
        disp(clipID)
    end
    disp(userID)
end


%% Calculating All MEG feats
SourceFolder='C:\Users\Pouya\Documents\MATLAB\DECAF\Data\';

SubjectIDs{1}='Sub01SDRM_TFAnalysis';
SubjectIDs{2}='Sub02CHEM_TFAnalysis';
SubjectIDs{3}='Sub03MARM_TFAnalysis';
SubjectIDs{4}='Sub04LUIM_TFAnalysis';
SubjectIDs{5}='Sub05CAMM_TFAnalysis';
SubjectIDs{6}='Sub06SASM_TFAnalysis';
SubjectIDs{7}='Sub07POUM_TFAnalysis';
SubjectIDs{8}='Sub08FRAM_TFAnalysis';
SubjectIDs{9}='Sub09MORM_TFAnalysis';
SubjectIDs{10}='Sub10FARM_TFAnalysis';
SubjectIDs{11}='Sub11FNZM_TFAnalysis';
SubjectIDs{12}='Sub12NEGM_TFAnalysis';
SubjectIDs{13}='Sub13SINM_TFAnalysis';
SubjectIDs{14}='Sub14RADM_TFAnalysis';
SubjectIDs{15}='Sub15NARM_TFAnalysis';
SubjectIDs{16}='Sub16AUDM_TFAnalysis';
SubjectIDs{17}='Sub17VIKM_TFAnalysis';
SubjectIDs{18}='Sub18LATM_TFAnalysis';
SubjectIDs{19}='Sub19TAWM_TFAnalysis';
SubjectIDs{20}='Sub20ELEM_TFAnalysis';
SubjectIDs{21}='Sub21CMLM_TFAnalysis';
SubjectIDs{22}='Sub22RMNM_TFAnalysis';
SubjectIDs{23}='Sub23FRCM_TFAnalysis';
SubjectIDs{24}='Sub24ALSM_TFAnalysis';
SubjectIDs{25}='Sub25SLMM_TFAnalysis';
SubjectIDs{26}='Sub26MRCM_TFAnalysis';
SubjectIDs{27}='Sub27PHLM_TFAnalysis';
SubjectIDs{28}='Sub28KTLM_TFAnalysis';
SubjectIDs{29}='Sub29CRIM_TFAnalysis';
SubjectIDs{30}='Sub30LORM_TFAnalysis';

load('C:\Users\Pouya\Documents\MATLAB\DECAF\Data\MusicPermutationList.mat');

%delta = [1:3];
theta = [4:7];
alpha = [8:15];
beta = [16:31];
gamma = [32:45];
freq_bands = {theta,alpha,beta,gamma};

AllMEGft = nan(30,40,102,4,60);
for userID = 1:length(SubjectIDs)
    %load(char(strcat(SourceFolder, 'DECAF4 TFA Raw\Music\',SubjectIDs(userID), '.mat')));
    load(char(strcat(SourceFolder, 'DECAF4 TFA AR2\Music\',SubjectIDs(userID), '.mat')));
    Perm=PermutationList(userID,:);
    thisMEGft = nan(40,102,4,60);
    for freq=1:4
        thisMEGft(:,:,freq,:) = nanmean(data_tf.powspctrm(:,1:102,freq_bands{freq},5:end-1),3);
    end
    
    AllMEGft(userID,Perm,:,:,:)=thisMEGft;
    disp(userID)
end

%% Calculating All MCA feats
SourceFolder='C:\Users\Pouya\Documents\MATLAB\DECAF\Data\';

AllVideoFt = nan(40,57,60);
AllAudioFt = nan(40,111,60);
MultimediaFt = nan(40,111+57);
AllMCAft = nan(40,111+57,60);

for clipID = 1:40
    load(char(strcat(SourceFolder, 'MCA\Music\', 'VidFt_', num2str(clipID), '.mat')));
    load(char(strcat(SourceFolder, 'MCA\Music\', 'AudFt_', num2str(clipID), '.mat')));
    
    AllVideoFt(clipID,:,:)=[VideoFeatures,VideoFeatures2]';
    AllAudioFt(clipID,:,:)=[AudioFeaturesDEAP,AudioFeaturesPlus]';
    
    AllMCAft(clipID,:,:) = [VideoFeatures,VideoFeatures2,AudioFeaturesDEAP,AudioFeaturesPlus]';
    
    MultimediaFt(clipID,:) = squeeze(mean([VideoFeatures,VideoFeatures2,AudioFeaturesDEAP,AudioFeaturesPlus]',2));
end

MultimediaFt(:,46)=[];
MultimediaFt(:,54)=[];

save('features','AllEEGft','AllMEGft','AllMCAft','-v7.3');% ,'resampledAllVideoFt_M','resampledAllMEGFt_M'