clc;
clear all;


%% Calculating All EEG feats
%% Time-Frequency Settings
Fs = 128;
sec_overlap = 5;
win_size = 50;
Cyc_freq = 1000;

SourceFolder='C:\Users\Pouya\Documents\MATLAB\DEAP\Data\';


physioFT = nan(32,40,32,4,60);
parfor userID = 1:32
    thisft = load(char(strcat(SourceFolder, 'data_preprocessed_matlab\s',num2str(userID,'%02i'), '.mat')));
    feat = thisft.data(:,37:40,128*3+1:end);
    for clipID = 1:40
        for channelID = 1:4
            for time_segment = 1:60
               
                signal = squeeze(feat(clipID,channelID,:));
                [s,f,t,p] = spectrogram(signal,win_size,sec_overlap,Cyc_freq,Fs);
            
            theta = mean(p(3:7,:),1);
            alpha = mean(p(8:15,:),1);
            beta = mean(p(16:31,:),1);
            low_gamma = mean(p(32:45,:),1);
            high_gamma = mean(p(46:100,:),1);
            
            TF_features = [theta;alpha;beta;low_gamma;high_gamma];
                
                [Pxx,F] = periodogram(squeeze(feat(clipID,channelID,time_segment*128-127:time_segment*128)),...
                    rectwin(length(feat(clipID,channelID,time_segment*128-127:time_segment*128))),...
                    length(feat(clipID,channelID,time_segment*128-127:time_segment*128)),128);
                theta = bandpower(Pxx,F,[3 7],'psd');
                alpha = bandpower(Pxx,F,[8 15],'psd');
                beta = bandpower(Pxx,F,[16 31],'psd');
                gamma = bandpower(Pxx,F,[32 45],'psd');

                physioFT(userID,clipID,channelID,:,time_segment) = [theta alpha beta gamma];
            end
        end
        disp(clipID)
    end
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