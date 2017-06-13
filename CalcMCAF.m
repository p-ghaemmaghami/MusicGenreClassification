%% Extract Audio Fetures
clc;
clear all;
addpath('mmread');
VideosRoot='D:\Project Audio Reconstruction\MusicVideos\';
FolderContent=dir(VideosRoot);
%addpath('kannumfcc');
addpath('rastamat');
addpath('mb_tracker');
addpath('ACA_Matlab');
SegLength=0.1; %(s)
for  FolderID=40:size(FolderContent,1)
    ThisItem=strcat(VideosRoot,FolderContent(FolderID).name);
    disp(ThisItem);
    if ~isdir(ThisItem)
        [~,FName,ext]=fileparts(ThisItem);
        [~,audioD] = mmread(ThisItem);
        audioD.data=nanmean(audioD.data,2);%Convert audio to mono
        ThsAdSegment=F0_NextAudioSegment([-1,0],audioD,SegLength);
        AudioFeaturesDEAP=[];
        AudioFeaturesPlus=[];
        tic
        while ThsAdSegment(1)+SegLength*audioD.rate-1==ThsAdSegment(2)
            %%
            fprintf('S%d :',round(ThsAdSegment(1)/audioD.rate));
            ThsAData=audioD.data(ThsAdSegment(1):ThsAdSegment(2));
            ThsARate=audioD.rate;
            % MFCC13R=kannumfcc(13,ThsAData,ThsARate);
            if sum(abs(ThsAData))==0
                ThzFeaturesDEAP=nan(1,56);
                ThzFeaturesPlus=nan(1,55);
                error('d');
            else
                [cepstra,aspectrum,pspectrum,LPC13R,LogE]= melfcc(ThsAData, ThsARate); MFCC13R=cepstra'; LPC13R=LPC13R';
                DMFCC13=nanmean(diff(MFCC13R));
                MFCC13=nanmean(MFCC13R);
                ACMFCC=nan(1,13);
                for MID=1:13
                    MFCC13R(~isfinite(MFCC13R))=0;
                    acf=autocorr(MFCC13R(:,MID));
                    ACMFCC(MID)=nanmean(acf(2,end));
                end
                LogEnergy=[nanmean(LogE(isfinite(LogE))),nanstd(LogE(isfinite(LogE)))];
                Energy = nanmean(abs(ThsAData).^2);
                % 12 Coefficients for LPC DEAP [59]
                DLPC13=nanmean(diff(LPC13R));
                LPC13=nanmean(LPC13R);
                %Formants and Pitch
                [F1, F2, F3, F4, Voice, Ptch] = mb_ftracker(ThsAData,ThsARate);
                Formants=[nanmean(F1) nanmean(F2) nanmean(F3) nanmean(F4)];
                Pitch=nanmean(Ptch);
                %Spectral Pitch Chroma
                [SpectralPitchChroma,t] = ComputeFeature('SpectralPitchChroma', ThsAData, ThsARate,[],round(ThsARate/50),round(ThsARate/100));
                PitchChromaF=[nanmean(SpectralPitchChroma,2)',nanstd(SpectralPitchChroma,[],2)'];
                %Spectral Flux
                [SpectralFlux,~] = ComputeFeature('SpectralFlux', ThsAData, ThsARate,[],round(ThsARate/50),round(ThsARate/100));
                SpectralFluxF=[nanmean(SpectralFlux),nanstd(SpectralFlux)];
                %Spectral Centroids
                [SpectralCentroid,~] = ComputeFeature('SpectralCentroid', ThsAData, ThsARate,[],round(ThsARate/50),round(ThsARate/100));
                SpectralCentroidF=[nanmean(SpectralCentroid),nanstd(SpectralCentroid)];
                %Spectral Roll off
                [SpectralRolloff,~] = ComputeFeature('SpectralRolloff', ThsAData, ThsARate,[],round(ThsARate/50),round(ThsARate/100));
                SpectralRolloffF=[nanmean(SpectralRolloff),nanstd(SpectralRolloff)];
                %Zero Crossing Rate
                [ZCrossRate,~] = ComputeFeature('TimeZeroCrossingRate', ThsAData, ThsARate,[],round(ThsARate/50),round(ThsARate/100));
                ZCrossRateF=[nanmean(ZCrossRate),nanstd(ZCrossRate)];
                HZCRR=nansum(sign(ZCrossRate-1.5*nanmean(ZCrossRate))+1)/(2*length(ZCrossRate));%DEAP Ref [60]
                %ZCrossRate=sum(abs(diff(ThsAData>0)))/((ThsAdSegment(2)-ThsAdSegment(1))*ThsARate/1000);
                %BandWidth, Delta Spectrum Mag, Bandwidth Energy ratio
                [Bandwidth,DSM,BER]= Moji_ComputeFeature('Bandwidth,DSM,BER',ThsAData, ThsARate,[],round(ThsARate/50),round(ThsARate/100));
                BandwidthF=[nanmean(Bandwidth),nanstd(Bandwidth)];
                DSMF=[nanmean(DSM),nanstd(DSM)];
                BERF=[nanmean(BER),nanstd(BER)];
                % Silence Ratio DEAP [62]
                frame50Length=round(ThsARate/500);
                FiftyFrames=[];
                Smpl=1;
                while Smpl+frame50Length<length(ThsAData)
                    FiftyFrames=[FiftyFrames,ThsAData(Smpl:Smpl+frame50Length)];
                    Smpl=Smpl+frame50Length;
                end
                %FiftyFrames=reshape(ThsAData,[],50)'; % divide into 50 frames
                RMSs=rms(FiftyFrames); %RMS over each frame
                RMSw=rms(ThsAData); % RMS over the whole window
                RMSTH= RMSw/2;% RMS threshold
                SilenceRatio=sum(RMSs<RMSTH)/length(RMSs);
                %
                ThzFeaturesDEAP=[MFCC13 DMFCC13 ACMFCC Energy Formants SpectralFluxF SpectralCentroidF DSMF BERF Pitch ZCrossRateF SilenceRatio];
                ThzFeaturesPlus=[HZCRR LPC13 DLPC13 PitchChromaF SpectralRolloffF BandwidthF];
            end
            AudioFeaturesDEAP=[AudioFeaturesDEAP; ThzFeaturesDEAP];
            AudioFeaturesPlus=[AudioFeaturesPlus; ThzFeaturesPlus];
            ThsAdSegment=F0_NextAudioSegment(ThsAdSegment,audioD,SegLength);
        end
        toc
        save(['MC_Features\AudFt_' FName '.mat'],'AudioFeaturesDEAP','AudioFeaturesPlus');
        fprintf(' This is done\n');
    end
end
%% Extract Video Fetures
clc;
clear all;
addpath('mmread');
VideosRoot='G:\COAF\MMCA\Videos\';
FolderContent=dir(VideosRoot);
addpath('colorspace');
for  FolderID=1:size(FolderContent,1)
    ThisItem=strcat(VideosRoot,FolderContent(FolderID).name);
    disp(ThisItem);
    if ~isdir(ThisItem)
        [~,FName,ext]=fileparts(ThisItem);
        [videoD,~] = mmread(ThisItem);
        ThsVdSegment=F0_NextVideoSegment([-1,0],videoD);
        VideoFeatures=[];
        VideoFeatures2=[];
        ThsVFRate=round(videoD.rate);
        % For Optical Flows .. Motion
        opticalFlow = vision.OpticalFlow('ReferenceFrameDelay', 1);
        opticalFlow.Method = 'Lucas-Kanade';
        opticalFlow.OutputValue = 'Magnitude-squared';%'Horizontal and vertical components in complex form';
        converter = vision.ImageDataTypeConverter;
        %
        tic
        while ThsVdSegment(1)<ThsVdSegment(2)
            fprintf('S%d :',round(ThsVdSegment(1)/ThsVFRate));
            ThsVData=videoD.frames(ThsVdSegment(1):ThsVdSegment(2));
            ThsVData=rmfield(ThsVData,'colormap');
            ThsVData=squeeze(struct2cell(ThsVData));
            LightingKey=[];
            ColorVariance=[];
            Color20HistH=[];
            Color20HistV=[];
            ShadowPropotion=[];
            MedianLightness=[];
            MedianSaturation=[];
            VisualDetails=[];
            Grayness=[];
            %             ColorEnergy=[]; %DEAP [30]
            for indx=1:length(ThsVData)
                ThsFrm=ThsVData{indx};
                %Frm_HSV = rgb2hsv(ThsFrm);
                Frm_HSV = colorspace(['HSV','<-RGB'],ThsFrm);
                % Lighting Key %DEAP
                Vs=reshape(Frm_HSV(:,:,3),1,[]);
                LightingKey=[LightingKey mean(Vs)*std(Vs)];
                % 20 bins color hist % DEAP [57]
                Hs=reshape(Frm_HSV(:,:,1),1,[]);
                Color20HistH=[Color20HistH; hist(Hs,20)/length(Hs)];
                Color20HistV=[Color20HistV; hist(Vs,20)/length(Vs)];
                % Color Variance %DEAP [30]
                Frm_LUV = colorspace(['Luv','<-RGB'],ThsFrm);
                Frm_LUV_Vect=[reshape(Frm_LUV(1,:,:),[],1),reshape(Frm_LUV(2,:,:),[],1),reshape(Frm_LUV(3,:,:),[],1)];
                ColorVariance=[ColorVariance det(cov(Frm_LUV_Vect))];
                %Median Lightness [30] [56]
                Frm_HSL = colorspace(['HSL','<-RGB'],ThsFrm);
                Ls=reshape(Frm_HSL(:,:,3),1,[]);
                MedianLightness=[MedianLightness median(Ls)];
                %Shadow propotion DEAP [30]
                ShadowPropotion=[ShadowPropotion sum(Ls<.18)/length(Ls)];
                %                 %Color Energy %DEAP [30]
                %                 ColorEnergy=[];
                Ss=reshape(Frm_HSL(:,:,2),1,[]); %Ranges from 0 to 360 degree 0:RED 240:BLUE 360:RED
                MedianSaturation=[MedianSaturation median(Ss)]; %DEAP [30] --> [19] :Arousal
                %                 Hs=reshape(Frm_HSL(:,:,1),1,[]); %Ranges from 0 to 360 degree 0:RED 240:BLUE 360:RED
                %                 Es=Hs;
                %                 Es(Hs<=240)=(-Es(Hs<=240)/240+1)/2+.75;
                %                 Es(Hs>240)=((Es(Hs>240)-240)/120)/2+.75;
                %                 sum(Es.*Ss.*Ls)
                %Grayness DEAP [30]
                Grayness=[Grayness sum(Ss<.2)/length(Ss)];
                %Visual Details DEAP [30]
                GLCM=graycomatrix(rgb2gray(ThsFrm),'NumLevels',256,'Offset',[0 2; -2 2; -2 0; -2 -2]);
                VisualDetails=[VisualDetails nanmean(nanmean(nanmean(GLCM)))];
            end
            Color20HistH=mean(Color20HistH); %DEAP [57]
            Color20HistV=mean(Color20HistV); %DEAP [57]
            %Visual Excitement DEAP [30]
            ThsFrm1=ThsVData{1}; % Initiating the first frame
            thresh_fd=2;
            % Calculating the pixle categories for 20x20 separation
            Height=size(ThsFrm1,1);
            Width=size(ThsFrm1,2);
            WidthSep20=ones(1,20)*round(Width/20);
            WidthSep20(end)=WidthSep20(end)+Width-sum(WidthSep20);
            HeightSep20=ones(1,20)*round(Height/20);
            HeightSep20(end)=HeightSep20(end)+Height-sum(HeightSep20);
            Xfds=[];
            for indx=2:length(ThsVData)
                ThsFrm2=ThsVData{indx};
                Frm1_HSV = colorspace(['HSV','<-RGB'],ThsFrm);
                Frm1_LUV = colorspace(['Luv','<-RGB'],ThsFrm1);
                Frm2_LUV = colorspace(['Luv','<-RGB'],ThsFrm2);
                L1Sep=mat2cell(squeeze(Frm1_LUV(:,:,1)), HeightSep20, WidthSep20);
                U1Sep=mat2cell(squeeze(Frm1_LUV(:,:,2)), HeightSep20, WidthSep20);
                V1Sep=mat2cell(squeeze(Frm1_LUV(:,:,3)), HeightSep20, WidthSep20);
                L2Sep=mat2cell(squeeze(Frm2_LUV(:,:,1)), HeightSep20, WidthSep20);
                U2Sep=mat2cell(squeeze(Frm2_LUV(:,:,2)), HeightSep20, WidthSep20);
                V2Sep=mat2cell(squeeze(Frm2_LUV(:,:,3)), HeightSep20, WidthSep20);
                L1=cellfun(@mean2,L1Sep);
                U1=cellfun(@mean2,U1Sep);
                V1=cellfun(@mean2,V1Sep);
                L2=cellfun(@mean2,L2Sep);
                U2=cellfun(@mean2,U2Sep);
                V2=cellfun(@mean2,V2Sep);
                s_avL=mean2(Frm1_HSV(:,:,3));
                if s_avL>=1/3
                    sL=1/3;
                else
                    sL=1/3+(s_avL-1/3)^2;
                end
                xfd=sqrt(((L1-L2).^2)*sL+((U1-U2).^2+(V1-V2).^2)*1/3);
                Xfds=[Xfds mean2(heaviside(xfd-thresh_fd))];
                ThsFrm1=ThsFrm2; %replacing the first with the second frame
            end
            Nc=length(ThsVData);
            VisualExcitement=(10/Nc)*sum(Xfds+Xfds.^0.75);
            % Motion fetures
            %Set the First Frame
            ThsFrm=ThsVData{1};
            ThsFrmGray=rgb2gray(ThsFrm);
            ThsFrmGray = step(converter, ThsFrmGray);
            of = step(opticalFlow, ThsFrmGray);
            Motion=[];
            for indx=1:length(ThsVData)
                ThsFrm=ThsVData{indx};
                ThsFrmGray=rgb2gray(ThsFrm);
                ThsFrmGray = step(converter, ThsFrmGray);
                of = step(opticalFlow, ThsFrmGray);
                Motion=[Motion nanmean(reshape(of,1,[]))];
            end
            ThsFeatures =[nanmean(LightingKey),nanmean(ColorVariance),...
                Color20HistH,Color20HistV,nanmean(ShadowPropotion),...
                nanmean(MedianLightness),nanmean(MedianSaturation),...
                nanmean(VisualDetails),nanmean(Grayness),VisualExcitement,...
                nanmean(Motion)];
            ThsFeatures2 =[nanstd(LightingKey),nanstd(ColorVariance),...
                nanstd(ShadowPropotion),nanstd(MedianLightness),nanstd(MedianSaturation),...
                nanstd(VisualDetails),nanstd(Grayness),nanstd(Motion)];
            VideoFeatures=[VideoFeatures; ThsFeatures];
            VideoFeatures2=[VideoFeatures2; ThsFeatures2];
            ThsVdSegment=F0_NextVideoSegment(ThsVdSegment,videoD);
        end
        toc
        save(['MV_Features\VidFt_' FName '.mat'],'VideoFeatures','VideoFeatures2');
        fprintf(' This is done\n');
    end
end
%% PutFeaturestogether
clc;
clear all;
addpath('mmread');
VideosRoot='D:\COAF\MMCA\Videos\';
FolderContent=dir(VideosRoot);
