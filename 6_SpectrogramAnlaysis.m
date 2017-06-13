clc;
clear all;

SourceFolder='C:\Users\Pouya\Documents\MATLAB\DECAF\Data\Stimuli\Videos-DEAP-all\';

%% Audio Signal & EEG signal
listing = dir(SourceFolder);
listing = {listing.name};
listing = listing(3:end);

for i=1:40
    disp(i)
    
    signal = audioread(char(strcat(SourceFolder,listing{i})));
    signal = signal(:,1);
    signal = downsample(signal,round(length(signal)/(128*60)));
    Audio_signal(i,:) = signal(end-(128*60-1):end);
    figure;plot(signal)
end

plot(squeeze(Audio_signal(18,:)))
save('Audio_signal','Audio_signal');


SourceFolder='C:\Users\Pouya\Documents\MATLAB\DEAP\Data\';
EEG_signal = nan(32,40,32,7680);
for userID = 1:32
    disp(userID)
    
    thisEEGft = load(char(strcat(SourceFolder, 'data_preprocessed_matlab\s',num2str(userID,'%02i'), '.mat')));
    feat = thisEEGft.data(:,1:32,128*3+1:end);
    for clipID = 1:40
        for channelID = 1:32
            
            signal = squeeze(feat(clipID,channelID,:));
            signal = signal(end-(128*60-1):end);

            EEG_signal(userID,clipID,channelID,:) = signal;

        end
        disp(clipID)
    end
end
save('EEG_signal','EEG_signal','-v7.3');

%% correlation in time domain
rMatrix = [];pMatrix=[];
for userID = 1:32
    for sensorID=1:32

        [r,p]=corr(squeeze(EEG_signal(userID,:,sensorID,:))',Audio_signal');
        rMatrix(userID,sensorID)=mean(diag(r));
        pMatrix(userID,sensorID)=pfast(diag(p));
    end
    disp(userID)
end
rMatrix(isnan(rMatrix))=0;
pMatrix(isnan(pMatrix))=1;

for sensorID = 1:32
    r_sensor(sensorID) = mean(squeeze(rMatrix(:,sensorID)));
    p_sensor(sensorID) = pfast(squeeze(pMatrix(:,sensorID)));
end

min(p_sensor)

%% config
Fs = 128;
sec_overlap = 8;%5;
win_size = 64;%50;
Cyc_freq = 128;

%% Audio Spectrogram Analysis
listing = dir(SourceFolder);
listing = {listing.name};
listing = listing(3:end);

TF_features = zeros(40,4,137);%TF_features = zeros(40,4,170);

for i=1:40
    disp(i)
    
    signal = audioread(char(strcat(SourceFolder,listing{i})));
    signal = signal(:,1);
    signal = downsample(signal,round(length(signal)/(128*60)));
    signal = signal(end-(128*60-1):end);
    
    [s,f,t,p] = spectrogram(signal,win_size,sec_overlap,Cyc_freq,Fs);
    
    theta = mean(p(3:7,:),1);
    alpha = mean(p(8:15,:),1);
    beta = mean(p(16:31,:),1);
    low_gamma = mean(p(32:45,:),1);
    
    TF_features(i,:,:) = [theta;alpha;beta;low_gamma];
    
    save('Audio_Spectrogram','TF_features');

end

figure
surf(squeeze(TF_features(i,:,:)))
view(5,70)

imagesc(squeeze(TF_features(i,:,:)))



%% EEG Spectrogram Analysis
SourceFolder='C:\Users\Pouya\Documents\MATLAB\DEAP\Data\';
Fs = 128;
Cyc_freq = 128;

EEG_TF = nan(32,40,32,4,137);%EEG_TF = nan(32,40,32,4,170);
for userID = 1:32
    disp(userID)
    
    thisEEGft = load(char(strcat(SourceFolder, 'data_preprocessed_matlab\s',num2str(userID,'%02i'), '.mat')));
    feat = thisEEGft.data(:,1:32,128*3+1:end);
    for clipID = 1:40
        for channelID = 1:32
            
            signal = squeeze(feat(clipID,channelID,:));
            signal = signal(end-(128*60-1):end);
            %signal = downsample(signal,floor(length(signal)/1000));
            %signal = signal(end-999:end);
            
            [s,f,t,p] = spectrogram(signal,win_size,sec_overlap,Cyc_freq,Fs);

            theta = mean(p(3:7,:),1);
            alpha = mean(p(8:15,:),1);
            beta = mean(p(16:31,:),1);
            low_gamma = mean(p(32:45,:),1);
            
            %figure;
            %imagesc(squeeze([theta;alpha;beta;low_gamma]))


            EEG_TF(userID,clipID,channelID,:,:) = [theta;alpha;beta;low_gamma];

        end
        disp(clipID)
    end
        save('EEG_Spectrogram','EEG_TF');
end

imagesc(squeeze([theta;alpha;beta;low_gamma]))

%% Audio slope
clc;
clear all;

load('EEG_Spectrogram');
load('Audio_Spectrogram');

Audio_slope = squeeze(mean(TF_features,2));
Audio_slope = diff(squeeze(mean(TF_features,2))')';

Audio_slope = smoothts(Audio_slope, 'g', 5);


p=[];r=[];
for sub = 1:32
    disp(sub)
        
    FT_source = reshape(squeeze(EEG_TF(sub,:,[8 26],:,:)),40,2,4*137);
    FT_target = Audio_slope;
    
    FT_source = squeeze(mean(FT_source,2));
    
    for clip = 1:40
        disp(clip)
        trainInd = setdiff(1:40,clip);

        parfor featID = 1:136
            [model,FitInfo] = fitrlinear(FT_source(trainInd,:),FT_target(trainInd,featID),'Learner','leastsquares','Regularization','ridge');%'ridge' 'lasso'
            YHat(clip,featID) = predict(model,squeeze(FT_source(clip,:)));
        end
        %imagesc(zscore(reg_filter))
        %imagesc((reshape(squeeze(YHat(18,:)),4,137)))
        %colormap jet
        %imagesc((reshape(squeeze(zscore(FT_target(18,:))),4,137)))
        %mafdr(p);
        
    end
    
    [r,p] = corr(YHat',FT_target');
    r=diag(r);
    p=diag(p);
    r_sub(sub) = mean(r)
    p_sub(sub) = pfast(p)
end

mean(r_sub)
pfast(p_sub)
std(r_sub)
range(r_sub)
   


%% Linear Regression
clc;
clear all;

load('EEG_Spectrogram');
load('Audio_Spectrogram');

size(EEG_TF)

p=[];r=[];cann_r=[];cmp=3
for sub = 1:32
    disp(sub)
        
    %FT_source = reshape(squeeze(EEG_TF(sub,:,:,:,:)),40,32,4*170);
    FT_source = reshape(squeeze(EEG_TF(sub,:,[8 26],:,:)),40,2,4*137);
    FT_target = reshape(squeeze(TF_features(:,:,:)),40,4*137);
    
    FT_source = squeeze(mean(FT_source,2));
    
    for clip = 1:40
        disp(clip)
        trainInd = setdiff(1:40,clip);
        
%         for sensor = 1:32
%             disp(sensor)
%             for featID = 1:88
%                 model = fitrlinear(squeeze(FT_source(trainInd,sensor,:)),FT_target(trainInd,featID),'KFold',4,'Learner','svm','Regularization','lasso');%'ridge' 'lasso'
%                 loss(sensor,featID) = kfoldLoss(model);
%             end
%         end
%         
%         [M,I] = min(loss);
%         for featID = 1:88
%             selected_FT(:,featID) = FT_source(:,I(featID),featID);
%         end
        parfor featID = 1:4*137
            %model = fitrlinear(selected_FT(trainInd,:),FT_target(trainInd,featID),'Learner','svm','Regularization','lasso');%'ridge' 'lasso'
            %YHat(clip,featID) = predict(model,squeeze(selected_FT(clip,:)));
            %[B,FitInfo] = lasso(FT_source(trainInd,:),FT_target(trainInd,featID),'CV',5);
            [model,FitInfo] = fitrlinear(FT_source(trainInd,:),FT_target(trainInd,featID),'Learner','leastsquares','Regularization','ridge');%'ridge' 'lasso'
            %reg_filter(:,featID) = model.Beta;
            YHat(clip,featID) = predict(model,squeeze(FT_source(clip,:)));
        end
        %imagesc(zscore(reg_filter))
        %imagesc((reshape(squeeze(YHat(18,:)),4,137)))
        %colormap jet
        %imagesc((reshape(squeeze(zscore(FT_target(18,:))),4,137)))
        %mafdr(p);
        
    end
    
    [r,p] = corr(YHat',FT_target');
    r=diag(r);
    p=diag(p);
    r_sub(sub) = mean(r)
    p_sub(sub) = pfast(p)
end

mean(r_sub)
pfast(p_sub)
std(r_sub)
range(r_sub)
    

    %for cmp=2:30
    cmp=3
    for clip=1:40
        trainInd = setdiff(1:40,clip);
        [A,B,r,U,V,stats] = canoncorr(YHat(trainInd,:),FT_target(trainInd,:));
        
        %U = (YHat(trainInd,:)-repmat(mean(YHat(trainInd,:)),39,1))*A;
        %V = (FT_target(trainInd,:)-repmat(mean(FT_target(trainInd,:)),39,1))*B;
        %diag(corr(U(:,1:10)',V(:,1:10)'));
        
        %diag(corr(((YHat(trainInd,:)-repmat(mean(YHat(trainInd,:)),39,1))*A(:,1:cmp))',((FT_target(trainInd,:)-repmat(mean(FT_target(trainInd,:)),39,1))*B(:,1:cmp))'))
        [cann_r(sub,clip),cann_p(sub,clip)] = corr(zscore(((YHat(clip,:)-repmat(mean(YHat(clip,:)),1,1))*A(:,1:cmp))'),zscore(((FT_target(clip,:)-repmat(mean(FT_target(clip,:)),1,1))*B(:,1:cmp))'));
        [r2(sub,clip),mse] = rsquare(zscore(((YHat(clip,:)-repmat(mean(YHat(clip,:)),1,1))*A(:,1:cmp))'),zscore(((FT_target(clip,:)-repmat(mean(FT_target(clip,:)),1,1))*B(:,1:cmp))'));
        
        %[cann_r(sub,clip),cann_p(sub,clip)] = corr((YHat(clip,:)*A(:,1:cmp))',(FT_target(clip,:)*B(:,1:cmp))');
        %[r2(sub,clip),mse] = rsquare((YHat(clip,:)*A(:,1:cmp))',(FT_target(clip,:)*B(:,1:cmp))');
    end
    
%     for i = 1:40
%         y=invspecgram(reshape(YHat(i,:),4,170),128,128,50/4,5); 
%         istft(reshape(YHat(i,:),4,170), 50/4, 128, 128)
%     end
    
end

for i = 1:32
   r_sub(i) = mean(cann_r(i,:));
   p_sub(i) = pfast(cann_p(i,:));
   r2_sub(i) = mean(r2(i,:))
end
mean(r2_sub)
mean(r_sub)
pfast(p_sub)
    
[r,p] = corr(YHat',FT_target');
for clip=1:40
    [r2(sub,clip),mse] = rsquare(YHat(clip,:),FT_target(clip,:));
end
mean(r2(sub,:))
std(r2(sub,:))

r=diag(r)
p=diag(p)
mean(r)
pfast(p)
for sub=1:32
   fusion(sub) = pfast(cann_p(sub,:)) ;
end
pfast(fusion)
    mean(r2(sub,:))
    
    mean(cann_r(sub,:))
    pfast(cann_p(sub,:))

    
    mean(cann_r)
    mean(cann_p)
    pfast(cann_p)
    
    for i=1:32
        fusion(i) = pfast(p_analysis(i,:));
    end

    %end
    max(avg)
    
    for clip=1:40
        [r2(clip),mse] = rsquare(YHat(clip,:),FT_target(clip,:));
    end
    
    [r,p] = corr(YHat',FT_target');
    r = diag(r);
    p = diag(p);
    mean(r)
    
    [A,B,r,U,V,stats] = canoncorr(YHat',FT_target');
    
    [r,p] = corr(A,B);
    
    idx = find(r>0.98);
    idx=1:5;
    
    [r,p] = corr(U(:,1),V(:,1));
    
    A_T=A';
    B_T=B';
    
    X_source = (U(:,idx)*A_T(idx,:))';
    X_target = (V(:,idx)*B_T(idx,:))';
    [r,p] = corr(abs(X_source)',FT_target');
    r = diag(r);
    mean(abs(r))
    
    [r,p] = corr(U,V);
    
    plot(U(:,1)',V(:,1)','.')
    xlabel('0.0025*Disp+0.020*HP-0.000025*Wgt')
    ylabel('-0.17*Accel-0.092*MPG')
    
    
    
    r_analysis(sub,:) = r;
    p_analysis(sub,:) = p;
    
    save('corr_analysis','r_analysis','p_analysis');
    
end
    
    

    
    
    

    %FT_source = squeeze(mean(FT_source,2));

    FT_source_2 = reshape(permute(FT_source, [2 1 3 4]),40*32,4*22);
    FT_target = reshape(FT_target,40,4*22);
    FT_target = repmat(FT_target, [40 1]);
    
    
    %a=FT_source_2(2,:);
    %b=FT_source(1,2,:,:);b=b(:)';
        
        
    for mdl = 1:88
        disp(mdl)
        model = fitrlinear(FT_source,FT_target(:,mdl),'Leaveout','on','Learner','svm','Regularization','lasso');%'ridge' 'lasso'
        label(:,mdl) = kfoldPredict(model);
    end
    
    for clip = 1:40
        y_hat = reshape(squeeze(label(clip,:)),4,22);
        y = reshape(squeeze(FT_target(clip,:)),4,22);
        
        [r(clip),p(clip)] = corr(y_hat(:),y(:))
    end
    
    mean(abs(r))
    
    y_hat = reshape(label,4,22);
    y = reshape(FT_target,4,22);
    
    M = MI_GG(label,FT_target)
    
    
    [r,p] = corr(FT_target',label');
    r=diag(r); p = diag(p);
    mean(r)

    [r(clip),p(clip)] = corr(Y(:), test_target(:))

        
        pfast(p)
        mean(p)
        mean(r)
        

        
end








