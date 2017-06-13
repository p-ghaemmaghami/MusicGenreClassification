function [] = corr_topology(brain_template,rvalue,pvalue,Title)


% load('Schema.mat');
% cfg = [];
% cfg.xlim = [-.01 0.01];
% cfg.zlim = [-0.15,.15];
% cfg.layout = 'neuromag306cmb.lay';
% cfg.comment = 'no';
% cfg.highlight='on';
% cfg.highlightcolor='r';
% cfg.markercolor='g';
% cfg.highlightsymbol='*';
% cfg.style= 'straight';
% cfg.highlightsize=8;
% cfg.colormap = gray(100);
% 
% cfg.parameter='avg';
% 
% 
% 
% Schema.individual=rvalue;
% Schema.avg = rvalue;
% cfg.highlightchannel=Schema.label(pvalue<0.05);
% ft_topoplotER(cfg,Schema); %colorbar;
% title(Title)
warning off;
f=figure(1);
if strcmp(brain_template,'MEG')
    load('Schema_MEG.mat');
else
    load('Schema_EEG.mat');
end
cfg = [];                            
cfg.xlim = [-.01 0.01];                
cfg.zlim = [-.05,.05];
if strcmp(brain_template,'MEG')
    cfg.layout = 'neuromag306cmb.lay';
else
    cfg.layout = 'biosemi32.lay';
end
cfg.comment = 'no';
cfg.highlight='on';
cfg.highlightcolor='r';
cfg.markercolor='g'; 
cfg.highlightsymbol='*';
cfg.style= 'straight';
cfg.highlightsize=8;
cfg.colormap = gray(100);
% cfg.colorbar = 'on';

Schema.individual=rvalue;
cfg.highlightchannel=Schema.label(pvalue<0.05);
ft_topoplotER(cfg,Schema); colorbar;
end