%% Implemented by Pouya Ghaemmaghami -- p.ghaemmaghami@unitn.it
%%
clc;
clear all;

Music_Genres_mat.GTZAN_Genres = {'Rock' 'Blues' 'Jazz' 'Metal' 'Pop' 'Disco' 'Hiphop' 'Reggae' 'Country' 'Classical'}';
Music_Genres_mat.DEAP_Genres = {'Pop/Disco/Dance/Techno' 'Rock/Metal'}';

Music_Genres_mat.Pouya = [1,2,1,2,2,2,1,1,2,2,...
                       1,1,1,1,1,1,1,1,1,1,...
                       2,2,1,2,1,1,1,1,1,1,...
                       2,2,2,2,2,2,2,2,2,2]';                                      
                   
Music_Genres_mat.Azad = [2,2,1,1,2,2,1,1,1,2,...
                       1,1,1,1,1,1,1,1,1,1,...
                       2,2,1,1,1,1,1,1,2,1,...
                       2,2,2,2,2,2,2,2,2,2]';  

Music_Genres_mat.Saameh = [1,2,2,1,2,2,1,1,1,2,...
                       1,1,1,1,1,1,1,1,1,1,...
                       2,2,2,1,1,1,1,2,1,1,...
                       2,2,2,1,2,2,2,2,2,2]';
                   
                   
Music_Genres_mat.majority = mode([Music_Genres_mat.Pouya,Music_Genres_mat.Azad,Music_Genres_mat.Saameh],2);

kappa(confusionmat(Music_Genres_mat.Pouya,Music_Genres_mat.Azad))
kappa(confusionmat(Music_Genres_mat.Pouya,Music_Genres_mat.Saameh))
kappa(confusionmat(Music_Genres_mat.Saameh,Music_Genres_mat.Azad))

save('Music_Genres_mat','Music_Genres_mat');
