%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             author : Guerrero Andréa, stagiaire LAAS/CNRS               %
%             subject : re-ID in a Camera network !MAIN!                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc; close all;
%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                      Parameters needed                            %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%% ___________________________ add directory ___________________________ %%

addpath(genpath('detecteur_ACF_dollar/'));
addpath(genpath('SDALF-master-matlab2015/'));
addpath(genpath('CamNet_dataset'));
addpath(genpath('xml_io_tools_2010_11_05/'));
addpath(genpath('addLibs/'));

%% ____________________________ PARAMETRES _____________________________ %%

% Imgs param (for different tests)
SUBfac  = 1;    % subsampling factor
H = 128*SUBfac; W = 64*SUBfac; % NORMALIZED dimensions
% symmetries var.
val    = 4;
delta = [H/val W/val]; % border limit (in order to avoid the local minimums at the image border)
varW    = W/5; % variance of gaussian kernel (torso-legs)
alpha = 0.5;

% HSV hist param
NBINs   = [16,16,4]; % hsv quantization
% MSCR
parMSCR.min_margin	= 0.003; %  0.0015;  % Set margin parameter
parMSCR.ainc		= 1.05;
parMSCR.min_size	= 15;
parMSCR.filter_size	= 3;
parMSCR.verbosefl	= 0;  % Uncomment this line to suppress text output
% dynamic MSCR clustering parameters
kmin=1;
kmax=5;
regularize = 1;
th = 1e-2;
covoption=1;
% Matching param.
pyy = 0.4; pcc = 0.6; pee = 0.5;

% maximum number of images per signatures
MAXCLUSTER = 10; % <= THIS NUMBER OF IMAGES WILL BE SPLIT (N/2 for gallery, N/2 for probe)

% head detection (ONLY IF USED)
DIMW    = 24*SUBfac;  % variance of radial gaussian kernel (head)
h_range = [0.0 0.1];  % skin det. param
s_range = [0.3 0.6];

% Do you want to PLOT ?
plotY=0;

% transformation structure for color space conversion, here : RGB values to CIE 1976 L*a*b*
reg  	=   makecform('srgb2lab');
%variance = var; %No need

% MASK ?
maskon = 1;

pas = 3; % On prend une image sur 'pas'

%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                     Extract videos datas                          %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% Data synchronisation using infos time

load('time_infos.mat');
 
temp_1 = time{1,2};
mini1 = 2988; maxi1 = 8000;
date_begin_1 = temp_1(:,mini1);
date_end_1 = temp_1(:,maxi1);
date = [date_begin_1 date_end_1];

% For cam 2 : PRG23
temp_2 = time{1,6};

for i = 1:2 
    i_heure = (temp_2(4,:) == date(4,i));
    ind_heure = find(i_heure);
    good_heure = temp_2(:,i_heure);
    i_min = (good_heure(5,:) == date(5,i));
    ind_min = find(i_min);
    good_min = good_heure(:,i_min);
    i_sec = (good_min(6,:) == date(6,i));
    ind_sec = find(i_sec);
    good_sec = good_min(:,i_sec);
    
    % Frame number for the beginning time
    inter(i) = ind_heure(ind_min(ind_sec(1)));
    
    clear ind_heure ind_min ind_sec good_heure good_min good_sec i_heure i_min i_sec;
end
mini2 = inter(1);
maxi2 = inter(2);

% For cam 3 : PRG7
temp_3 = time{1,3};
for i = 1:2 
    i_heure = (temp_3(4,:) == date(4,i));
    ind_heure = find(i_heure);
    good_heure = temp_3(:,i_heure);
    i_min = (good_heure(5,:) == date(5,i));
    ind_min = find(i_min);
    good_min = good_heure(:,i_min);
    i_sec = (good_min(6,:) == date(6,i));
    ind_sec = find(i_sec);
    good_sec = good_min(:,i_sec);
    
    % Frame number for the beginning time
    inter(i) = ind_heure(ind_min(ind_sec(1)));
    
    clear ind_heure ind_min ind_sec good_heure good_min good_sec i_heure i_min i_sec;
end
mini3 = inter(1);
maxi3 = inter(2);

% For cam 4 : PRG1
temp_4 = time{1,1};
for i = 1:2 
    i_heure = (temp_4(4,:) == date(4,i));
    ind_heure = find(i_heure);
    good_heure = temp_4(:,i_heure);
    i_min = (good_heure(5,:) == date(5,i));
    ind_min = find(i_min);
    good_min = good_heure(:,i_min);
    i_sec = (good_min(6,:) == date(6,i));
    ind_sec = find(i_sec);
    good_sec = good_min(:,i_sec);
    
    % Frame number for the beginning time
    inter(i) = ind_heure(ind_min(ind_sec(1)));
    
    clear ind_heure ind_min ind_sec good_heure good_min good_sec i_heure i_min i_sec;
end
mini4 = inter(1);
maxi4 = inter(2);

clear inter

%% Extract frame for the time interval chosen : /!\ TO DO ONLY THE FIRST TIME TO EXTRACT IMAGES YOU WANT  /!\

%CAM 1 2 3 : test_data_synchro.m

 % Extraction quatrième cam 
%images
repertoire_im = 'Image/S1/PRG1/frame/';
ext = '*.jpg';
files = dir([repertoire_im, ext]);
nfiles = floor(numel(files)/pas);
%mask              
repertoire_msk = 'Image/S1/PRG1/mask/';
ext = '*.jpg';
files = dir([repertoire_msk, ext]);
nMask = floor(numel(files)/pas);

cpt=1; figure;
 for i = (mini4+1):pas:maxi4;%nfiles
     list_images_4{cpt} = imread([repertoire_im,'frame', int2str(i), '.jpg' ]);
     num_frames_4(cpt) = i;
     mask_4{cpt} = imread([repertoire_msk,'frame', int2str(i), '.jpg' ]);
    
%     imshow(list_images_4{cpt});
     cpt = cpt+1;
 end   
 
list_images = [list_images_4];
mask = [mask_4];
num_frames = [num_frames_4];
num_cam = [4.*ones(size(mask_4))];


save('test_data_synchro_4cam/liste_images.mat','list_images', '-v7.3');
save('test_data_synchro_4cam/mask/msk.mat','mask', '-v7.3');
save('test_data_synchro_4cam/nb_frames.mat','num_frames', '-v7.3');
save('test_data_synchro_4cam/num_cam.mat','num_cam', '-v7.3');

n_frames = cpt-1;

% Segmentatio FG/BG loading (+ morfological operations)
if maskon
    msk = mask;
end

%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                     Detection personnes                           %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% /!\ TO LOAD IF YOU HAVE ALREADY EXTRACT IMAGES /!\
load('test_data_synchro_4cam/nb_frames.mat');
load('test_data_synchro_4cam/num_cam.mat');
load('test_data_synchro_4cam/liste_images.mat');

%% ACF detector /!\ TO DO ONLY IF YOU HAVEN'T ALREADY EXTRACT IMAGETTE /!\

% load ACF detector model
load('detector/models/AcfInriaDetector.mat');
acfDetector = detector;
clear detector;


nframes = numel(list_images);

% Open output files
outputAcf = fopen('test_data_synchro_4cam.txt','w+');

%figure;
for f=1:nframes
    frame = list_images{f};
    % Grow image size for better detection (far from camera)
    frame_resize = imresize(frame, 2,'bilinear');
    
   %imshow(frame_resize)
    
    % ACF
    % Detect in frame
    acfdetecs = acfDetect(frame_resize,acfDetector);

    nb_detect(f) = size(acfdetecs,1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % -- Build bounding box -- %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for dets=1:size(acfdetecs,1)
        curdet = acfdetecs(dets,:);
        % Write frame number and detections to file
        fprintf(outputAcf,'%d,%d,%d,%d,%d,%f,%d,%d\n',f,curdet(:), num_cam(f), num_frames(f)); % array of bounding boxes or cell array of bbs
        
       %rectangle('Position',curdet(1:4), 'LineWidth',2, 'EdgeColor','g');
    end
    
    %drawnow;
end
fclose(outputAcf);

clear  c n_detect nb_detect nb_frames_tmp
disp('Finished Processing!')

%% Ground truth /!\ TO DO : IT'S THE GROUND TRUTH  /!\

annotation_4cam;

%% Read bbs infos /!\ TO DO ONLY IF YOU HAVEN'T ALREADY EXTRACT USABLE BBS /!\

bbs = load('test_data_synchro_4cam.txt');
% load('test_data_synchro_4cam/liste_images.mat');
% load('test_data_synchro_4cam/nb_frames.mat');
% load('test_data_synchro_4cam/mask/msk.mat');

% Répertoires de travail
rep_mask = 'test_data_synchro_4cam/mask/';
%msk = mask;
rep_imagette = 'test_data_synchro_4cam/imagette/';

% To obtain imagette and mask
wrote_ids_4cam;

save([rep_mask,'mask_fin'],'mask_fin');     %% saving
save([rep_imagette,'imagette'],'picture');     %% saving

save('test_data_synchro_4cam/stock_xcenter.mat','stock_xcenter');     %% saving
save('test_data_synchro_4cam/stock_ycenter.mat','stock_ycenter');     %% saving
save('test_data_synchro_4cam/stock_bbs.mat','stock_bbs');     %% saving

%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%               Create tracklets and apply kmeans                   %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% Parameters 

dir_e = 'test_data_synchro_4cam/imagette/';
dir_e_list = dir(strcat(dir_e,'/*.jpg'));
nb_k_means = 3;

% Number of cameras
ncameras = 1;
cameras = 4;

%% DON'T TOUCH IF YOU WANT TO USE AGAIN EXISTING DATA !!!

% load('test_data_synchro_4cam/mask/mask_fin.mat');
% load('test_data_synchro_4cam/imagette/imagette.mat');

% % Create tracklets
% N = 15;%number of bin
% tracking_4cam;
% 
% %Apply kmeans on this tracklets
% apply_kmean;

% % /!\ TO DO ONLY IF YOU HAVEN'T MASK /!\
% fprintf('Masking in progress...')
% for n = 1:ncameras
%     for o = 1:length(identity_mean{n})
%        
%         for i = 1:nb_k_means
%             % Extracting histogram to compute bhattacharya distance
%             img     =   picture{pic_nb_mean{n}{o}(i)}; 
%             %masking
%             Mask_pic{n}{o}{i} = roipoly(img);
% % %         figure(1) 
% % %         subplot(1,2,1), imshow(picture{pic_nb_mean{n}{o}(i)});
% % %         subplot(1,2,2), imshow(Mask_pic{n}{o}{i});
% % %         xx = waitforbuttonpress ;
%         end
%      end
% end
% fprintf('OK \n');
% 
% %% saving
% 
% save([rep_mask,'Mask_pic'],'Mask_pic');     
% 
% save('test_data_synchro_4cam/pic_nb_mean.mat','pic_nb_mean', '-v7.3'); % ORDER MASK !!!!
% save('test_data_synchro_4cam/track_mean.mat','track_mean', '-v7.3');
% save('test_data_synchro_4cam/identity_mean.mat','identity_mean', '-v7.3');
% save('test_data_synchro_4cam/bbs_mean.mat','bbs_mean', '-v7.3');
% save('test_data_synchro_4cam/frame_nb_mean.mat','frame_nb_mean', '-v7.3');
% save('test_data_synchro_4cam/histR_mean.mat','histR_mean', '-v7.3');
% save('test_data_synchro_4cam/histG_mean.mat','histG_mean', '-v7.3');
% save('test_data_synchro_4cam/histB_mean.mat','histB_mean', '-v7.3');

%% Join data cam 1 2 3 (cf test_data_synchro.m) and cam 4

% -- TRACKLETS -- %

load('test_data_synchro_4cam/tracklets.mat');
track_tmp = tracklets_final;
clear tracklets_final
load('test_data_synchro/tracklets.mat');
tracklets_final{4} = track_tmp{1};
clear track_tmp

load('test_data_synchro_4cam/tracklets_picnb.mat');
Pic_tmp = Pic_final;
clear Pic_final
load('test_data_synchro/tracklets_picnb.mat');
Pic_final{4} = Pic_tmp{1};
clear Pic_tmp;

load('test_data_synchro_4cam/tracklets_framenb.mat');
Frame_tmp = Frame_number_final;
clear Frame_number_final;
load('test_data_synchro/tracklets_framenb.mat');
Frame_number_final{4} = Frame_tmp{1};

load('test_data_synchro_4cam/tracklets_ids.mat');
id_tmp = identity_final;
clear identity_final
load('test_data_synchro/tracklets_ids.mat');
identity_final{4} = id_tmp{1};
clear id_tmp

load('test_data_synchro_4cam/tracklets_bbs.mat');
bbs_tmp = bbs_final;
clear bbs_final
load('test_data_synchro/tracklets_bbs.mat');
bbs_final{4} = bbs_tmp{1};
clear bbs_tmp


% -- K-MEANS -- %

% Mask 'homemade'
load('test_data_synchro_4cam/mask/Mask_pic.mat');% CAM 4
pic_nb_tmp = Mask_pic;
clear Mask_pic;
load('test_data_synchro/mask/Mask_pic.mat'); % CAM 1 2 3 
Mask_pic{4} = pic_nb_tmp{4};
clear pic_nb_tmp;

% other info
load('test_data_synchro_4cam/pic_nb_mean.mat'); % CAM 4
pic_nb_tmp = pic_nb_mean;
clear pic_nb_mean;
load('test_data_synchro/pic_nb_mean.mat'); % CAM 1 2 3 
pic_nb_mean{4} = pic_nb_tmp{4};
clear pic_nb_tmp;

load('test_data_synchro_4cam/track_mean.mat'); % CAM 4
pic_nb_tmp = track_mean;
clear track_mean;
load('test_data_synchro/track_mean.mat'); % CAM 1 2 3 
track_mean{4} = pic_nb_tmp{4};
clear pic_nb_tmp;

load('test_data_synchro_4cam/identity_mean.mat'); % CAM 4
pic_nb_tmp = identity_mean;
clear identity_mean;
load('test_data_synchro/identity_mean.mat'); % CAM 1 2 3 
identity_mean{4} = pic_nb_tmp{4};
clear pic_nb_tmp;

load('test_data_synchro_4cam/bbs_mean.mat'); %  CAM 4
pic_nb_tmp = bbs_mean;
clear bbs_mean;
load('test_data_synchro/bbs_mean.mat'); % CAM 1 2 3 
bbs_mean{4} = pic_nb_tmp{4};
clear pic_nb_tmp;

load('test_data_synchro_4cam/frame_nb_mean.mat'); %  CAM 4
pic_nb_tmp = frame_nb_mean;
clear frame_nb_mean;
load('test_data_synchro/frame_nb_mean.mat'); % CAM 1 2 3 
frame_nb_mean{4} = pic_nb_tmp{4};
clear pic_nb_tmp;

load('test_data_synchro_4cam/imagette/imagette.mat'); % CAM 4 
picture_tmp = picture;
clear picture;
load('test_data_synchro/imagette/imagette.mat'); % CAM 1 2 3 
picture_tmp_1 = picture;
clear picture;
picture = [picture_tmp_1,picture_tmp];
clear picture_tmp_1 picture_tmp;

load('test_data_synchro_4cam/nb_frames.mat'); % CAM 4 
nb_fr_2 = num_frames;
clear num_frames;
load('test_data_synchro/nb_frames.mat'); % CAM 1 2 3
nb_fr_1 = num_frames;
clear num_frames;
% Frames number of the videos
real_nb_frames = [nb_fr_1,nb_fr_2];
clear nb_fr_1 nb_fr_2;


%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                Extract Begin and end of tracks                    %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% New number of cameras
ncameras = 4;
cameras = [1 2 3 4];

%% Find tracklets corresponding to k-means result
mini = [mini1 mini2 mini3 mini4];

for n = 1:ncameras
    
    track_pic_cam = Pic_final{n};
    
    for t = 1:length(track_pic_cam)
        
        pic_curr_track = track_pic_cam{t};
        
        for k = 1:length(pic_nb_mean{n})
            
            
            % we take one of the 3 k-means pic to find correspongding track
            pic_curr_kmeans = pic_nb_mean{n}{k}(1);
            
            match = find(pic_curr_track == pic_curr_kmeans);
            if ~isempty(match)
                
                frames_track =  bbs_final{n}{k}(:,7);
                
                % We keep the frame nb of the beginning and the ending of the
                % tracklet
                begin_track{n}{k} = ones(size(pic_nb_mean{n}{k})) * (min(frames_track) - mini(n));
                end_track{n}{k} = ones(size(pic_nb_mean{n}{k})) * (max(frames_track) - mini(n));
                
            end 
        end
    end
end

%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                       SDALF application                           %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% LOAD DATASET

% Vectorization of our dataset
l = cellfun('length', frame_nb_mean);

all_mask = []; all_begin = []; all_end = []; all_id = []; all_track = []; 
all_bbs = []; all_frame_nb = []; all_pic_nb = []; all_histR = []; all_histG = []; all_histB = [];
for i = 1:ncameras
    
    id_tmp = cell2mat(identity_mean{i});
    track_tmp = cell2mat(track_mean{i});
    bbs_tmp = cell2mat(bbs_mean{i});
    frame_nb_tmp = cell2mat(frame_nb_mean{i});
    pic_nb_tmp = cell2mat(pic_nb_mean{i});
    frame_begin_tmp = cell2mat(begin_track{i});
    frame_end_tmp = cell2mat(end_track{i});
    
    
    for o = 1:length(identity_mean{i})
        Mask_tmp = cell2mat(Mask_pic{i}{o});
        for j = 1:nb_k_means
            
            all_mask = [all_mask; Mask_tmp(:,(j-1)*W+1:j*W)];
%             figure(1),subplot(1,2,1), imshow(Mask_tmp(:,(j-1)*W+1:j*W));
%                     subplot(1,2,2), imshow(picture{pic_nb_mean{i}{o}(j)});
%             
%             pause(1)
        end
    end
    
    all_id = [all_id; reshape(id_tmp,l(i)*nb_k_means,1)];
    all_track = [all_track; reshape(track_tmp,l(i)*nb_k_means,2)];
    all_bbs = [all_bbs; reshape(bbs_tmp,l(i)*nb_k_means,7)];
    all_frame_nb = [all_frame_nb; reshape(frame_nb_tmp,l(i)*nb_k_means,1)];
    all_pic_nb = [all_pic_nb; reshape(pic_nb_tmp,l(i)*nb_k_means,1)];
    all_begin = [all_begin; reshape(frame_begin_tmp,l(i)*nb_k_means,1)];
    all_end = [all_end; reshape(frame_end_tmp,l(i)*nb_k_means,1)];
    
    
    clear id_tmp track_tmp bbs_tmp frame_nb_tmp histR_tmp histG_tmp histB_tmp frame_begin_tmp frame_end_tmp
end

%Sort by id number
[all_id, index] = sort(all_id);
all_track = all_track(index,:);
all_bbs = all_bbs(index,:);
all_frame_nb = all_frame_nb(index,:);
all_pic_nb = all_pic_nb(index,:);
all_begin = all_begin(index,:);
all_end = all_end(index,:);
% index_mask = (index -ones(size(index))).*H +ones(size(index));
all_mask_tmp = [];
for i = 1:length(index)
    index_mask = (index(i) -1).*H +1;
    all_mask_tmp = [ all_mask_tmp ; all_mask(index_mask:index_mask+H-1,:)];
end
clear all_mask;
all_mask = all_mask_tmp;
clear all_mask_tmp;

% for i = 1:length(index)
%         figure(1) 
%         subplot(1,2,1), imshow(picture{all_pic_nb(i)});
%         subplot(1,2,2), imshow(all_mask_tmp((i-1)*H +1:i*H,:));
%         pause(1)
% end


num_frames = length(all_frame_nb);
avoid_inds = [];
permit_inds = setdiff(1:num_frames,avoid_inds);


if plotY % to see results
    h1 = figure;
end
search_range_H  =   [delta(1),H-delta(1)];
search_range_W  =   [delta(2),W-delta(2)];

%%%%%%%%%%%%%%%%%%%%%%%%%
% -- Loading Dataset -- %
%%%%%%%%%%%%%%%%%%%%%%%%%

dataname = 'camnet';
fprintf([dataname ' Dataset LOADING...'])
hwait = waitbar(0,'Dataset LOADING...');
%figure;

msk_tmp = all_mask;
clear mask_fin
for i=1:length(permit_inds)
    nb = all_pic_nb(i);
    
    img = picture{nb};
    msk_fin = msk_tmp((i-1)*H+1:i*H,:);
    
    % keep useful pic and mask
    dataset(:,:,:,i) = img;
    mask_fin(:,:,i) = msk_fin;

%          subplot(1,2,1), imshow(dataset(:,:,:,i));
%          subplot(1,2,2), imshow(mask_fin(:,:,i));
%          pause(1)
        
    waitbar(i/length(permit_inds),hwait)
end
fprintf('OK \n');
close(hwait)



%% Division in 3 part and kernel map computation

rep = 'test_data_synchro_4cam/';
namefile = [rep 'mapKern_div3_' num2str(SUBfac) '.mat'];
A = dir(namefile);

maxplot = 40;

if ~isempty(A)
    load(namefile)
    fprintf('Division in 3 part and kernel map computation LOADING... OK\n')
else
    fprintf('Division in 3 part and kernel map computation COMPUTATION... ')
    tic
    mapkern_div3;
    tt(1) = toc;
    save(namefile,'MAP_KRNL','TLanti','BUsim','LEGsim','HDanti',...
        'head_det','head_det_flag');     %% saving
    fprintf('OK \n');
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % -- Features extraction -- %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 1) MSCR
% namefile = [rep 'MSCR_' num2str(SUBfac) '.mat'];
% A = dir(namefile);
% if ~isempty(A)
%     load(namefile)
%     fprintf('MSCR LOADING... OK\n')
% else
%     fprintf('MSCR COMPUTATION... ')
%     tic
%     ExtractMSCR;
%     tt(2) = toc;
%     save(namefile,'Blobs');     %% saving
%     fprintf('OK \n');
% end

% 2) Weighted HSV histogram
namefile = [rep 'wHSV_' num2str(SUBfac) '.mat'];
A = dir(namefile);
if ~isempty(A)
    load(namefile)
    fprintf('Weighted HSV hist LOADING... OK\n')
else
    fprintf('Weighted HSV hist COMPUTATION... ')
    tic
    EstimatewHSV; %EstimatewHSV_withoutV;
    tt(3) = toc;
    save(namefile,'whisto2');     %% saving
    fprintf('OK \n');
end


%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                      Re-Identification                            %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%% Dataset preparation

clear num_cam


dir_e = 'test_data_synchro/imagette/'; %CAM 1 2 3
dir_e_list_1 = dir(strcat(dir_e,'/*.jpg'));
clear dir_e

dir_e = 'test_data_synchro_4cam/imagette/';
dir_e_list_2 = dir(strcat(dir_e,'/*.jpg')); %CAM 4
dir_e_list = [dir_e_list_1; dir_e_list_2];

clear dir_e_list_1 dir_e_list_2

% Dataset indexes extraction 
dset = SetDataset_adapte_4cam(permit_inds,dir_e_list,all_pic_nb,real_nb_frames,all_begin,all_end); % contains indexes and all corresponding data of an id

% Dataset adaptation for the paper solver
cpt = 0;
for i = 1:length(dset) % for each id
    N = unique(dset(i).cam); % For 1 id we have N cameras
    
    for n = 1:length(N)
        [l,c] = find(dset(i).cam == N(n));
        
        ped(cpt+n).dynmodels = dset(i).globalindex(c);
        ped(cpt+n).cam = dset(i).cam(c(1));
        ped(cpt+n).id = i;
        
        % Pic and frame number 
        ped(cpt+n).pic = dset(i).pic(c);
        ped(cpt+n).frame = dset(i).frame(c);
        
        % Beginning and ending of the track corresponding
        ped(cpt+n).begin = dset(i).begin(c);
        ped(cpt+n).end = dset(i).end(c);
    end
    cpt = cpt + n;
end
clear cpt N n;

% % -- Graphic representation -- %
% cpt = cpt+1
% for i = 1:length(ped(cpt).pic)
%     x = ped(cpt).pic;
%     id = ped(cpt).id
%     img = picture{x(i)};
%     imshow(img);
%     pause(0.5);
%     k = waitforbuttonpress ;
% end

%% Re-id test

test_number = 1;

for t = 1:test_number
    
    fprintf(['Test ' num2str(t) '...'])
    
    ped_back = ped;
    clear ped
    
    % -- We select randomly MAXCLUSTER ped -- %
    
    cpt = 0; n = 1; m = 1;
    for m = 1:length(ped_back) % for each id in each cam where he appears
       
         if length(ped_back(m).dynmodels) >= nb_k_means % We only keep id that appears at least 3 time (3 detections) in one cam

            in = 1:3; % selection of the 3 data
            ped(n).dynmodels = ped_back(m).dynmodels(in); 
            ped(n).rnddd = 1:length(ped(n).dynmodels);
            
            % Add frame and pic number
            ped(n).pic = ped_back(m).pic(in);
            ped(n).frame = ped_back(m).frame(in);
            
            % Add cam and id number info
            ped(n).cam = ped_back(m).cam;
            ped(n).id = ped_back(m).id;
            
            % Add begin and end of track
            ped(cpt+n).begin = ped_back(m).begin(in);
            ped(cpt+n).end = ped_back(m).end(in);

            
%             % -- Graphic representation -- %     
%             for j = 1:min(length(ped_back(m).dynmodels), MAXCLUSTER)
%                indic = ped_back(m).pic(j);
%                img = picture{indic};
%                mask_f = mask_fin(:,:,indic);
%
%                subplot(1,2,1), imshow(img);
%                subplot(1,2,2), imshow(mask_f);
%                pause(1)
%                k = waitforbuttonpress ;
%             end      

            n = n+1;
         else
            %we don't take in account id appearing less than 3 times in
            %a camera FoV
         end
         
    end
    clear cpt N n;

    
    % -- Matching -- %
    
    %%% 1) Matching MSCR
   % MSCRmatch_ajout3combi; 
    
    %%%%%%%% TEST WITH wHSV ONLY %%%%%%%%
    
    % 2) Matching wHSV
    wHSVmatch_ajout3combi; 
    
    
    % -- Get pairwise similarity -- %
    crossvalidation_ajout3combi;
    
    distance_pair{t} = pairwise_sim_synchro; 
    clear pairwise_sim_synchro;
    
    fprintf('Ok \n')
    ped = ped_back;
end

%% Put pairwise_similarity in shape to compute supervisor part

CPairs = combnk(cameras,2);

pairwise_sim = cell(ncameras,ncameras);
for i = 1:size(CPairs,1)
    for t = 1:test_number
        dist_tmp = distance_pair{t};
        pairwise_tmp(:,:,t) = dist_tmp{CPairs(i,1),CPairs(i,2)};
    end
    pairwise_sim{CPairs(i,1),CPairs(i,2)} = pairwise_tmp;
    clear pairwise_tmp;
end

% % % If you want to save to test supervisor part
% save('pairwise_sim_synchro_4cam_1test.mat','pairwise_sim');
% save('pairwise_4cam_trackinfo.mat','pairwise');

