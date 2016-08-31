%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             author : Guerrero Andréa, stagiaire LAAS/CNRS               %
%             subject : re-ID in a Camera network !MAIN!                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc; close all;
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

%% Extract frame for the time interval chosen   /!\ TO DO ONLY THE FIRST TIME TO EXTRACT IMAGES YOU WANT  /!\

% Extraction premiere cam
%images
repertoire_im = 'Image/S1/PRG6/frame/';
ext = '*.jpg';
files = dir([repertoire_im, ext]);
nfiles = floor(numel(files)/pas);
%mask      
repertoire_msk = 'Image/S1/PRG6/mask/';
ext = '*.jpg';
files = dir([repertoire_msk, ext]);
nMask = floor(numel(files)/pas);

cpt=1;
figure;
 for i = (mini1+1):pas:maxi1;%nfiles
     list_images_1{cpt}= imread([repertoire_im,'frame', int2str(i), '.jpg' ]);
     num_frames_1(cpt) = i;
     mask_1{cpt} = imread([repertoire_msk,'frame', int2str(i), '.jpg' ]);
     
     %imshow(list_images_1{cpt});
     cpt = cpt+1;
 end

 
% Extraction deuxieme cam 
%images
repertoire_im = 'Image/S1/PRG23/frame/';
ext = '*.jpg';
files = dir([repertoire_im, ext]);
nfiles = floor(numel(files)/pas);
%mask              
repertoire_msk = 'Image/S1/PRG23/mask/';
ext = '*.jpg';
files = dir([repertoire_msk, ext]);
nMask = floor(numel(files)/pas);

cpt=1; figure;
 for i = (mini2+1):pas:maxi2;%nfiles
     list_images_2{cpt} = imread([repertoire_im,'frame', int2str(i), '.jpg' ]);
     num_frames_2(cpt) = i;
     mask_2{cpt} = imread([repertoire_msk,'frame', int2str(i), '.jpg' ]);
    
    %imshow(list_images_2{cpt});
     cpt = cpt+1;
 end       
 
% Extraction troisieme cam 
%images
repertoire_im = 'Image/S1/PRG7/frame/';
ext = '*.jpg';
files = dir([repertoire_im, ext]);
nfiles = floor(numel(files)/pas);
%mask              
repertoire_msk = 'Image/S1/PRG7/mask/';
ext = '*.jpg';
files = dir([repertoire_msk, ext]);
nMask = floor(numel(files)/pas);

cpt=1; figure;
 for i = (mini3+1):pas:maxi3;%nfiles
     list_images_3{cpt} = imread([repertoire_im,'frame', int2str(i), '.jpg' ]);
     num_frames_3(cpt) = i;
     mask_3{cpt} = imread([repertoire_msk,'frame', int2str(i), '.jpg' ]);
    
    %imshow(list_images_3{cpt});
     cpt = cpt+1;
 end       
 
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
    
    imshow(list_images_4{cpt});
     cpt = cpt+1;
 end   
 
list_images = [list_images_1 list_images_2 list_images_3 list_images_4];
mask = [mask_1 mask_2 mask_3 mask_4];
num_frames = [num_frames_1 num_frames_2 num_frames_3 num_frames_4];
num_cam = [ones(size(mask_1)) 2.*ones(size(mask_2)) 3.*ones(size(mask_3)) 4.*ones(size(mask_4))];

save('test_data_synchro/liste_images.mat','list_images', '-v7.3');
save('test_data_synchro/mask/msk.mat','mask', '-v7.3');
save('test_data_synchro/nb_frames.mat','num_frames', '-v7.3');
save('test_data_synchro/num_cam.mat','num_cam', '-v7.3');

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
load('test_data_synchro/nb_frames.mat');
load('test_data_synchro/num_cam.mat');
load('test_data_synchro/liste_images.mat');
%%  ACF detector   /!\ TO DO ONLY IF YOU HAVEN'T ALREADY EXTRACT IMAGETTE /!\

% load ACF detector model
load('detector/models/AcfInriaDetector.mat');
acfDetector = detector;
clear detector;

nframes = numel(list_images);

% Open output files
outputAcf = fopen('test_data_synchro.txt','w+');

figure;
for f=100:nframes
    frame = list_images{f};
    % Grow image size for better detection (far from camera)
    frame_resize = imresize(frame, 2,'bilinear');
    
   imshow(frame_resize)
    
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
       % fprintf(outputAcf,'%d,%d,%d,%d,%d,%f,%d,%d\n',f,curdet(:), num_cam(f), num_frames(f)); % array of bounding boxes or cell array of bbs
        
       rectangle('Position',curdet(1:4), 'LineWidth',2, 'EdgeColor','g');
    end
    
    drawnow;
end
fclose(outputAcf);

clear  c n_detect nb_detect nb_frames_tmp
disp('Finished Processing!')

%% Ground truth /!\ TO DO : IT'S THE GROUND TRUTH  /!\

annotation;

%% Read bbs infos /!\ TO DO ONLY IF YOU HAVEN'T ALREADY EXTRACT USABLE BBS /!\

bbs = load('test_data_synchro.txt');
% load('test_data_synchro/liste_images.mat');
% load('test_data_synchro/nb_frames.mat');
% load('test_data_synchro/mask/msk.mat');
% Répertoires de travail
rep_mask = 'test_data_synchro/mask/';
%msk = mask;
rep_imagette = 'test_data_synchro/imagette/';

% To obtain imagette and mask
wrote_ids;

save([rep_mask,'mask_fin'],'mask_fin');     %% saving
save([rep_imagette,'imagette'],'picture');     %% saving

save('test_data_synchro/stock_xcenter.mat','stock_xcenter');     %% saving
save('test_data_synchro/stock_ycenter.mat','stock_ycenter');     %% saving
save('test_data_synchro/stock_bbs.mat','stock_bbs');     %% saving

%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%               Create tracklets and apply kmeans                   %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%% Create tracklets and apply kmeans
load('test_data_synchro/imagette/imagette.mat');
load('test_data_synchro/mask/mask_fin.mat');

dir_e = 'test_data_synchro/imagette/';
dir_e_list = dir(strcat(dir_e,'/*.jpg'));
nb_k_means = 3;

%% DON'T TOUCH IF YOU WANT TO USE AGAIN EXISTING DATA !!!

% % % Create tracklets
% % N = 15;%number of bin
% % tracking;
% % 
% % %Apply kmeans on this tracklets
% % 
% % apply_kmean;


% fprintf('Masking in progress...')
% for n = 3%1:ncameras
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
% 
% 
% save([rep_mask,'Mask_pic'],'Mask_pic');     %% saving
% 
% save('test_data_synchro/pic_nb_mean.mat','pic_nb_mean', '-v7.3'); % ORDER MASK !!!!
% save('test_data_synchro/track_mean.mat','track_mean', '-v7.3');
% save('test_data_synchro/identity_mean.mat','identity_mean', '-v7.3');
% save('test_data_synchro/bbs_mean.mat','bbs_mean', '-v7.3');
% save('test_data_synchro/frame_nb_mean.mat','frame_nb_mean', '-v7.3');
% save('test_data_synchro/histR_mean.mat','histR_mean', '-v7.3');
% save('test_data_synchro/histG_mean.mat','histG_mean', '-v7.3');
% save('test_data_synchro/histB_mean.mat','histB_mean', '-v7.3');

%% Apply SDALF

% Mask 'homemade'
load('test_data_synchro/mask/Mask_pic.mat');

% k-means OUTPUT k=3
load('test_data_synchro/pic_nb_mean.mat'); % ORDER MASK !!!!
load('test_data_synchro/track_mean.mat');
load('test_data_synchro/identity_mean.mat');
load('test_data_synchro/bbs_mean.mat');
load('test_data_synchro/frame_nb_mean.mat');


N = 15;%number of bin for tracking

% Number of cameras
ncameras = 3;
cameras = [1 2 3];

% Vectorization of our dataset
l = cellfun('length', frame_nb_mean);

all_mask = []; all_id = []; all_track = []; all_bbs = []; all_frame_nb = []; all_pic_nb = []; all_histR = []; all_histG = []; all_histB = [];
for i = 1:ncameras
    id_tmp = cell2mat(identity_mean{i});
    track_tmp = cell2mat(track_mean{i});
    bbs_tmp = cell2mat(bbs_mean{i});
    frame_nb_tmp = cell2mat(frame_nb_mean{i});
    pic_nb_tmp = cell2mat(pic_nb_mean{i});
    
    for o = 1:length(identity_mean{i})
        Mask_tmp = cell2mat(Mask_pic{i}{o});
        for j = 1:nb_k_means
            all_mask = [all_mask; Mask_tmp(:,(j-1)*W+1:j*W)];
%             figure(1),subplot(1,2,1), imshow(Mask_tmp(:,(j-1)*W+1:j*W));
%                         subplot(1,2,2), imshow(picture{pic_nb_mean{i}{o}(j)});
%             pause(1)
        end
    end
    
    all_id = [all_id; reshape(id_tmp,l(i)*nb_k_means,1)];
    all_track = [all_track; reshape(track_tmp,l(i)*nb_k_means,2)];
    all_bbs = [all_bbs; reshape(bbs_tmp,l(i)*nb_k_means,7)];
    all_frame_nb = [all_frame_nb; reshape(frame_nb_tmp,l(i)*nb_k_means,1)];
    all_pic_nb = [all_pic_nb; reshape(pic_nb_tmp,l(i)*nb_k_means,1)];
%     
    
    clear id_tmp track_tmp bbs_tmp frame_nb_tmp histR_tmp histG_tmp histB_tmp
end

%Sort by id number
[all_id, index] = sort(all_id);
all_track = all_track(index,:);
all_bbs = all_bbs(index,:);
all_frame_nb = all_frame_nb(index,:);
all_pic_nb = all_pic_nb(index,:);
% all_histR = all_histR(index,:);
% all_histG = all_histG(index,:);
% all_histB = all_histB(index,:);
index_mask = (index -ones(size(index))).*H +ones(size(index));
for i = 1:length(index_mask)
    all_mask((i-1)*H +1:i*H,:) = all_mask(index_mask(i):index_mask(i)+H-1,:);
end


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

         subplot(1,2,1), imshow(dataset(:,:,:,i));
         subplot(1,2,2), imshow(mask_fin(:,:,i));
         pause(1)
        
    waitbar(i/length(permit_inds),hwait)
end
fprintf('OK \n');
close(hwait)



%% Division in 3 part and kernel map computation
rep = 'test_data_synchro/';
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% -- Features extraction -- %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) MSCR
namefile = [rep 'MSCR_' num2str(SUBfac) '.mat'];
A = dir(namefile);
if ~isempty(A)
    load(namefile)
    fprintf('MSCR LOADING... OK\n')
else
    fprintf('MSCR COMPUTATION... ')
    tic
    ExtractMSCR;
    tt(2) = toc;
    save(namefile,'Blobs');     %% saving
    fprintf('OK \n');
end

% 2) Weighted HSV histogram
namefile = [rep 'wHSV_' num2str(SUBfac) '.mat'];
A = dir(namefile);
if ~isempty(A)
    load(namefile)
    fprintf('Weighted HSV hist LOADING... OK\n')
else
    fprintf('Weighted HSV hist COMPUTATION... ')
    tic
    EstimatewHSV;
    tt(3) = toc;
    save(namefile,'whisto2');     %% saving
    fprintf('OK \n');
end

%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                      Re-Identification                            %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%% ________________Re-Id avec matching SDALF_ To get the pairwise similarity _____________________ %%
clear num_cam

% Dataset inds extraction 
dset = SetDataset_adapte(permit_inds,dir_e_list,all_pic_nb); % contains indexes and all corresponding data of an id

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
    end
    cpt = cpt + n;
end
clear cpt N n;
% cpt = cpt+1
% for i = 1:length(ped(cpt).pic)
%     x = ped(cpt).pic;
%     id = ped(cpt).id
%     img = picture{x(i)};
%     imshow(img);
%     pause(0.5);
%     k = waitforbuttonpress ;
% end
%%
for t = 1:10
    % component of dynamic feature selection
    fprintf(['Test ' num2str(t) '...'])
    ped_back = ped;
    clear ped
    cpt = 0; n = 1; m = 1;
    for m = 1:length(ped_back) % for each id in each cam where he appears
       
         if length(ped_back(m).dynmodels) >= nb_k_means % We only keep id that appears at least 3 time (3 detections) in one cam

            in = randperm(length(ped_back(m).dynmodels)); % random selection of the data 4 dynamic feature
            ped(n).dynmodels = ped_back(m).dynmodels(in(1:min(length(ped_back(m).dynmodels), MAXCLUSTER))); % in : bool = 1 si arg(1) est dans arg(2)
            ped(n).rnddd = 1:length(ped(n).dynmodels);
            % Add frame and pic number
            ped(n).pic = ped_back(m).pic(in(1:min(length(ped_back(m).dynmodels), MAXCLUSTER)));
            ped(n).frame = ped_back(m).frame(in(1:min(length(ped_back(m).dynmodels), MAXCLUSTER)));
            % Add cam and id number info
            ped(n).cam = ped_back(m).cam;
            ped(n).id = ped_back(m).id;
            
%             for k =1:3
%                 subplot(1,3,1), imshow(picture{ped(n).pic(1)});
%                   subplot(1,3,2), imshow(picture{ped(n).pic(2)});
%                     subplot(1,3,3), imshow(picture{ped(n).pic(3)});
%                     
%             end
%             xxx = waitforbuttonpress ;
            
%      end      
            n = n+1;
         else
            %we don't take in account id appearing less than 3 times in
            %a camera FoV
         end
         
    end
    clear cpt N n;

    % Matching 
    
    % 1) Matching MSCR
   %%% MSCRmatch_ajout3combi; %MSCRmatch_DynVSDyn_Cl_adapte;
    
   
   %%%%%%%% TEST WITH wHSV ONLY %%%%%%%%
    % 2) Matching wHSV
    wHSVmatch_ajout3combi; %wHSVmatch_DynVSDyn_adapte;
    
    % -- Get pairwise similarity -- %
    crossvalidation_ajout3combi; %crossvalidation_Dyn_adapte;
    
    distance_pair{t} = pairwise_sim_synchro; 
    clear pairwise_sim_synchro;
    
    fprintf('Ok \n')
    ped = ped_back;
end

CPairs = combnk(cameras,2);

pairwise_sim = cell(ncameras,ncameras);
for i = 1:ncameras
    for t = 1:10
        dist_tmp = distance_pair{t};
        pairwise_tmp(:,:,t) = permute(dist_tmp{CPairs(i,1),CPairs(i,2)},[2 1]);
    end
    pairwise_sim{CPairs(i,1),CPairs(i,2)} = pairwise_tmp;
    clear pairwise_tmp;
end
%% If you want to save to test supervisor part
save('pairwise_sim_synchro.mat','pairwise_sim');

