%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             author : Guerrero Andréa, stagiaire LAAS/CNRS               %
%             subject : Build Ground Truth to verify re-id                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Ground truth

load('tracklets.mat');

ids4 = tree(1,:); % Corresponding to cam 4 (PRG1)

m=1;
for k = 1:length(ids4)
    if(isstruct(ids4{1,k}))
        for l=1:length(ids4{1,k}.Frame)
            % Output = frame numbers corresponding to id(k)
            num_frames(k,l) = ids4{1,k}.Frame(l).ATTRIBUTE.Number;
            bbs_truth(m,1) = ids4{1,k}.Frame(l).ATTRIBUTE.Number;
            bbs_truth(m,2) = ids4{1,k}.Frame(l).Left_X;
            bbs_truth(m,3) = ids4{1,k}.Frame(l).Left_Y;
            bbs_truth(m,4) = ids4{1,k}.Frame(l).Width;
            bbs_truth(m,5) = ids4{1,k}.Frame(l).Height;
            bbs_truth(m,6) = ids4{1,k}.ATTRIBUTE.id; % id number
            bbs_truth(m,7) = 4;
            m = m+1;
        end
    end
end

%% Truth cam 4
[row4,col4]= find(((mini4+1) <= bbs_truth(:,1)) & (maxi4 >= bbs_truth(:,1)) & ( bbs_truth(:,7) == 4) );

bbs_gt4_tmp = bbs_truth(row4,:);
res = mod(bbs_gt4_tmp(:,1)-(mini4),pas);
[l,c] = find(res == 1);
bbs_gt4 = bbs_gt4_tmp(l,:);

x = bbs_gt4(:,1)-mini4;
[x, idx] = sort(x);
bbs_gt4 = bbs_gt4(idx,:);

[ubbs, ind] = unique(bbs_gt4(:,6));
nb_per_frame = histc(bbs_gt4(:,6),ubbs);

bad = find(nb_per_frame <3);

for i=1:length(bad)
    elim = find(bbs_gt4(:,6) == ubbs(bad(i)));
    bbs_gt4(elim,:) = [];
end

c = unique(bbs_gt4(:,6));
clear col4 row4 bbs_gt4_tmp res c nb_per_frame ubbs

%% Synchronisation des ids inter-cam et intra

%Homme en t-shirt blanc et jean 108
%_cam1
l = find(bbs_gt4(:,6) == 56 | bbs_gt4(:,6) == 74 | bbs_gt4(:,6) == 84 | bbs_gt4(:,6) == 92 | bbs_gt4(:,6) == 93 );
bbs_gt4(l,6) = 108;

%Homme pull bleu lunette 25
%_cam1
l = find(bbs_gt4(:,6) == 66 | bbs_gt4(:,6) == 67 );
bbs_gt4(l,6) = 25;


%Homme chemisette orange carreaux 109
%_cam1
l = find(bbs_gt4(:,6) == 28 | bbs_gt4(:,6) == 80 | bbs_gt4(:,6) == 81 );
bbs_gt4(l,6) = 109;


%Femme t shirt rose jean 111
%_cam1
l = find(bbs_gt4(:,6) == 24 | bbs_gt4(:,6) == 78 );
bbs_gt4(l,6) = 111;


%Homme polo rayé violet 170 
%_cam1
l = find(bbs_gt4(:,6) == 57 | bbs_gt4(:,6) == 76 );
bbs_gt4(l,6) = 170;



%Homme tshirt rayé sacados bouteille 126
%_cam1
l = find(bbs_gt4(:,6) == 37 | bbs_gt4(:,6) == 44 | bbs_gt4(:,6) == 45 | bbs_gt4(:,6) == 50 );
bbs_gt4(l,6) = 126;


%Homme polo noir & jean 136
%_cam1
l = find(bbs_gt4(:,6) == 59 | bbs_gt4(:,6) == 103);
bbs_gt4(l,6) = 136;


%Femme chemise rose pantalon noir 152 
%_cam1
l = find(bbs_gt4(:,6) == 65);
bbs_gt4(l,6) = 152;


%Homme lunettes tshirt blanc sac a dos 123
%_cam1
l = find(bbs_gt4(:,6) == 38 | bbs_gt4(:,6) == 41);
bbs_gt4(l,6) = 123;

%Homme tshirt rose short 155
%_cam1
l = find(bbs_gt4(:,6) == 70 | bbs_gt4(:,6) == 83);
bbs_gt4(l,6) = 155;

%Homme t shirt bleu & jean 158
%_cam1
l = find(bbs_gt4(:,6) == 68 | bbs_gt4(:,6) == 82);
bbs_gt4(l,6) = 158;

%____cam1 ONLY
%H sac a dos noir tshirt gris 29
l = find(bbs_gt4(:,6) == 49 | bbs_gt4(:,6) == 55);
bbs_gt4(l,6) = 29;

%H sac a dos short noir tshirt blanc 30
l = find(bbs_gt4(:,6) == 48 | bbs_gt4(:,6) == 54);
bbs_gt4(l,6) = 30;

%H sac a dos bleu tshirt bleu short carrx 60

%H pull vert sac a dos noir jean 86
l = find(bbs_gt4(:,6) == 88 | bbs_gt4(:,6) == 91 | bbs_gt4(:,6) == 94 | bbs_gt4(:,6) == 99);
bbs_gt4(l,6) = 86;


%% Fusion des cams

bbs_gt = bbs_gt4;
%% ---------------- Representation graphique pour debug ---------------- %%

%% Pour visualiser les images d'un id particulier
% figure
% id = 86;  re = 'Image/S1/PRG1/frame/';
% 
% ol = find(bbs_gt4(:,6) == id);
% track = bbs_gt4(ol,:);
% l = length(ol);
% numero = track(:,1); 
% for a=1:l
%     pic1 = imread([re 'frame' num2str(numero(a)) '.jpg']);
%     position_old = [track(a,2) track(a,3) track(a,4) track(a,5)];
%     
%     imshow(pic1)
%     rectangle('Position',position_old, 'LineWidth',2, 'EdgeColor','g');
%     drawnow;
% end