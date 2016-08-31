%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             author : Guerrero Andréa, stagiaire LAAS/CNRS               %
%             subject : Build Ground Truth to verify re-id                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Ground truth

load('tracklets.mat');

ids = tree(2,:); % Corresponding to cam 6 (PRG6)
ids2 = tree(6,:); % Corresponding to cam 23 (PRG23)
ids3 = tree(3,:); % Corresponding to cam 7 (PRG6)

m=1;
for k = 1:length(ids)
    if(isstruct(ids{1,k}))
        for l=1:length(ids{1,k}.Frame)
            % Output = frame numbers corresponding to id(k)
            num_frames(k,l) = ids{1,k}.Frame(l).ATTRIBUTE.Number;
            bbs_truth(m,1) = ids{1,k}.Frame(l).ATTRIBUTE.Number;
            bbs_truth(m,2) = ids{1,k}.Frame(l).Left_X;
            bbs_truth(m,3) = ids{1,k}.Frame(l).Left_Y;
            bbs_truth(m,4) = ids{1,k}.Frame(l).Width;
            bbs_truth(m,5) = ids{1,k}.Frame(l).Height;
            bbs_truth(m,6) = ids{1,k}.ATTRIBUTE.id; % id number
            bbs_truth(m,7) = 1;
            m = m+1;
        end
        %id(1,k) = ids{1,k}.ATTRIBUTE.id; % id number
    end
end
for k = 1:length(ids2)
    if(isstruct(ids2{1,k}))
        for l=1:length(ids2{1,k}.Frame)
            % Output = frame numbers corresponding to id(k)
            num_frames(k,l) = ids2{1,k}.Frame(l).ATTRIBUTE.Number;
            bbs_truth(m,1) = ids2{1,k}.Frame(l).ATTRIBUTE.Number;
            bbs_truth(m,2) = ids2{1,k}.Frame(l).Left_X;
            bbs_truth(m,3) = ids2{1,k}.Frame(l).Left_Y;
            bbs_truth(m,4) = ids2{1,k}.Frame(l).Width;
            bbs_truth(m,5) = ids2{1,k}.Frame(l).Height;
            bbs_truth(m,6) = ids2{1,k}.ATTRIBUTE.id; % id number
            bbs_truth(m,7) = 2;
            m = m+1;
        end
        %id(1,k) = ids{1,k}.ATTRIBUTE.id; % id number
    end
end
for k = 1:length(ids3)
    if(isstruct(ids3{1,k}))
        for l=1:length(ids3{1,k}.Frame)
            % Output = frame numbers corresponding to id(k)
            num_frames(k,l) = ids3{1,k}.Frame(l).ATTRIBUTE.Number;
            bbs_truth(m,1) = ids3{1,k}.Frame(l).ATTRIBUTE.Number;
            bbs_truth(m,2) = ids3{1,k}.Frame(l).Left_X;
            bbs_truth(m,3) = ids3{1,k}.Frame(l).Left_Y;
            bbs_truth(m,4) = ids3{1,k}.Frame(l).Width;
            bbs_truth(m,5) = ids3{1,k}.Frame(l).Height;
            bbs_truth(m,6) = ids3{1,k}.ATTRIBUTE.id; % id number
            bbs_truth(m,7) = 3;
            m = m+1;
        end
    end
end


% n_max = 3; %nb max de frames separant 2ids fusionnes
%% Truth cam 1
[row1,col1]= find(((mini1+1) <= bbs_truth(:,1)) & (maxi1 >= bbs_truth(:,1)) & ( bbs_truth(:,7) == 1));

bbs_gt1_tmp = bbs_truth(row1,:);
res = mod(bbs_gt1_tmp(:,1)-(mini1),pas);
[l,c] = find(res == 1);
bbs_gt1 = bbs_gt1_tmp(l,:);

x = bbs_gt1(:,1)-mini1;
[x, idx] = sort(x);
bbs_gt1 = bbs_gt1(idx,:);

[ubbs, ind] = unique(bbs_gt1(:,6));
nb_per_frame = histc(bbs_gt1(:,6),ubbs);

bad = find(nb_per_frame <3);

for i=1:length(bad)
    elim = find(bbs_gt1(:,6) == ubbs(bad(i)));
    bbs_gt1(elim,:) = [];
end

ubbs = unique(bbs_gt1(:,6));

clear col1 row1 bbs_gt1_tmp res c nb_per_frame ubbs
%% Truth cam 2
[row2,col2]= find(((mini2+1) <= bbs_truth(:,1)) & (maxi2 >= bbs_truth(:,1)) & ( bbs_truth(:,7) == 2) );

bbs_gt2_tmp = bbs_truth(row2,:);
res = mod(bbs_gt2_tmp(:,1)-(mini2),pas);
[l,c] = find(res == 1);
bbs_gt2 = bbs_gt2_tmp(l,:);

x = bbs_gt2(:,1)-mini2;
[x, idx] = sort(x);
bbs_gt2 = bbs_gt2(idx,:);
%bbs_gt2(:,1) = x;

[ubbs, ind] = unique(bbs_gt2(:,6));
nb_per_frame = histc(bbs_gt2(:,6),ubbs);

bad = find(nb_per_frame <3);

for i=1:length(bad)
    elim = find(bbs_gt2(:,6) == ubbs(bad(i)));
    bbs_gt2(elim,:) = [];
end

c = unique(bbs_gt2(:,6));

clear col2 row2 bbs_gt2_tmp res c nb_per_frame ubbs
%% Truth cam 3
[row3,col3]= find(((mini3+1) <= bbs_truth(:,1)) & (maxi3 >= bbs_truth(:,1)) & ( bbs_truth(:,7) == 3) );

bbs_gt3_tmp = bbs_truth(row3,:);
res = mod(bbs_gt3_tmp(:,1)-(mini3),pas);
[l,c] = find(res == 1);
bbs_gt3 = bbs_gt3_tmp(l,:);

x = bbs_gt3(:,1)-mini3;
[x, idx] = sort(x);
bbs_gt3 = bbs_gt3(idx,:);

[ubbs, ind] = unique(bbs_gt3(:,6));
nb_per_frame = histc(bbs_gt3(:,6),ubbs);

bad = find(nb_per_frame <3);

for i=1:length(bad)
    elim = find(bbs_gt3(:,6) == ubbs(bad(i)));
    bbs_gt3(elim,:) = [];
end

c = unique(bbs_gt3(:,6));
clear col3 row3 bbs_gt3_tmp res c nb_per_frame ubbs

%% Synchronisation des ids inter-cam et intra

%Homme en t-shirt blanc et jean 108
%_cam6
l = find(bbs_gt1(:,6) == 113 | bbs_gt1(:,6) == 117 | bbs_gt1(:,6) == 140 | bbs_gt1(:,6) == 142 | bbs_gt1(:,6) == 188);
bbs_gt1(l,6) = 108;
%_cam23
l = find(bbs_gt2(:,6) == 68);
bbs_gt2(l,6) = 108;
%_cam7
l = find(bbs_gt3(:,6) == 164 | bbs_gt3(:,6) == 58 | bbs_gt3(:,6) == 60 | bbs_gt3(:,6) == 100 | bbs_gt3(:,6) == 102 | bbs_gt3(:,6) == 103  | bbs_gt3(:,6) == 176 | bbs_gt3(:,6) == 172 | bbs_gt3(:,6) == 56);
bbs_gt3(l,6) = 108;


%Homme pull bleu lunette 25
%_cam23
l = find(bbs_gt2(:,6) == 65 | bbs_gt2(:,6) == 82);
bbs_gt2(l,6) = 25;
%_cam7
l = find(bbs_gt3(:,6) == 107 | bbs_gt3(:,6) == 109 | bbs_gt3(:,6) == 118);
bbs_gt3(l,6) = 25;

%Homme chemisette orange carreaux 109
%_cam6
l = find(bbs_gt1(:,6) == 173 | bbs_gt1(:,6) == 179 | bbs_gt1(:,6) == 115);
bbs_gt1(l,6) = 109;
%_cam23
l = find(bbs_gt2(:,6) == 35 | bbs_gt2(:,6) == 50 | bbs_gt2(:,6) == 51 | bbs_gt2(:,6) == 87 | bbs_gt2(:,6) == 89 );
bbs_gt2(l,6) = 109;
%_cam7
l = find(bbs_gt3(:,6) == 167);
bbs_gt3(l,6) = 109;

%Femme t shirt rose jean 111
%_cam6
l = find(bbs_gt1(:,6) == 174 | bbs_gt1(:,6) == 175);
bbs_gt1(l,6) = 111;
%_cam23
l = find(bbs_gt2(:,6) == 34 | bbs_gt2(:,6) == 45 | bbs_gt2(:,6) == 86);
bbs_gt2(l,6) = 111;
%_cam7
l = find(bbs_gt3(:,6) == 72 | bbs_gt3(:,6) == 73 | bbs_gt3(:,6) == 175 | bbs_gt3(:,6) == 177);
bbs_gt3(l,6) = 111;

%Homme polo rayé violet 170 
%_cam6
%_cam23
l = find(bbs_gt2(:,6) == 43 | bbs_gt2(:,6) == 78 | bbs_gt2(:,6) == 16 | bbs_gt2(:,6) == 18);
bbs_gt2(l,6) = 170;
%_cam7
l = find(bbs_gt3(:,6) == 68 | bbs_gt3(:,6) == 69 | bbs_gt3(:,6) == 116 | bbs_gt3(:,6) == 117);
bbs_gt3(l,6) = 170;



%Homme tshirt rayé sacados bouteille 126
%_cam6
l = find(bbs_gt1(:,6) == 128 | bbs_gt1(:,6) == 134 );
bbs_gt1(l,6) = 126;

%Homme polo noir & jean 136
%_cam6
l = find(bbs_gt1(:,6) == 195 );
bbs_gt1(l,6) = 136;
%_cam23
l = find(bbs_gt2(:,6) == 20 | bbs_gt2(:,6) == 56 | bbs_gt2(:,6) == 72 | bbs_gt2(:,6) == 75);
bbs_gt2(l,6) = 136;
%_cam7
l = find(bbs_gt3(:,6) == 81 | bbs_gt3(:,6) == 82 | bbs_gt3(:,6) == 192 | bbs_gt3(:,6) == 193 | bbs_gt3(:,6) == 195 | bbs_gt3(:,6) == 197 | bbs_gt3(:,6) == 196);
bbs_gt3(l,6) = 136;

%Femme chemise rose pantalon noir 152 
%_cam6
%_cam23
l = find(bbs_gt2(:,6) == 84 | bbs_gt2(:,6) == 83 | bbs_gt2(:,6) == 26 | bbs_gt2(:,6) == 32 | bbs_gt2(:,6) == 63 | bbs_gt2(:,6) == 66);
bbs_gt2(l,6) = 152;
%_cam7
l = find(bbs_gt3(:,6) == 110 | bbs_gt3(:,6) == 123 );
bbs_gt3(l,6) = 152;

%Homme lunettes tshirt blanc sac a dos 123
%_cam6
l = find(bbs_gt1(:,6) == 129 | bbs_gt1(:,6) == 131 | bbs_gt1(:,6) == 132);
bbs_gt1(l,6) = 123;
%_cam7
l = find(bbs_gt3(:,6) == 83 | bbs_gt3(:,6) == 88 );
bbs_gt3(l,6) = 123;

%Homme tshirt rose short 155
%_cam6
l = find(bbs_gt1(:,6) == 157 | bbs_gt1(:,6) == 191 | bbs_gt1(:,6) == 164 | bbs_gt1(:,6) == 165 | bbs_gt1(:,6) == 167 | bbs_gt1(:,6) == 180 | bbs_gt1(:,6) == 181 | bbs_gt1(:,6) == 185 | bbs_gt1(:,6) == 187 | bbs_gt1(:,6) == 194);
bbs_gt1(l,6) = 155;
%_cam23
l = find(bbs_gt2(:,6) == 30 | bbs_gt2(:,6) == 31 | bbs_gt2(:,6) == 97 | bbs_gt2(:,6) == 99 );
bbs_gt2(l,6) = 155;
%_cam7
l = find(bbs_gt3(:,6) == 132 | bbs_gt3(:,6) == 133 | bbs_gt3(:,6) == 140 | bbs_gt3(:,6) == 53 | bbs_gt3(:,6) == 57 | bbs_gt3(:,6) == 135);
bbs_gt3(l,6) = 155;

%Homme t shirt bleu & jean 158
%_cam6
l = find(bbs_gt1(:,6) == 189);
bbs_gt1(l,6) = 158;
%_cam23
l = find(bbs_gt2(:,6) == 28 | bbs_gt2(:,6) == 93 | bbs_gt2(:,6) == 98);
bbs_gt2(l,6) = 158;
%_cam7
l = find(bbs_gt3(:,6) == 128 | bbs_gt3(:,6) == 131 | bbs_gt3(:,6) == 54);
bbs_gt3(l,6) = 158;

%Homme tshirt blanc short noir sac a dos 159
%_cam6
l = find(bbs_gt1(:,6) == 160 | bbs_gt1(:,6) == 156);
bbs_gt1(l,6) = 159;

%____cam23 ONLY
%H tshirt violet short noir sacados 33

%Femme chemise a carreaux couette 37
l = find(bbs_gt2(:,6) == 41);
bbs_gt2(l,6) = 37;

%Homme en marcel sacados rouge 67
l = find(bbs_gt2(:,6) == 69);
bbs_gt2(l,6) = 67;

%Homme t shirt gris jean lunette 70

%Femme sacados veste rouge 103

%____cam7 ONLY
%H short a carreaux sacados 95
l = find(bbs_gt3(:,6) == 97);
bbs_gt3(l,6) = 95;

%Femme pantalon haut et malette 146
l = find(bbs_gt3(:,6) == 149);
bbs_gt3(l,6) = 146;


%% Fusion des cams

bbs_gt = [bbs_gt1;bbs_gt2;bbs_gt3];% bbs_gt4];
%% ---------------- Representation graphique pour debug ---------------- %%

%% Pour visualiser les images d'un id particulier
% figure
% id = 146;  re = 'Image/S1/PRG7/frame/';
% 
% ol = find(bbs_gt3(:,6) == id);
% track = bbs_gt3(ol,:);
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