% load('test_data_synchro/imagette/imagette.mat');
% load('test_data_synchro/mask/mask_fin.mat');
% % To show results on our images
% load('test_data_synchro/liste_images.mat');

% % stock_xcenter : x coord. of bbs center, size ->  1*number of frame 
% % stock_ycenter : y coord. of bbs center, same size
% 
load('test_data_synchro_4cam/stock_xcenter.mat');
load('test_data_synchro_4cam/stock_ycenter.mat');
load('test_data_synchro_4cam/stock_bbs.mat');
% 
dir_e = 'test_data_synchro_4cam/imagette/';
dir_e_list = dir(strcat(dir_e,'/*.jpg'));


fprintf(' Tracking : Data LOADING...')
hwait = waitbar(0,'Tracking : Data LOADING...');
for i=1:length(dir_e_list)
    n = dir_e_list(i).name;

    %Extract info of every picture
    matches = regexp(n,'\d+','match');
    idp(i) = str2num(matches{1});
    npic(i) = str2num(matches{2});
    nimage(i) = str2num(matches{3});
    ncam(i) = str2num(matches{4});
    
%     figure(1)
%     subplot(1,2,1)
%     imshow(list_images{nimage(i)})
%     subplot(1,2,2)
%     imshow(picture{npic(i)})
    
    % Extracting histogram to compute bhattacharya distance
    img     =   picture{npic(i)-1924}; 
    %masking
    B = double(mask_fin(:,:,npic(i)-1924)~=0);
    img = double(img).*cat(3,B,B,B); % mask application
    %equalization
    [Ha,S,V] = rgb2hsv(img);
    Ve = (V - min(min(V))) / (max(max(V)) - min(min(V)));
    %Ve = histeq(V(B==1)); Veq = V; Veq(B == 1) = Ve;
    img = cat(3, Ha,S,Ve);
    img = hsv2rgb(img); 
    %histogram
    %im = rgb2gray(img);
    red{i}= imhist(img(:,:,1),N)/sum(imhist(img(:,:,1)));
    green{i}= imhist(img(:,:,2),N)/sum(imhist(img(:,:,2)));
    blue{i}= imhist(img(:,:,3),N)/sum(imhist(img(:,:,3)));
    
    waitbar(i/length(dir_e_list),hwait);
end
fprintf('OK \n');
close(hwait)


% save('test_data_synchro/nimage.mat', 'nimage');
% save('test_data_synchro/npic.mat', 'npic');
% save('test_data_synchro/idp.mat', 'idp');
% save('test_data_synchro/ncam.mat', 'ncam');



%% Save center of bbs per pair of frame
%  
seuil = 150;
for cam = 1:ncameras
    [l,c] = find(ncam == cameras(cam));
    
    % Take info of current cam
    pic_cam = npic(c);
    [pic_cam, rang] = sort(pic_cam);
    c = c(rang);
    idp_cam = idp(c);
    im_cam = nimage(c);
    
%     figure(1)
%     for i = 1 :length(c)
%         subplot(1,2,1)
%         imshow(imresize(list_images{im_cam(i)},2,'bilinear'));hold on;
%         plot(stock_ycenter{pic_cam(i)},stock_xcenter{pic_cam(i)},  'xr'); hold off;
%         subplot(1,2,2)
%         imshow(picture{pic_cam(i)})
%         pause(0.25);
%     end

    for i=1:length(c)
        center_x{i} = stock_xcenter{pic_cam(i)-1924};
        center_y{i} = stock_ycenter{pic_cam(i)-1924};
        
        bbs_stock{i} = stock_bbs{pic_cam(i)-1924};
        
        histR_cam{i} = red{c(i)};
        histG_cam{i} = green{c(i)};
        histB_cam{i} = blue{c(i)};
    end
    
    % Stock in relation to the frame number
    j = 1; cpt = 0;
    while (j ~= length(c)-1)
        cpt = cpt +1;
        %Number of person for the current frame
        nb_pic_curr = histc(im_cam,im_cam(j));
        nb_pic_next = histc(im_cam,im_cam(j+nb_pic_curr));

        %Person ID present in the current frame
        [id_curr{cam,cpt}, ind_curr] = sort(idp_cam(j:j+nb_pic_curr-1));
        [id_next{cam,cpt}, ind_next] = sort(idp_cam(j+nb_pic_curr:j+nb_pic_curr+nb_pic_next-1));
        
%         mat_dist = zeros(nb_pic_curr,nb_pic_next);
        cptl = 0; 
        for l = ind_curr
            cptl = cptl+1;
            cptm = 0;
            for m = ind_next
                cptm = cptm+1;
                % Stock person position : [x1 y1 x2 y2 .. xn yn])
                curr(cptl,:) = [center_x{j+l-1} center_y{j+l-1}];
                next(cptm,:) = [center_x{j+nb_pic_curr+m-1} center_y{j+nb_pic_curr+m-1}];
                
                % Stock bbs data
                bbscurr(cptl,:) = bbs_stock{j+l-1} ;
                bbsnext(cptm,:) = bbs_stock{j+nb_pic_curr+m-1};
                
                % Stock hist RGB
                hcurrR(cptl,:) = histR_cam{j+l-1};
                hnextR(cptm,:) = histR_cam{j+nb_pic_curr+m-1};
                
                hcurrG(cptl,:) = histG_cam{j+l-1};
                hnextG(cptm,:) = histG_cam{j+nb_pic_curr+m-1};
                
                hcurrB(cptl,:) = histB_cam{j+l-1};
                hnextB(cptm,:) = histB_cam{j+nb_pic_curr+m-1};
                
                % Stock current info : picture, frame and id number
                im_curr(cptl,:) = im_cam(j+l-1);
                pic_curr(cptl,:) = pic_cam(j+l-1);
                
                im_next(cptm,:) = im_cam(j+nb_pic_curr+m-1);
                pic_next(cptm,:) = pic_cam(j+nb_pic_curr+m-1);
                
             end
        end
        
        % Stock picture and frame number
        number_frames_curr{cam,cpt} = im_curr;
        number_detect_curr{cam,cpt} = pic_curr;
        
        number_frames_next{cam,cpt} = im_next;
        number_detect_next{cam,cpt} = pic_next;
        
        % Stock bbs data and histogram RGB
        pos_curr{cam,cpt} = curr;
        pos_next{cam,cpt} = next;
        
        bbs_curr{cam,cpt} = bbscurr;
        bbs_next{cam,cpt} = bbsnext;
        
        histR_curr{cam,cpt} = hcurrR;
        histG_curr{cam,cpt} = hcurrG;
        histB_curr{cam,cpt} = hcurrB;
        
        histR_next{cam,cpt} = hnextR;
        histG_next{cam,cpt} = hnextG;
        histB_next{cam,cpt} = hnextB;
        
        clear  bbscurr bbsnext mat_dist curr next hcurrR hcurrG hcurrB hnextR hnextG hnextB im_curr pic_curr idp_curr im_next pic_next idp_next;
        j = j + l;
    end
    
    clear bbs_stock l c pic_cam idp_cam im_cam histR_cam histG_cam histB_cam;
    
end

%% Build tracklets
% clear tracklets identity bbs_trac_cam cpt_dead histR_bha histG_bha histB_bha histR_trac histG_trac histB_trac cpt_diff_frame FrameNb_trac
% % To know what threshold choose
% r = 0.02 + (0.1-0.02).*rand(3000,1)
% for j = 1:3000
% 
% %-- threshold_efficiency : 3756 full-try
% try_number = 3756;
% threshold = r(1:try_number);
% 
% for curr_try = 1:try_number
%     var_tot(curr_try) = var(data_variance{curr_try});
% end
% plot(1:try_number,var_tot);
% title('variance globale de chaque essai');
% 
% [l,c] = min(var_tot);
% threshold(c) % optimum threshold : 0.0225

   %%
coef_dis = 0.4; %Give more or less importance to bhattacharyya and euclidian distances
seuil_dis = 100;
%seuil_criter = 0.05;
seuil_alive = 10; %Nb of tracklets used : the last 'seuil_alive' tracklets
seuil_dbha_cam = [0.065; 0.065; 0.065; 0.05];%r(j);
coef_dbha = ones(1,3).*0.2;
for cam=1:ncameras
    cpt = 1; new = 1; % New tracklet is created
    
    seuil_dbha = seuil_dbha_cam(cameras(cam));
    
    % Initialization
    tracklets{new} = [];
    identity{new} = [];
    histR_trac{new} = [];
    histG_trac{new} = [];
    histB_trac{new} = [];
    FrameNb_trac{new} = [];
    bbs_trac{new} = [];
    pic_trac{new} = [];
    
    % cpt frame number of tracklet t 'dead'
    cpt_diff_frame{new} = 0;
    cpt_dead{new} = 0;
    
    % While there is a position
    while( cpt < size(pos_curr,2) && ~isempty(pos_curr{cam,cpt})) 
         
        % Take current position
        p_frame_curr = pos_curr{cam,cpt};
        idcurr = id_curr{cam,cpt};
        histoRcurr = histR_curr{cam,cpt};
        histoGcurr = histG_curr{cam,cpt};
        histoBcurr = histB_curr{cam,cpt};
        %Frame number
        im_curr = number_frames_curr{cam,cpt};
        %Picture number
        pic_curr = number_detect_curr{cam,cpt};
        %Bounding Box
        bbscurr = bbs_curr{cam,cpt};
        
        % Take next position
        p_frame_next = pos_next{cam,cpt};
        idnext = id_next{cam,cpt};
        histoRnext = histR_next{cam,cpt};
        histoGnext = histG_next{cam,cpt};
        histoBnext = histB_next{cam,cpt};
        %Frame number
        im_next = number_frames_next{cam,cpt};
        %Picture number
        pic_next = number_detect_next{cam,cpt};
        %Bounding Box
        bbsnext = bbs_next{cam,cpt};
        
        % All combinaisons we can have between im_curr and im_next
        a=1:length(idcurr);
        b=1:length(idnext);

        aa=repmat(a',numel(b),1);
        bb=repmat(b,numel(a),1);
        possible = [aa(:) bb(:)]; %Combinaisons
        cpt_pers2 = 0; 
        
        flag = 0;
        while ~isempty(possible)
           
            pers1 = possible(1,1);
            ind_commun = find(possible(:,1) == pers1);
            
            flag = 0;
          %  while ~isempty(p_frame_next)
%                 pente_curr = (p_frame_next(pers2,2)-p_frame_curr(pers1,2))/(p_frame_next(pers2,1)-p_frame_curr(pers1,1)+eps);
            while( cpt_pers2 < length(ind_commun) && flag == 0 )

                cpt_pers2 = cpt_pers2+1;
                pers2 = possible(ind_commun(cpt_pers2),2);
                
%                 if (idcurr(pers1) == idnext(pers2))
%                     disp('GOOD');
%                 else
%                     disp('BAD');
%                 end
                
                ok = 0;

                number_empty = size(cellfun('isempty',tracklets),2);
                criterion = Inf.*ones(1,number_empty);
                for t = 1:number_empty
                    if cpt_diff_frame{t} < seuil_alive && ~isempty(cpt_diff_frame{t})
                        % -- Between curr and next
                        % distance  euclidienne
                        V = p_frame_curr(pers1,:) - p_frame_next(pers2,:);
                        D = sqrt(V * V');
                        % distance bhattacharya
                        d_bhaR = compareHists(histoRcurr(pers1,:),histoRnext(pers2,:));
                        d_bhaG = compareHists(histoGcurr(pers1,:),histoGnext(pers2,:));
                        d_bhaB = compareHists(histoBcurr(pers1,:),histoBnext(pers2,:));
                        d_bha = [d_bhaR, d_bhaG, d_bhaB];

                        if (cellfun('isempty',histR_trac(t)))

                            if ( D < seuil_dis && floor(sum(d_bha < seuil_dbha)/3))
                                % Save for current tracklet
                                histR_trac{new} = [histoRcurr(pers1,:); histoRnext(pers2,:)];
                                histG_trac{new} = [histoGcurr(pers1,:); histoGnext(pers2,:)];
                                histB_trac{new} = [histoBcurr(pers1,:); histoBnext(pers2,:)];
                                tracklets{new} = [p_frame_curr(pers1,:); p_frame_next(pers2,:)];
                                identity{new} = [idcurr(pers1); idnext(pers2)];
                                %Frame number
                                FrameNb_trac{new} = [im_curr(pers1); im_next(pers2)];
                                %Picture number
                                pic_trac{new} = [pic_curr(pers1); pic_next(pers2)];
                                %Bounding box
                                bbs_trac{new} = [bbscurr(pers1,:); bbsnext(pers2,:)];

                                ind1 = find(possible(:,1) == pers1); 
                                ind2 = find(possible(:,2) == pers2 & possible(:,1) ~= pers1);
                                ind = [ind1;ind2];
                                possible(ind,:) = [];
                                % Re-nitialization after deleting people 
                                cpt_pers2 = 0;

                                cpt=cpt+2;
                            end
                            cpt_diff_frame{t} = 0;
                            flag = 1;
                        else 
                            track = tracklets{t};
                            hR_tr = histR_trac{t};
                            hG_tr = histG_trac{t};
                            hB_tr = histB_trac{t};
                            Frame_tr = FrameNb_trac{t};
                            Picture_tr = pic_trac{t};
                            % We take the last info of the track
                            tr = track(end,:);
                            histoR_tr = hR_tr(end,:);
                            histoG_tr = hG_tr(end,:);
                            histoB_tr = hB_tr(end,:);
                            NbFrame_tr = Frame_tr(end,:);
                            Pic_tr = Picture_tr(end,:);

                            cpt_diff_frame{t} = im_curr(pers1)-NbFrame_tr;

                            % -- Between curr and track
                            % distance euclidienne
                            V = p_frame_curr(pers1,:) - tr;
                            D_tr = sqrt(V * V');
                            % distance bhattacharya
                            d_bhaR_tr = compareHists(histoRcurr(pers1,:),histoR_tr);
                            d_bhaG_tr = compareHists(histoGcurr(pers1,:),histoG_tr);
                            d_bhaB_tr = compareHists(histoBcurr(pers1,:),histoB_tr);
                            d_bha_tr = [d_bhaR_tr, d_bhaG_tr, d_bhaB_tr];

                            criter = coef_dbha* d_bha_tr'  + coef_dis*D_tr/800;

                            if ( D < seuil_dis && floor(sum(d_bha < seuil_dbha)/3))

                                if (D_tr < seuil_dis && floor(sum(d_bha_tr < seuil_dbha)/3))
                                    criterion(t) = criter;
                                    % We search an existing tracklets that'll be
                                    % coherent with current pair
                                    ok = 1;
                                end
                                if ( t == number_empty && ok)
                                    [~, keep] = min(criterion);
                                    % We keep the best of them
                                    tracklets{keep} = [tracklets{keep}; p_frame_curr(pers1,:); p_frame_next(pers2,:)];
                                    identity{keep} = [identity{keep}; idcurr(pers1); idnext(pers2)];
                                    FrameNb_trac{keep} = [FrameNb_trac{keep}; im_curr(pers1); im_next(pers2)];
                                    pic_trac{keep} = [pic_trac{keep}; pic_curr(pers1); pic_next(pers2) ];
                                    % Save hist for current association
                                    histR_trac{keep} = [histR_trac{keep}; histoRcurr(pers1,:); histoRnext(pers2,:)];
                                    histG_trac{keep} = [histG_trac{keep}; histoGcurr(pers1,:); histoGnext(pers2,:)];
                                    histB_trac{keep} = [histB_trac{keep}; histoBcurr(pers1,:); histoBnext(pers2,:)];
                                    bbs_trac{keep} = [bbs_trac{keep}; bbscurr(pers1,:); bbsnext(pers2,:)];

                                    ind1 = find(possible(:,1) == pers1); 
                                    ind2 = find(possible(:,2) == pers2 & possible(:,1) ~= pers1);
                                    ind = [ind1;ind2];
                                    possible(ind,:) = [];
                                    % Re-nitialization after deleting people 
                                    cpt_pers2 = 0;

                                    other = 1:number_empty;
                                    dead = (other - keep.*ones(size(other))) ~=0;
                                    if (size(dead,2) ==1 && dead == 0); 
                                    else
                                        for i=1:size(find(dead == 1),2)
                                            cpt_dead{i} = cpt_dead{i}+1; 
                                        end
                                    end

                                    cpt=cpt+2;
                                    flag = 1;
                                 end
                            else
                                if (D_tr < seuil_dis && floor(sum(d_bha_tr < seuil_dbha)/3))

                                    criterion(t) = criter;
                                    % We search an existing tracklets that'll be
                                    % coherent with current pair
                                    ok = 1;
                                end
                                if ( t == number_empty && ok)
                                    [~, keep] = min(criterion);
                                    % We keep the best of them
                                    tracklets{keep} = [tracklets{keep}; p_frame_curr(pers1,:)];
                                    identity{keep} = [identity{keep}; idcurr(pers1)];
                                    FrameNb_trac{keep} = [FrameNb_trac{keep}; im_curr(pers1)];
                                    pic_trac{keep} = [pic_trac{keep}; pic_curr(pers1)];


                                    % Save hist for current association
                                    histR_trac{keep} = [histR_trac{keep}; histoRcurr(pers1,:)];
                                    histG_trac{keep} = [histG_trac{keep}; histoGcurr(pers1,:)];
                                    histB_trac{keep} = [histB_trac{keep}; histoBcurr(pers1,:)];
                                    bbs_trac{keep} = [bbs_trac{keep}; bbscurr(pers1,:)];

                                    ind = find(possible(:,1) == pers1);
                                    possible(ind,:) = [];
                                    % Re-nitialization after deleting people 
                                    cpt_pers2 = 0;

                                    cpt=cpt+1;

                                    other = 1:number_empty;
                                    dead = (other - keep.*ones(size(other))) ~=0;
                                    if (size(dead,2) ==1 && dead == 0); 
                                    else
                                        for i=1:size(find(dead == 1),2)
                                            cpt_dead{i} = cpt_dead{i}+1; 
                                        end
                                    end
                                    
                                    flag = 1;
                                end
                            end
                        end
                    end
                end
                clear criterion
            end
            % If there isn't any match with existing tracklets
            if flag == 0 
                new = new + 1;
                tracklets{new} = p_frame_curr(pers1,:);
                identity{new} = idcurr(pers1);
                FrameNb_trac{new} = im_curr(pers1);
                pic_trac{new} = pic_curr(pers1);
                histR_trac{new} = histoRcurr(pers1,:);
                histG_trac{new} = histoGcurr(pers1,:);
                histB_trac{new} = histoBcurr(pers1,:);
                bbs_trac{new} =  bbscurr(pers1,:);

                ind = find(possible(:,1) == pers1);
                possible(ind,:) = [];

                cpt_diff_frame{new} = 0;

                cpt_dead{new} = 0;
                cpt=cpt+1;
                
                cpt_pers2 = 0; 
            end
            
        end
    end
    
    tracklets_cam{cam} = tracklets;
    identity_cam{cam} = identity;
    histR_trac_cam{cam} = histR_trac;
    histG_trac_cam{cam} = histG_trac;
    histB_trac_cam{cam} = histB_trac;
    FrameNb_trac_cam{cam} = FrameNb_trac;
    bbs_trac_cam{cam} = bbs_trac;
    pic_trac_cam{cam} = pic_trac;
    
    clear tracklets identity histR_trac histG_trac histB_trac bbs_trac FrameNb_trac a aa b B bb D pic_trac
    
end

   %%%%
% for kk = 1:size(cellfun('isempty',identity),2)
%     varia(kk) = var(identity{2,kk});
% end
% number{j} = ~cellfun('isempty',tracklets(cam, :));
% data_variance{j} = varia;
% data_identity{j} = identity;
% clear tracklets identity cpt_dead histR_bha histG_bha histB_bha
% end

% save('test_data_synchro/data_identity.mat', 'data_identity');
% save('test_data_synchro/data_variance.mat', 'data_variance');
% save('test_data_synchro/threshold.mat', 'r');

%% To merge tracklets we build 'contour tracing'
clear identity_merge track_merge Frame_nb_merge Pic_merge

%We take one cam at a time
 for cam = 1:ncameras; 

    %Max Number that separate present and other
    seuil_nb_frame = 30; 

    seuil_bbs = 0.15;% We accept an error of 'seuil_bbs*100'%
    seuil_dbha = 0.2;

    
    bbs_cam = bbs_trac_cam{1,cam};
    tracklets = tracklets_cam{cam};
    identity = identity_cam{cam};

    % Frame number corresponding to cam
    Frame_number = FrameNb_trac_cam{cam};
    % Picture number
    pic_nb = pic_trac_cam{cam};
    
    % Color info
    histR_trac = histR_trac_cam{cam};
    histG_trac = histG_trac_cam{cam};
    histB_trac = histB_trac_cam{cam};
    % for j =1:2
    present = 1; cpt_track = 1;
    while present <= length(tracklets)
        flag = 0;

        % Present tracklet
        track_pres = tracklets{present};
        id_pres = identity{present};
%         %%%
%         disp('pers present');
%         id_pres(1)
        %%%
        Frame_pres = Frame_number{present};
        Pic_pres = pic_nb{present};
        bbs_pres = bbs_cam{present};
        Nb_tr = Frame_pres(end,:);
        R_pres = histR_trac{present};
        G_pres = histG_trac{present};
        B_pres = histB_trac{present};

        if present+1 < length(tracklets)
            possible = present+1:length(tracklets);
        else
            break;
        end
        cpt = 1;
        for other = possible;
            % Look for coherent frame development
            Frame_other = Frame_number{other};
            Pic_other = pic_nb{other};
            Nb_tr_other = Frame_other(1,:);
            interest(cpt) = (Nb_tr_other - Nb_tr);
            cpt = cpt +1;
        end
        % Time coherent tracklets we want to compare
        if(~isempty(possible))
            ind_other = (interest < seuil_nb_frame & interest > 0);
            ind_other = possible(ind_other);
        end

         clear interest;

        % We want to merge tracklets with a coherent trajectory and time data
        for other = ind_other 

            % Other tracklet we want to compare with present
            track_other = tracklets{other};
            id_other = identity{other};
            bbs_other = bbs_cam{other};
            Pic_other = pic_nb{other};

            n = size(track_pres,1);%nb of data in track_pres
            m = size(track_other,1);

            R_other = histR_trac{other};
            G_other = histG_trac{other};
            B_other = histB_trac{other};

%             %%%
%             disp('pers other');
%             id_other(1)
%             %%%
            Frame_other = Frame_number{other};

            % Case where we have only 1 track present and more other
            if n == 1 && m > 1

                % We take track info
                tr = track_pres(end,:);
                NbFrame_tr = Frame_pres(end,:);
                bbs_tr = bbs_pres(end,:);

                % We build a direction vector with the two or three last track 
                mini = 2;
                maxi = min(3,size(track_other,1));
                cpt_d = 0;
                for i = mini:maxi
                    cpt_d = cpt_d+1;
                    % We take the two last info of the track
                    tr = track_other(i-1:i,:);
                    NbFrame_other = Frame_other(i-1:i,:);
                    diff_frame = NbFrame_other(2) - NbFrame_other(1);

                    % Build the direction vector
                    other_1 = tr(1,:);
                    other_2 = tr(2,:);
                    direction(cpt_d,:) = (other_2 - other_1)/diff_frame;
                end
                direct = mean(direction);
                clear direction;

                direct_apply = other_1 - direct.*(Frame_other(1,:) - NbFrame_tr);

                d_bhaR = compareHists(R_pres,R_other(1,:));
                d_bhaG = compareHists(G_pres,G_other(1,:));
                d_bhaB = compareHists(B_pres,B_other(1,:));
                d_bha = [d_bhaR, d_bhaG, d_bhaB];

                if ((bbs_tr(2)+bbs_tr(4))*(1+seuil_bbs(cam)) > direct_apply(1) && direct_apply(1) > bbs_tr(2)*(1-seuil_bbs(cam))  ...
                   && (bbs_tr(1)+bbs_tr(3))*(1+seuil_bbs(cam)) > direct_apply(2) && direct_apply(2) > bbs_tr(1)*(1-seuil_bbs(cam)) ...
                   && sum(d_bha) < seuil_dbha(cam))

                    % We update (and sort by frame number) the info of the present tracklet
                    [Frame_pres,ind] = sort([Frame_pres; Frame_other]);
                    Pic_pres = [Pic_pres; Pic_other];
                    Pic_pres = Pic_pres(ind);
                    track_pres = [track_pres; track_other];
                    track_pres = track_pres(ind,:);
                    id_pres = [id_pres; id_other];
                    id_pres = id_pres(ind);
                    bbs_pres = [bbs_pres; bbs_other];
                    bbs_pres = bbs_pres(ind,:);

                    R_pres = [R_pres; R_other];
                    R_pres = R_pres(ind,:);
                    G_pres = [G_pres; G_other];
                    G_pres = G_pres(ind,:);
                    B_pres = [B_pres; B_other];
                    B_pres = B_pres(ind,:);

                    % We merge the two tracklets infos
                    track_merge{cpt_track} = track_pres;
                    identity_merge{cpt_track} = id_pres;
                    Frame_nb_merge{cpt_track} = Frame_pres;
                    Pic_merge{cpt_track} = Pic_pres;
                    bbs_merge{cpt_track} = bbs_pres;

                    histR_merge{cpt_track} = R_pres;
                    histG_merge{cpt_track} = G_pres;
                    histB_merge{cpt_track} = B_pres;

                    % We delete data already merged
                    histR_trac{other} = 0;
                    histG_trac{other} = 0;
                    histB_trac{other} = 0;

                    tracklets{other} = 0;
                    identity{other} = 0;
                    bbs_cam{other} = 0;
                    Frame_number{other} = 0;
                    pic_nb{other} = 0;

                    flag = 1;

                end

            % Case where we have only 1 track other and more present
            elseif n > 1
                % We take other track info
                tr_other = track_other(1,:);
                NbFrame_tr_other = Frame_other(1,:);
                bbs_tr_other = bbs_other(1,:);

                % We build a direction vector with the two or three last track 
                mini = max(2,size(track_pres,1)-3);
                maxi = size(track_pres,1);
                cpt_d = 0;
                for i = mini:maxi
                    cpt_d = cpt_d+1;
                    % We take the two last info of the track
                    tr = track_pres(i-1:i,:);
                    NbFrame_tr = Frame_pres(i-1:i,:);
                    diff_frame = NbFrame_tr(2) - NbFrame_tr(1);

                    % Build the direction vector
                    pres_1 = tr(1,:);
                    pres_2 = tr(2,:);
                    direction(cpt_d,:) = (pres_2 - pres_1)/diff_frame;
                end
                direct = mean(direction,1);
                clear direction;

                direct_apply = track_pres(end,:) + direct.*(NbFrame_tr_other - Frame_pres(end));

                d_bhaR = compareHists(R_pres(end,:),R_other(1,:));
                d_bhaG = compareHists(G_pres(end,:),G_other(1,:));
                d_bhaB = compareHists(B_pres(end,:),B_other(1,:));
                d_bha = [d_bhaR, d_bhaG, d_bhaB];

                if ((bbs_tr_other(2)+bbs_tr_other(4))*(1+seuil_bbs(cam)) > direct_apply(1) && direct_apply(1) > bbs_tr_other(2)*(1-seuil_bbs(cam))  ...
                   && (bbs_tr_other(1)+bbs_tr_other(3))*(1+seuil_bbs(cam)) > direct_apply(2) && direct_apply(2) > bbs_tr_other(1)*(1-seuil_bbs(cam)) ...
                   & sum(d_bha) < seuil_dbha(cam) )

                    % We update (and sort by frame number) the info of the present tracklet
                    [Frame_pres,ind] = sort([Frame_pres; Frame_other]);
                    Pic_pres = [Pic_pres; Pic_other];
                    Pic_pres = Pic_pres(ind);
                    track_pres = [track_pres; track_other];
                    track_pres = track_pres(ind,:);
                    id_pres = [id_pres; id_other];
                    id_pres = id_pres(ind);
                    bbs_pres = [bbs_pres; bbs_other];
                    bbs_pres = bbs_pres(ind,:);

                    R_pres = [R_pres; R_other];
                    R_pres = R_pres(ind,:);
                    G_pres = [G_pres; G_other];
                    G_pres = G_pres(ind,:);
                    B_pres = [B_pres; B_other];
                    B_pres = B_pres(ind,:);

                    % We merge the two tracklets infos
                    track_merge{cpt_track} = track_pres;
                    identity_merge{cpt_track} = id_pres;
                    Frame_nb_merge{cpt_track} = Frame_pres;
                    bbs_merge{cpt_track} = bbs_pres;
                    Pic_merge{cpt_track} = Pic_pres;

                    histR_merge{cpt_track} = R_pres;
                    histG_merge{cpt_track} = G_pres;
                    histB_merge{cpt_track} = B_pres;

                    clear histR histG histB

                    % We delete data already merged
                    histR_trac{other} = 0;
                    histG_trac{other} = 0;
                    histB_trac{other} = 0;

                    tracklets{other} = 0;
                    identity{other} = 0;
                    bbs_cam{other} = 0;
                    Frame_number{other} = 0;
                    pic_nb{other} = 0;

                    flag = 1;
                end

            % Case where we have only 1 track present and other
            elseif n == 1 && m == 1
                %distance bhattacharya
                d_bhaR = compareHists(R_pres,R_other);
                d_bhaG = compareHists(G_pres,G_other);
                d_bhaB = compareHists(B_pres,B_other);
                d_bha = [d_bhaR, d_bhaG, d_bhaB];

                if sum(d_bha) < 0.3
                    % We update (and sort by frame number) the info of the present tracklet
                    [Frame_pres,ind] = sort([Frame_pres; Frame_other]);
                    track_pres = [track_pres; track_other];
                    track_pres = track_pres(ind,:);
                    id_pres = [id_pres; id_other];
                    id_pres = id_pres(ind);
                    bbs_pres = [bbs_pres; bbs_other];
                    bbs_pres = bbs_pres(ind,:);

                    R_pres = [R_pres; R_other];
                    R_pres = R_pres(ind,:);
                    G_pres = [G_pres; G_other];
                    G_pres = G_pres(ind,:);
                    B_pres = [B_pres; B_other];
                    B_pres = B_pres(ind,:);

                    % We merge the two tracklets infos
                    track_merge{cpt_track} = track_pres;
                    identity_merge{cpt_track} = id_pres;
                    Frame_nb_merge{cpt_track} = Frame_pres;
                    Pic_merge{cpt_track} = Pic_pres;
                    bbs_merge{cpt_track} = bbs_pres;

                    histR_merge{cpt_track} = R_pres;
                    histG_merge{cpt_track} = G_pres;
                    histB_merge{cpt_track} = B_pres;

                    % We delete data already merged
                    histR_trac{other} = 0;
                    histG_trac{other} = 0;
                    histB_trac{other} = 0;

                    tracklets{other} = 0;
                    identity{other} = 0;
                    bbs_cam{other} = 0;
                    Frame_number{other} = 0;
                    pic_nb{other} = 0;

                    flag = 1;
                end
            end

        end

        % We delete index of merged track
        idxZeros = cellfun(@(c)(isequal(c,0)), histR_trac);
        histR_trac(idxZeros) = [];
        idxZeros = cellfun(@(c)(isequal(c,0)), histG_trac);
        histG_trac(idxZeros) = [];
        idxZeros = cellfun(@(c)(isequal(c,0)), histB_trac);
        histB_trac(idxZeros) = [];

        idxZeros = cellfun(@(c)(isequal(c,0)), tracklets);
        tracklets(idxZeros) = [];
        idxZeros = cellfun(@(c)(isequal(c,0)), identity);
        identity(idxZeros) = [];
        idxZeros = cellfun(@(c)(isequal(c,0)), bbs_cam);
        bbs_cam(idxZeros) = [];
        idxZeros = cellfun(@(c)(isequal(c,0)), Frame_number);
        Frame_number(idxZeros) = [];
        idxZeros = cellfun(@(c)(isequal(c,0)), pic_nb);
        pic_nb(idxZeros) = [];

        % If there is not merging to do
        if(flag == 0)
            % We only keep the present tracklets

            track_merge{cpt_track} = track_pres;
            identity_merge{cpt_track} = id_pres; 
            Frame_nb_merge{cpt_track} = Frame_pres;
            Pic_merge{cpt_track} = Pic_pres;
            bbs_merge{cpt_track} = bbs_pres;

            histR_merge{cpt_track} = histR_trac{present}; 
            histG_merge{cpt_track} = histG_trac{present};
            histB_merge{cpt_track} = histB_trac{present};
        end

        clear R_pres G_pres B_pres track_pres id_pres Frame_pres Pic_pres

        cpt_track = cpt_track+1;
        present = present +1;
    end
    
    clear tracklets identity Frame_number bbs_cam histR_trac histG_trac histB_trac pic_nb
    
    cellsz = cellfun(@length,identity_merge);
    indx = find(cellsz == 1);
    Frame_nb_merge(indx) = [];
    Pic_merge(indx) = [];
    track_merge(indx) = [];
    identity_merge(indx) = [];
    bbs_merge(indx) = [];
    
    histR_merge(indx) = [];
    histG_merge(indx) = [];
    histB_merge(indx) = [];
    
    
    % We record all of our variable per cam
    Frame_number_final{cam} = Frame_nb_merge;
    Pic_final{cam} = Pic_merge;
    identity_final{cam} = identity_merge;
    tracklets_final{cam} = track_merge;
    histR_final{cam} = histR_merge;
    histG_final{cam} = histG_merge;
    histB_final{cam} = histB_merge;
    bbs_final{cam} = bbs_merge;
    
    clear track_merge identity_merge Frame_nb_merge bbs_merge histR_merge histG_merge histB_merge Pic_merge
end


% save('test_data_synchro_4cam/tracklets.mat', 'tracklets_final');
% save('test_data_synchro_4cam/tracklets_picnb.mat', 'Pic_final');
% save('test_data_synchro_4cam/tracklets_framenb.mat', 'Frame_number_final');
% save('test_data_synchro_4cam/tracklets_ids.mat', 'identity_final');
% save('test_data_synchro_4cam/tracklets_bbs.mat', 'bbs_final');



%%
%-- Graphic representation  -- %

% plot_type = ['r*-'; 'g+-'; 'b*-'; 'c+-'; 'm*-'];
% 
%    cam = 4
%    identity_merge =  identity_final{cam}  ;
%    track_merge = tracklets_final{cam}  ;
%    Frame_nb_merge = Frame_number_final{cam}  ;
% 
% for i = 1:length(identity_merge)
%     
%     images = Frame_nb_merge{i};
%     position = track_merge{i};
%     id = identity_merge{i};
%     
%     for j = 1:length(images)
%         im = list_images{images(j)};
%         pos = position(j,:);
%         id(j)
%         figure(1)
%         imshow(imresize(im,2,'bilinear')); hold on;
%         plot(pos(2),pos(1), plot_type(mod(i,5)+1,:), 'LineWidth',2);hold off;
%         pause(0.5);
%     end
%     
%     disp('change id');
% end











