%%_ Apply kmean on our tracklets to keep the k best detections in one _%% 
%%_ tracklet.                                                         _%%

for cam = 1:ncameras

    cpt = 0;
    
    Frame_nb = Frame_number_final{cam};
    Pic_nb = Pic_final{cam};
    identity = identity_final{cam};
    track = tracklets_final{cam};
    histR = histR_final{cam};
    histG = histG_final{cam};
    histB = histB_final{cam};
    bbs_track = bbs_final{cam};

    i = 1;
    while (~isempty(identity))
        
        track_curr = track{i};
        id_curr = identity{i};
        frame_curr = Frame_nb{i};
        pic_curr = Pic_nb{i};
        bbs_curr = bbs_track{i};
        hR = histR{i};
        hG = histG{i};
        hB = histB{i};
        
  
        
        
        if (length(id_curr)>= 3) % we consider only tracklets with more than 3 position
            cpt = cpt +1;
            
            % Extract confidence score
            score_ACF = bbs_curr(:,5);
            
            % We take as constraint : RGB histogram and confidence score
            data =  [hR hB hG score_ACF];

            % K-means
            [idx, centroids] = kmeans(data, nb_k_means,'Distance','cityblock');
            medoid = pdist2(data,centroids);
            [~, ind_medoid] = min(medoid,[],1);

            % keep the 3 nearest representation of the centroid
            track_keep{cpt} = track_curr(ind_medoid,:);
            identity_keep{cpt} = id_curr(ind_medoid);
            Frame_nb_keep{cpt} = frame_curr(ind_medoid);
            Pic_nb_keep{cpt} = pic_curr(ind_medoid);
            bbs_track_keep{cpt} = bbs_curr(ind_medoid,:);
            
            histR_keep{cpt} = hR(ind_medoid,:);
            histG_keep{cpt} = hG(ind_medoid,:);
            histB_keep{cpt} = hB(ind_medoid,:);

        else
            % 
        end

        track(i) = [];
        identity(i) = [];
        Frame_nb(i) = [];
        Pic_nb(i) = [];
        bbs_track(i) = [];
        histR(i) = [];
        histG(i) = [];
        histB(i) = [];
        clear track_curr id_curr frame_curr bbs_curr hR hG hB pic_curr

    end

    track_mean{cam} = track_keep;
    identity_mean{cam} = identity_keep;
    bbs_mean{cam} = bbs_track_keep;
    frame_nb_mean{cam} = Frame_nb_keep;
    pic_nb_mean{cam} = Pic_nb_keep;
    histR_mean{cam} = histR_keep;
    histG_mean{cam} = histG_keep;
    histB_mean{cam} = histB_keep;
    
    clear track_keep identity_keep bbs_track_keep Frame_nb_keep histR_keep histG_keep histB_keep Pic_nb_keep
end


%%
% for n = 1:ncameras
%     for o = 1:length(identity_mean{n})
%         figure(1);
%         subplot(1,3,1)
%         imshow(picture{pic_nb_mean{n}{o}(1)});
%         subplot(1,3,2)
%         imshow(picture{pic_nb_mean{n}{o}(2)});
%         subplot(1,3,3)
%         imshow(picture{pic_nb_mean{n}{o}(3)});
%         pause(1)
%         
%     end
% end






