%% validation for dynamic feature (dynamic feature computing is done before)

for n=1:size(CPairs,1)
%     %MSCR
%     mscr_y = distances_y_cam{n};
%     mscr_color = distances_color_cam{n};

    % wHSV
    wHSV = final_dist_hist{n};
    
    for i = 1:size(wHSV,1)
%        mscr_y(i,:) =  mscr_y(i,:)./max(mscr_y(i,:));
%        mscr_color(i,:) =  mscr_color(i,:)./max(mscr_color(i,:));
%        final_dist(i,:) = (pyy*mscr_y(i,:) + pcc*mscr_color(i,:)) +  wHSV(i,:); 
         final_dist(i,:) = wHSV(i,:); 
        final_dist(i,:) = final_dist(i,:)./max(final_dist(i,:));

    end
    
%     for i = 1:size(wHSV,2)
%         final_dist(:,i) = wHSV(:,i); 
%         final_dist(:,i) = final_dist(:,i)./max(final_dist(:,i));
%     end
    
    final_distances{n} = final_dist;
    
    clear final_mscr_y final_mscr_color final_dist wHSV
end


%% Sort ids

Id = Id_hist;

CPairs = combnk(cameras,2);

pairwise_sim_synchro = cell(ncameras, ncameras);

cpt = 0;
for m=1:ncameras-1
    
    for n=m+1:ncameras

        cpt = cpt+1;
        % ID
        l = Id{m};
        c = Id{n};
        
        % Track info
        l_begin = Beginning{m};
        c_begin = Beginning{n};
        l_end = Ending{m};
        c_end = Ending{n};
        
        ind_l = 1:length(l);
        ind_c = 1:length(c);
        
        if(length(l)<length(c))
            a = c;
            b = [l zeros(1,(length(c)-length(l)))];
        else
            a = l;
            b = [c zeros(1,(length(l)-length(c)))];
        end
        % Find identical values in l and c, that is to say the ids present
        % in both cameras m and n
        A = a; indices_a = []; indices_b = [];
        for i = 1:length(l)
            both = find(A==b);
            
            % Limited interval : 1-length(l)
            lim = fix((both+(i-1)-1)/length(A));%i-1
            lim = lim*length(l);
            
            indices_a = [indices_a (both+(i-1)-lim)];%i-1
            indices_b = [indices_b both];
            A = circshift(A',-1)';
        end
        indices_a = sort(indices_a);
        indices_b = sort(indices_b);
        
        if(length(l)<length(c))
            indice_c = indices_a;
            indice_l = indices_b;
        else
            indice_c = indices_b;
            indice_l = indices_a;
        end
        
        % Find other ids no present in each cam
        else_l = []; else_c = [];
        for i = 1:length(l)
            ll = find(ind_l(i) == indice_l);
            if isempty(ll)
                else_l = [else_l ind_l(i)];
            end
        end
        for i = 1:length(c)
            cc = find(ind_c(i) == indice_c);
            if isempty(cc)
                else_c = [else_c ind_c(i)];
            end
        end

        ind_l = [indice_l else_l];
        ind_c = [indice_c else_c];
        
        clear indice_l indice_c else_l else_c
        
        l = l(ind_l);
        c = c(ind_c);
        
        rang_l{cpt} = ind_l;
        rang_c{cpt} = ind_c;
        
        % ID ordered
        id_l{cpt} = l;
        id_c{cpt} = c;
        
        %record cam order
        cam_l{cpt} = m;
        cam_c{cpt} = n;
        
        % Begin and end ordered
        Frame_beginning_l{cpt} = l_begin(ind_l);
        Frame_beginning_c{cpt} = c_begin(ind_c);
        Frame_ending_l{cpt} = l_end(ind_l);
        Frame_ending_c{cpt} = c_end(ind_c);
        
        clear c ind_c ind_l l l_begin c_begin l_end c_end
    end
end


for n = 1:size(CPairs,1)%ncameras
    
    line_n = find(CPairs(:,1) == cam_l{n} & CPairs(:,2) == cam_c{n});
    
    final_distTMP = final_distances{line_n};
    
    row = rang_l{n};
    col = rang_c{n};
    
    final_dist_sorted =  permute(final_distTMP(col,row), [2 1]);
    pairwise_sim_synchro{CPairs(line_n,1),CPairs(line_n,2)} = ones(size(final_dist_sorted)) - final_dist_sorted;
    
    pairwise.begin_l{CPairs(line_n,1),CPairs(line_n,2)} = Frame_beginning_l{n};
    pairwise.begin_c{CPairs(line_n,1),CPairs(line_n,2)} = Frame_beginning_c{n};
    pairwise.end_l{CPairs(line_n,1),CPairs(line_n,2)} = Frame_ending_l{n};
    pairwise.end_c{CPairs(line_n,1),CPairs(line_n,2)} = Frame_ending_c{n};
    
    clear final_distTMP final_dist_sorted
    
end
 
clear Frame_beginning_l Frame_beginning_c Frame_ending_l Frame_ending_c
    
    
