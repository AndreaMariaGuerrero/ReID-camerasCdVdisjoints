%% Dynamic feature computation
switch dataname
	case {'ETHZ1','ETHZ2','ETHZ3'} 
		clear BlobsDyn
		ii = 1;
		for i = 1:length(dset) % for each ped
			
			% dyn 1
			BlobsDyn(ii).mvec = [];
			BlobsDyn(ii).pvec = [];
			dd = uint16(length(ped(i).dynmodels)/2);
			rnddd = ped(i).rnddd; 
			for j = 1:dd % for each view/2
				ind = find(ped(i).dynmodels(rnddd(j)) == dset(i).globalindex); % we are searching for the local index (in permits_ind)
				BlobsDyn(ii).mvec = [BlobsDyn(ii).mvec,Blobs(dset(i).index(ind)).mvec];
				BlobsDyn(ii).pvec = [BlobsDyn(ii).pvec,Blobs(dset(i).index(ind)).pvec];
			end
			ii = ii+1;
			
			% dyn 2
			BlobsDyn(ii).mvec = [];
			BlobsDyn(ii).pvec = [];
			for j = dd+1:length(ped(i).dynmodels) % for each view/2
				ind = find(ped(i).dynmodels(rnddd(j)) == dset(i).globalindex); % we are searching for the local index (in permits_ind)
				BlobsDyn(ii).mvec = [BlobsDyn(ii).mvec,Blobs(dset(i).index(ind)).mvec];
				BlobsDyn(ii).pvec = [BlobsDyn(ii).pvec,Blobs(dset(i).index(ind)).pvec];
			end
			ii = ii+1;
			
		end
	case {'iLIDS','fablab'} 
		clear BlobsDyn
		ii = 1;
		for i = 1:length(dset) % for each ped
			
			% dyn 1
			BlobsDyn(ii).mvec = [];
			BlobsDyn(ii).pvec = [];
			dd = uint16(length(ped(i).dynmodels)/2);
			rnddd = ped(i).rnddd; 
			for j = 1:dd % for each view/2
				ind = find(ped(i).dynmodels(rnddd(j)) == dset(i).globalindex); % we are searching for the local index (in permits_ind)
				BlobsDyn(ii).mvec = [BlobsDyn(ii).mvec,Blobs(dset(i).index(ind)).mvec];
				BlobsDyn(ii).pvec = [BlobsDyn(ii).pvec,Blobs(dset(i).index(ind)).pvec];
			end
			ii = ii+1;
			
			% dyn 2
			BlobsDyn(ii).mvec = [];
			BlobsDyn(ii).pvec = [];
			for j = dd+1:length(ped(i).dynmodels) % for each view/2
				ind = find(ped(i).dynmodels(rnddd(j)) == dset(i).globalindex); % we are searching for the local index (in permits_ind)
				BlobsDyn(ii).mvec = [BlobsDyn(ii).mvec,Blobs(dset(i).index(ind)).mvec];
				BlobsDyn(ii).pvec = [BlobsDyn(ii).pvec,Blobs(dset(i).index(ind)).pvec];
			end
			ii = ii+1;	
        end
    case {'camnet'} %%% MODIF
		clear BlobsDyn Id Nb_pic

        for i = 1:ncameras
            % Init
            BlobsDyn{i} = [];
            Id{i} = [];
            Nb_pic{i} = [];
                
            index = false(1, length(ped)); 
            for k=1:length(ped)
                % We sort our data by cam
                index(k) = (ped(k).cam == i);
            end
            ped_selec{i} = ped(index);
            num_index = find(index == 1);
            
            % Init
            cpt = 1;
            BlobsDyn_tmp{cpt}.mvec = [];
            BlobsDyn_tmp{cpt}.pvec = [];
            
            for k = 1:length(ped_selec{i}) % for 1 cam
                for j = 1:nb_k_means
                    ind = find(ped(num_index(k)).dynmodels(j) == dset( ped(num_index(k)).id).globalindex); % we are searching for the local index (in permits_ind)
                    BlobsDyn_tmp{cpt}.mvec = Blobs( dset(ped(num_index(k)).id).index(ind) ).mvec;
                    BlobsDyn_tmp{cpt}.pvec = Blobs( dset(ped(num_index(k)).id).index(ind) ).pvec;
                    cpt = cpt + 1;
                    
                %Pic nb info for graphic representation
                Nb_pic{i} = [ Nb_pic{i}, ped(num_index(k)).pic(j)];               
                end
                
                % Add id info
                Id{i}= [Id{i},  dset(ped(num_index(k)).id).id];
                

            end
            
            BlobsDyn{i} = [BlobsDyn{i}, BlobsDyn_tmp]; % size :"33 33 39" : we take 3 detections for every id
            
            clear BlobsDyn_tmp ind index ped_selec
        end
end

%% MSCR distances computation

hh = waitbar(0,'Image features computation...');
bestk   = zeros(length(BlobsDyn),1);
bestmu  = zeros(5,length(BlobsDyn));
bestcov = zeros(5,5,length(BlobsDyn));
cpt = 0;
for i=1:ncameras
    for ii=1:length(BlobsDyn{i})
        cpt = cpt+1;
        %%---- MSE analysis
        mser    =   BlobsDyn{i}{ii};

        [keep_list,BB]= blobs_inside_image(mser.mvec,[H,W]);% used to compute BB
        keep_list = 1:size(mser.mvec,2);
        
        
        for j=1:length(keep_list);
            centroid(:,j) = BB(1:2,1,j)./[W;H];
            colour(:,j) = applycform(mser.pvec(:,j)', reg);
        end

        % blobs clustering  for dynamic feature!
        data = [centroid;colour];
        [cl{i}.bestk{ii},bestpp,cl{i}.bestmu{ii},cl{i}.bestcov{ii},dl,countf] = mixtures4(data,kmin,min(kmax,size(data,2)),regularize,th,covoption);
        num_data = size(data,2);
        cl{i}.label{ii}	 = zeros(num_data,1);
        ll		 = zeros(num_data,1);
        for j = 1:num_data
            vote = zeros(cl{i}.bestk{ii},1);
            for k=1:cl{i}.bestk{ii}
                vote(k)= mvnpdf(data(:,j),cl{i}.bestmu{ii}(:,k),abs(cl{i}.bestcov{ii}(:,:,k)));
            end
            [ll(j), cl{i}.label{ii}(j)] = max(vote);
        end

        dataf_tmp{ii}.Mmvec=centroid;
        dataf_tmp{ii}.Mpvec=colour;

        clear centroid colour
        
        waitbar(ii/length(BlobsDyn{i}),hh);
    end
    dataf{i} = dataf_tmp;
    clear dataf_tmp
end
close(hh);

%%

%Camera combinaisons
CPairs = combnk(cameras,2);
numCPairs = size(CPairs, 1);


disp('Distances computation MSCR...');
for n = 1:numCPairs
    
    % We consider data of a camera pair
    vec_cam1 = dataf{CPairs(n,1)};
    vec_cam2 = dataf{CPairs(n,2)};
    
    for ii = 1:length(vec_cam2)/nb_k_means
        % We have three detections for one id
        i = (ii-1)*nb_k_means+1:ii*nb_k_means;
        
        Identities(1) = Id{CPairs(n,2)}(ii);
        
        
        for jj = 1:length(vec_cam1)/nb_k_means
            % We have three detections for one id
            j = (jj-1)*nb_k_means+1:jj*nb_k_means;

            Identities(2) = Id{CPairs(n,1)}(jj);
            
            %Signature combinaisons du to k-means
            Combi_kmeans = perms(j);
            
            for combi = 1:size(Combi_kmeans,1)
                
                j_combi = Combi_kmeans(combi,:);
                
                for x = 1:nb_k_means
                    
                    Mmvec{1} = vec_cam2{i(x)}.Mmvec;
                    Mpvec{1} = vec_cam2{i(x)}.Mpvec;
                    num(1)= size(Mmvec{1},2);
                    
                    Mmvec{2} = vec_cam1{j_combi(x)}.Mmvec;
                    Mpvec{2} = vec_cam1{j_combi(x)}.Mpvec;
                    num(2)= size(Mmvec{2},2);

                    dist_y_n	= cell(length(vec_cam1)/nb_k_means,1);
                    dist_color_n= cell(length(vec_cam1)/nb_k_means,1);

                    % -- Calcul distances -- %
                    [thrash,max_ind]= max([num(1),num(2)]);
                    max_i = mod(max_ind,2)+1;
                    min_i = max_ind;
                    max_info = num(max_i);
                    min_info = num(min_i);

                    dist_y = 1000*ones(min_info, max_info);
                    dist_color = 1000*ones(min_info, max_info);


                    for k=1:max_info % smallest
                        % cluster selection
                        vote = zeros(cl{CPairs(n,2)}.bestk{i(x)},1);
                        for kl = 1:cl{CPairs(n,2)}.bestk{i(x)}
                            vote(kl)= mvnpdf([Mmvec{max_i}(:,k);Mpvec{max_i}(:,k)],cl{CPairs(n,2)}.bestmu{i(x)}(:,kl),abs(cl{CPairs(n,2)}.bestcov{i(x)}(:,:,kl)));
                        end
                        [trash, NNind] = max(vote); % Nearest Neighbor data association
                        bl_into_cl = find(cl{CPairs(n,2)}.label{i(x)} == NNind); % blobs into selected cluster

                        for h=1:length(bl_into_cl) % biggest						
                            dist_y(bl_into_cl(h),k) = abs(Mmvec{max_i}(2,k)-Mmvec{min_i}(2,bl_into_cl(h)));
                            dist_color(bl_into_cl(h),k) = sqrt(sum((Mpvec{max_i}(:,k)-Mpvec{min_i}(:,bl_into_cl(h))).^2));
                        end
                    end

                    % check outliers
                    %Position
                    ref_y = min(dist_y); me_ref_y = mean(ref_y); std_ref_y = std(ref_y);
                    %Colour
                    ref_color = min(dist_color); me_ref_color = mean(ref_color); std_ref_color = std(ref_color);
                    %Keep list
                    good = find((ref_y<=(me_ref_y+3.5*std_ref_y))&(ref_color<=(me_ref_color+3.5*std_ref_color))); 

                    max_useful_info = length(good);
                    dist_y2 = dist_y(:,good);
                    dist_color2 = dist_color(:,good);

                    %%normalize
                    dist_y_n = dist_y./max(dist_y2(:));
                    dist_color_n = dist_color./max(dist_color2(:));

                    %%distance computation
                    totdist_n = ( pyy*dist_y_n(:,good)  + pcc*dist_color_n(:,good) ) ;

                    %%Minimization
                    [min_dist,matching] = min(totdist_n);

                    useful_i = sub2ind(size(totdist_n),[matching]',[1:max_useful_info]');
                    
                    % -- Stock to compare all combinaisons -- %
                    final_dist_y_tmp(x,combi)  = sum(dist_y2(useful_i))/max_useful_info;
                    final_dist_color_tmp(x,combi)  = sum(dist_color2(useful_i))/max_useful_info;
                    
                    clear dist_y dist_color;
                    clear dist_y_n dist_color_n;
                end
                
                
            end
            
%         ind_pic1 = Nb_pic{CPairs(n,1)}(j);
%         ind_pic2 = Nb_pic{CPairs(n,2)}(i);
%         
%         mask_f1 = mask_fin(:,:,ind_pic1);
%         mask_f2 = mask_fin(:,:,ind_pic2);
% 
%         subplot(4,3,1), imshow(picture{ind_pic1(1)});
%         subplot(4,3,2), imshow(picture{ind_pic1(2)});
%         subplot(4,3,3), imshow(picture{ind_pic1(3)});
%         subplot(4,3,4), imshow(mask_f1(:,:,1));
%         subplot(4,3,5), imshow(mask_f1(:,:,2));
%         subplot(4,3,6), imshow(mask_f1(:,:,3));
%         
%         subplot(4,3,7), imshow(picture{ind_pic2(1)});
%         subplot(4,3,8), imshow(picture{ind_pic2(2)});
%         subplot(4,3,9), imshow(picture{ind_pic2(3)});
%         subplot(4,3,10), imshow(mask_f2(:,:,1));
%         subplot(4,3,11), imshow(mask_f2(:,:,2));
%         subplot(4,3,12), imshow(mask_f2(:,:,3));
%         pause(1)
       
        
            %%%% AJOUT2
            final_dist_y_tmp = final_dist_y_tmp;%./max(max(final_dist_y_tmp));
            final_dist_color_tmp = final_dist_color_tmp;%./max(max(final_dist_color_tmp));
            final_dist_mscr_tmp = pyy*final_dist_y_tmp +  pcc*final_dist_color_tmp;
            
            
            [~, ind_min_dist] = min(final_dist_mscr_tmp(:));
            [r_min,c_min] = ind2sub(size(final_dist_mscr_tmp),ind_min_dist);
            % We take min of 3 detections
            final_dist_y(ii,jj)  = final_dist_y_tmp(r_min,c_min);
            final_dist_color(ii,jj) = final_dist_color_tmp(r_min,c_min);

            %min of sum 3 detect
%             [~, ind_min_dist] = min(sum(final_dist_mscr_tmp,2));
%             final_dist_y(ii,jj)  = sum(final_dist_y_tmp(:,ind_min_dist));
%             final_dist_color(ii,jj) = sum(final_dist_color_tmp(:,ind_min_dist));
            
            %%%%
            clear final_dist_y_tmp final_dist_color_tmp;
            % Save ids
            id1(ii,jj) = Identities(1);
            id2(ii,jj) = Identities(2);
		
        end
        
        
%         final_dist_y(ii,:) =  final_dist_y(ii,:)./max(final_dist_y(ii,:));
%         final_dist_color(ii,:) = final_dist_color(ii,:)./max(final_dist_color(ii,:));
        final_mscr_dist(ii,:) = (pyy*final_dist_y(ii,:) +  pcc*final_dist_color(ii,:) ); 
        
        
    end
    
    Id1{n} = id1;
    Id2{n} = id2;
    
    distances_y_cam{n} = final_dist_y;
    distances_color_cam{n} = final_dist_color;
    distances_mscr{n} = final_mscr_dist;

    clear final_dist_y final_dist_color final_mscr_dist id1 id2;
    
end
